# chatbot_pipeline.py
import os
import sqlite3
import pickle
import trafilatura
import openai
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi

# --- Basic Auth setup ---
security = HTTPBasic()
USER = os.getenv("API_USER", "admin")
PASS = os.getenv("API_PASS", "6434e108a8efccf2e8629862b70af80f")

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != USER or credentials.password != PASS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- Environment & API key ---
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OpenAI API key. Make sure OPENAI_API_KEY is set.")

# --- SQLite setup ---
storage_dir = os.getenv("STORAGE_DIR", "/opt/render/project/src/storage")
if not os.path.isdir(storage_dir):
    os.makedirs(storage_dir)
DB_PATH = os.path.join(storage_dir, "chatbot_data.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
# Create domains table
c.execute('''
CREATE TABLE IF NOT EXISTS domains (
    domain TEXT PRIMARY KEY,
    index_blob BLOB,
    chunks_blob BLOB,
    urls_blob BLOB,
    chunk_url_map_blob BLOB
)''')
# Ensure urls_blob and chunk_url_map_blob exist for migrations
cols = [row[1] for row in c.execute("PRAGMA table_info(domains)")]
if 'urls_blob' not in cols:
    c.execute("ALTER TABLE domains ADD COLUMN urls_blob BLOB")
if 'chunk_url_map_blob' not in cols:
    c.execute("ALTER TABLE domains ADD COLUMN chunk_url_map_blob BLOB")
conn.commit()

# --- In-memory stores ---
index_store = {}
chunks_store = {}
urls_store = {}
chunk_url_map_store = {}

# Preload persisted data
for domain, ib, cb, ub, cub in c.execute(
    "SELECT domain,index_blob,chunks_blob,urls_blob,chunk_url_map_blob FROM domains"
):
    try:
        index_store[domain] = pickle.loads(ib)
        chunks_store[domain] = pickle.loads(cb)
        urls_store[domain] = pickle.loads(ub)
        chunk_url_map_store[domain] = pickle.loads(cub)
    except Exception:
        continue

# --- FastAPI setup ---
app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #https://chatbot-frontend-zeta-tawny.vercel.app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom OpenAPI security

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="Chatbot Embedding API",
        version="1.0.0",
        description="API for chatbot management and queries",
        routes=app.routes,
    )
    schema["components"]["securitySchemes"] = {"basicAuth": {"type": "http", "scheme": "basic"}}
    protected = ["/create-chatbot","/ask","/domains","/domains/{domain}/info","/queries","/queries/{domain}"]
    for path in protected:
        if path in schema["paths"]:
            for method in schema["paths"][path]:
                schema["paths"][path][method]["security"] = [{"basicAuth": []}]
    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi

# --- Models ---
class DomainRequest(BaseModel):
    domain: str

class QueryRequest(BaseModel):
    domain: str
    question: str

# --- Utility functions ---
def normalize(domain: str) -> str:
    clean = domain.strip().lower()
    for prefix in ("https://", "http://", "www."):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
    return clean.rstrip("/")


def fetch_internal_links(base_url: str, max_links: int) -> list[str]:
    links = {base_url}
    try:
        resp = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        netloc = urlparse(base_url).netloc
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            p = urlparse(href)
            if p.scheme in ("http", "https") and p.netloc == netloc:
                links.add(href)
            if len(links) >= max_links:
                break
    except:
        pass
    return list(links)

# --- Endpoints ---
@app.post("/create-chatbot")
async def create_chatbot(req: DomainRequest, user: str = Depends(get_current_user)):
    dom = normalize(req.domain)
    base_url = f"https://{dom}"
    existing = urls_store.get(dom, [])
    if existing:
        desired = len(existing) + 10
        all_links = fetch_internal_links(base_url, desired)
        urls = existing + [u for u in all_links if u not in existing]
    else:
        urls = fetch_internal_links(base_url, 20)
    urls_store[dom] = urls

    idx = index_store.get(dom, faiss.IndexFlatL2(1536))
    chunks = chunks_store.get(dom, [])
    mapping = chunk_url_map_store.get(dom, [])

    for url in urls[len(chunks):]:
        raw = trafilatura.fetch_url(url)
        text = trafilatura.extract(raw) if raw else ""
        parts = [text[i:i+1000] for i in range(0, len(text), 1000)]
        for part in parts:
            emb = openai.embeddings.create(input=part, model="text-embedding-3-small").data[0].embedding
            idx.add(np.array([emb], dtype='float32'))
            chunks.append(part)
            mapping.append(url)

    index_store[dom] = idx
    chunks_store[dom] = chunks
    chunk_url_map_store[dom] = mapping

    c.execute(
        "INSERT OR REPLACE INTO domains(domain,index_blob,chunks_blob,urls_blob,chunk_url_map_blob) VALUES(?,?,?,?,?)",
        (dom, pickle.dumps(idx), pickle.dumps(chunks), pickle.dumps(urls), pickle.dumps(mapping))
    )
    conn.commit()

    return {"chatbot_url": f"https://chatbot-frontend-zeta-tawny.vercel.app/{dom}", "indexed": True, "fetched_urls": urls}

@app.post("/ask")
async def ask_bot(req: QueryRequest, user: str = Depends(get_current_user)):
    dom = normalize(req.domain)
    if dom not in index_store:
        row = c.execute(
            "SELECT index_blob,chunks_blob,urls_blob,chunk_url_map_blob FROM domains WHERE domain=?", (dom,)
        ).fetchone()
        if row:
            index_store[dom] = pickle.loads(row[0])
            chunks_store[dom] = pickle.loads(row[1])
            urls_store[dom] = pickle.loads(row[2])
            chunk_url_map_store[dom] = pickle.loads(row[3])

    idx = index_store.get(dom)
    chunks = chunks_store.get(dom, [])
    mapping = chunk_url_map_store.get(dom, [])

    if not idx or not chunks:
        return {"answer": "No content indexed yet. Create chatbot first."}

    emb = openai.embeddings.create(input=req.question, model="text-embedding-3-small").data[0].embedding
    D, I = idx.search(np.array([emb], dtype='float32'), k=3)
    sel = [chunks[i] for i in I[0] if i < len(chunks)]
    context = "\n---\n".join(sel)
    prompt = f"Answer based on context:\n{context}\nQ: {req.question}"
    resp = openai.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    ans = resp.choices[0].message.content.strip()

    sources = [mapping[i] for i in I[0] if i < len(mapping)]
    sources = list(dict.fromkeys(sources))

    c.execute(
        "INSERT INTO queries(domain,question,answer,sources_blob) VALUES(?,?,?,?)",
        (dom, req.question, ans, pickle.dumps(sources))
    )
    conn.commit()
    return {"answer": ans, "sources": sources}

# Public proxy
@app.post( "/client/create-chatbot" )
async def client_create(req: DomainRequest):
    return await create_chatbot(req, user=USER)

@app.post( "/client/ask" )
async def client_ask(req: QueryRequest):
    return await ask_bot(req, user=USER)

@app.get( "/client/domains/{domain}/info" )
async def client_info(domain: str):
    return {"fetched_urls": urls_store.get(normalize(domain), [])}

# Admin endpoints
@app.get( "/domains" )
async def list_domains(user: str = Depends(get_current_user)):
    return {"domains": list(urls_store.keys())}

@app.get( "/domains/{domain}/info" )
async def domain_info(domain: str, user: str = Depends(get_current_user)):
    return {"fetched_urls": urls_store.get(normalize(domain), [])}

@app.get( "/queries" )
async def list_queries(user: str = Depends(get_current_user)):
    rows = c.execute("SELECT id, domain, question, answer, timestamp FROM queries").fetchall()
    return rows

@app.get( "/queries/{domain}" )
async def queries_for(domain: str, user: str = Depends(get_current_user)):
    rows = c.execute(
        "SELECT id, question, answer, timestamp FROM queries WHERE domain=?", (normalize(domain),)
    ).fetchall()
    return rows
