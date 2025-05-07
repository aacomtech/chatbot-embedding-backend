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
    by_url_blob BLOB
)''')
# Create queries table
c.execute('''
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT,
    question TEXT,
    answer TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)''')
conn.commit()

# --- In-memory stores ---
index_store = {}
flat_chunks_store = {}
urls_store = {}
by_url_store = {}

# Preload persisted data
for domain, ib, cb, ub, bub in c.execute(
    "SELECT domain, index_blob, chunks_blob, urls_blob, by_url_blob FROM domains"
):
    try:
        index_store[domain] = pickle.loads(ib)
        flat_chunks_store[domain] = pickle.loads(cb)
        urls_store[domain] = pickle.loads(ub)
        by_url_store[domain] = pickle.loads(bub)
    except Exception:
        continue

# --- FastAPI setup ---
app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")
# CORS middleware - allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chatbot-frontend-zeta-tawny.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAPI security
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
    protected = ["/create-chatbot", "/ask", "/domains", "/domains/{domain}/info", "/queries", "/queries/{domain}"]
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
    return domain.strip().lower().removeprefix("https://").removeprefix("http://").removeprefix("www.").rstrip("/")

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
# Protected: create chatbot
@app.post("/create-chatbot")
async def create_chatbot(req: DomainRequest, user: str = Depends(get_current_user)):
    dom = normalize(req.domain)
    base_url = f"https://{dom}"
    existing = urls_store.get(dom, [])
    if existing:
        desired = len(existing) + 10
        all_links = fetch_internal_links(base_url, desired)
        new = [u for u in all_links if u not in existing]
        urls = existing + new
    else:
        urls = fetch_internal_links(base_url, 20)
    urls_store[dom] = urls

    # index and chunks
    idx = index_store.get(dom, faiss.IndexFlatL2(1536))
    chunks = flat_chunks_store.get(dom, [])
    by_url = by_url_store.get(dom, {})

    start = len(chunks)
    for url in urls[start:]:
        raw = trafilatura.fetch_url(url)
        text = trafilatura.extract(raw) if raw else ""
        parts = [text[i:i+1000] for i in range(0, len(text), 1000)]
        by_url[url] = parts
        for part in parts:
            emb = openai.embeddings.create(input=part, model="text-embedding-3-small").data[0].embedding
            idx.add(np.array([emb], dtype='float32'))
            chunks.append(part)

    index_store[dom] = idx
    flat_chunks_store[dom] = chunks
    by_url_store[dom] = by_url

    # persist
    c.execute(
        "INSERT OR REPLACE INTO domains (domain,index_blob,chunks_blob,urls_blob,by_url_blob) VALUES (?,?,?,?,?)",
        (dom, pickle.dumps(idx), pickle.dumps(chunks), pickle.dumps(urls), pickle.dumps(by_url))
    )
    conn.commit()

    return {"chatbot_url": f"https://chatbot-frontend-zeta-tawny.vercel.app/{dom}", "indexed": True, "fetched_urls": urls}

# Protected: ask chatbot
@app.post("/ask")
async def ask_bot(req: QueryRequest, user: str = Depends(get_current_user)):
    dom = normalize(req.domain)
    if dom not in index_store:
        row = c.execute("SELECT index_blob,chunks_blob,urls_blob,by_url_blob FROM domains WHERE domain=?", (dom,)).fetchone()
        if row:
            index_store[dom] = pickle.loads(row[0])
            flat_chunks_store[dom] = pickle.loads(row[1])
            urls_store[dom] = pickle.loads(row[2])
            by_url_store[dom] = pickle.loads(row[3])
    idx = index_store.get(dom)
    chunks = flat_chunks_store.get(dom, [])
    by_url = by_url_store.get(dom, {})
    if not idx or not chunks:
        return {"answer": "No content indexed yet. Create chatbot first."}

    emb = openai.embeddings.create(input=req.question, model="text-embedding-3-small").data[0].embedding
    D, I = idx.search(np.array([emb], dtype='float32'), k=3)
    sel = [chunks[i] for i in I[0] if i < len(chunks)]
    context = "\n---\n".join(sel)
    prompt = f"Answer based on context:\n{context}\nQ: {req.question}"
    resp = openai.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}])
    ans = resp.choices[0].message.content.strip()

    sources = []
    for url, parts in by_url.items():
        if any(chunks[i] in parts for i in I[0] if i < len(chunks)):
            sources.append(url)
    sources = list(dict.fromkeys(sources))

    c.execute("INSERT INTO queries(domain,question,answer) VALUES(?,?,?)", (dom, req.question, ans))
    conn.commit()
    return {"answer": ans, "sources": sources}

# Public proxy endpoints
@app.post("/client/create-chatbot")
async def client_create(req: DomainRequest):
    return await create_chatbot(req, user=USER)

@app.post("/client/ask")
async def client_ask(req: QueryRequest):
    return await ask_bot(req, user=USER)

@app.get("/client/domains/{domain}/info")
async def client_info(domain: str):
    dom = normalize(domain)
    return {"fetched_urls": urls_store.get(dom, [])}

# Admin endpoints
@app.get("/domains")
async def list_domains(user: str = Depends(get_current_user)):
    return {"domains": list(urls_store.keys())}

@app.get("/domains/{domain}/info")
async def domain_info(domain: str, user: str = Depends(get_current_user)):
    dom = normalize(domain)
    return {"fetched_urls": urls_store.get(dom, [])}

@app.get("/queries")
async def list_queries(user: str = Depends(get_current_user)):
    rows = c.execute("SELECT id,domain,question,answer,timestamp FROM queries").fetchall()
    return rows

@app.get("/queries/{domain}")
async def queries_for(domain: str, user: str = Depends(get_current_user)):
    dom = normalize(domain)
    rows = c.execute("SELECT id,question,answer,timestamp FROM queries WHERE domain=?",(dom,)).fetchall()
    return rows
