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

os.makedirs(storage_dir, exist_ok=True)
DB_PATH = os.path.join(storage_dir, "chatbot_data.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
# Create or migrate domains table
c.execute('''CREATE TABLE IF NOT EXISTS domains (
    domain TEXT PRIMARY KEY,
    index_blob BLOB,
    chunks_blob BLOB,
    urls_blob BLOB
)''')
# Ensure urls_blob exists
cols = [row[1] for row in c.execute("PRAGMA table_info(domains)")]
if 'urls_blob' not in cols:
    c.execute("ALTER TABLE domains ADD COLUMN urls_blob BLOB")
# Create queries table
c.execute('''CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT,
    question TEXT,
    answer TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)''')
conn.commit()

# --- In-memory stores ---
index_store = {}
chunks_store = {}
urls_store = {}

# Preload persisted data
for domain, ib, cb, ub in c.execute("SELECT domain, index_blob, chunks_blob, urls_blob FROM domains"):
    try:
        index_store[domain] = pickle.loads(ib)
        chunks_store[domain] = pickle.loads(cb)
        urls_store[domain] = pickle.loads(ub) if ub else []
    except:
        continue

# --- FastAPI setup ---
app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
    protected = ["/create-chatbot", "/ask", "/domains", "/domains/{domain}/info", "/queries", "/queries/{domain}"]
    for path in schema.get("paths", {}):
        if path in protected:
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
    return domain.replace("https://", "").replace("http://", "").replace("www.", "").strip("/")

def fetch_internal_links(base_url: str, max_links: int) -> list[str]:
    try:
        resp = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        links = {base_url}
        netloc = urlparse(base_url).netloc
        for tag in soup.find_all("a", href=True):
            href = urljoin(base_url, tag["href"])
            p = urlparse(href)
            if p.netloc == netloc and p.scheme.startswith("http"):
                links.add(href)
            if len(links) >= max_links:
                break
        return list(links)
    except:
        return [base_url]

# --- Protected endpoints ---
@app.post("/create-chatbot")
async def create_chatbot(req: DomainRequest, user: str = Depends(get_current_user)):
    dom = normalize(req.domain)
    base_url = f"https://{dom}"

    # Determine URLs to crawl
    existing = urls_store.get(dom, [])
    if existing:
        desired = len(existing) + 10
        all_links = fetch_internal_links(base_url, max_links=desired)
        new_links = [u for u in all_links if u not in existing]
        urls = existing + new_links
    else:
        urls = fetch_internal_links(base_url, max_links=20)
    urls_store[dom] = urls

    # Build or update index
    if dom in index_store:
        idx = index_store[dom]
        chunks = chunks_store[dom]
    else:
        idx = faiss.IndexFlatL2(1536)
        chunks = []

    start = len(chunks)
    for url in urls[start:]:
        raw = trafilatura.fetch_url(url)
        text = trafilatura.extract(raw) if raw else None
        if not text:
            continue
        for i in range(0, len(text), 1000):
            chunk = text[i:i+1000]
            emb = openai.embeddings.create(input=chunk, model="text-embedding-3-small").data[0].embedding
            idx.add(np.array([emb]).astype('float32'))
            chunks.append(chunk)

    index_store[dom] = idx
    chunks_store[dom] = chunks

    # Persist
    c.execute(
        "INSERT OR REPLACE INTO domains (domain, index_blob, chunks_blob, urls_blob) VALUES (?, ?, ?, ?)",
        (dom, pickle.dumps(idx), pickle.dumps(chunks), pickle.dumps(urls))
    )
    conn.commit()

    return {"chatbot_url": f"https://chatbot-frontend-zeta-tawny.vercel.app/{dom.replace('.', '-')}", "indexed": True, "fetched_urls": urls}

@app.post("/ask")
async def ask_bot(req: QueryRequest, user: str = Depends(get_current_user)):
    dom = normalize(req.domain)
    if dom not in index_store:
        row = c.execute("SELECT index_blob, chunks_blob, urls_blob FROM domains WHERE domain=?", (dom,)).fetchone()
        if row:
            index_store[dom] = pickle.loads(row[0])
            chunks_store[dom] = pickle.loads(row[1])
            urls_store[dom] = pickle.loads(row[2])
    idx = index_store.get(dom)
    chunks = chunks_store.get(dom, [])
    if not idx or not chunks:
        return {"answer": "No content indexed yet. Create chatbot first."}
    emb = openai.embeddings.create(input=req.question, model="text-embedding-3-small").data[0].embedding
    D, I = idx.search(np.array([emb]).astype('float32'), k=3)
    sel = [chunks[i] for i in I[0] if i < len(chunks)]
    context = "\n---\n".join(sel)
    prompt = f"Answer based on context:\n{context}\nQ: {req.question}"
    resp = openai.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    ans = resp.choices[0].message.content.strip()
    c.execute("INSERT INTO queries (domain, question, answer) VALUES (?, ?, ?)", (dom, req.question, ans))
    conn.commit()
    return {"answer": ans}

# --- Client proxy endpoints ---
@app.post("/client/create-chatbot")
async def client_create_chatbot(req: DomainRequest):
    return await create_chatbot(req, user=USER)

@app.post("/client/ask")
async def client_ask(req: QueryRequest):
    return await ask_bot(req, user=USER)

# --- Admin endpoints ---
@app.get("/domains")
async def list_domains(user: str = Depends(get_current_user)):
    return {"domains": list(urls_store.keys())}

@app.get("/domains/{domain}/info")
async def domain_info(domain: str, user: str = Depends(get_current_user)):
    dom = normalize(domain)
    return {"fetched_urls": urls_store.get(dom, [])}

@app.get("/queries")
async def list_queries(user: str = Depends(get_current_user)):
    rows = c.execute("SELECT id, domain, question, answer, timestamp FROM queries").fetchall()
    return rows

@app.get("/queries/{domain}")
async def queries_for(domain: str, user: str = Depends(get_current_user)):
    dom = normalize(domain)
    rows = c.execute("SELECT id, question, answer, timestamp FROM queries WHERE domain=?", (dom,)).fetchall()
    return rows
