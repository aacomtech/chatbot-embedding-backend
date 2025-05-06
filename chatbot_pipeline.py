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

# --- Basic Auth setup for protected endpoints ---
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

# --- SQLite setup with persistent storage ---
storage_dir = os.getenv("STORAGE_DIR", "/opt/render/project/src/storage")
os.makedirs(storage_dir, exist_ok=True)
DB_PATH = os.path.join(storage_dir, "chatbot_data.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

# Create or migrate domains table
c.execute(
    '''CREATE TABLE IF NOT EXISTS domains (
         domain TEXT PRIMARY KEY,
         index_blob BLOB,
         chunks_blob BLOB
    )'''
)
# Add urls_blob column if missing
cols = [row[1] for row in c.execute("PRAGMA table_info(domains)")]
if 'urls_blob' not in cols:
    c.execute("ALTER TABLE domains ADD COLUMN urls_blob BLOB")

# Create queries log table
c.execute(
    '''CREATE TABLE IF NOT EXISTS queries (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         domain TEXT,
         question TEXT,
         answer TEXT,
         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )'''
)
conn.commit()

# --- In-memory stores ---
index_store = {}
chunks_store = {}
urls_store = {}

# Load persisted indices and URL lists on startup
for domain, ib, cb, ub in c.execute("SELECT domain, index_blob, chunks_blob, urls_blob FROM domains"):
    try:
        index_store[domain] = pickle.loads(ib)
        chunks_store[domain] = pickle.loads(cb)
        urls_store[domain] = pickle.loads(ub) if ub else []
    except Exception:
        continue

# --- FastAPI setup ---
app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    except Exception:
        return [base_url]

# --- Protected endpoint: create-chatbot ---
@app.post("/create-chatbot")
async def create_chatbot(req: DomainRequest, user: str = Depends(get_current_user)):
    dom = normalize(req.domain)
    base_url = f"https://{dom}"

    # Determine URLs to crawl
    if dom in urls_store and urls_store[dom]:
        existing = urls_store[dom]
        desired = len(existing) + 10
        all_links = fetch_internal_links(base_url, max_links=desired)
        new_links = [u for u in all_links if u not in existing]
        urls = existing + new_links
    else:
        urls = fetch_internal_links(base_url, max_links=20)
    urls_store[dom] = urls

    # Build or update FAISS index
    if dom not in index_store:
        idx = faiss.IndexFlatL2(1536)
        chunks = []
    else:
        idx = index_store[dom]
        chunks = chunks_store[dom]

    # Crawl only new URLs
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

    # Update in-memory stores
    index_store[dom] = idx
    chunks_store[dom] = chunks

    # Persist to SQLite
    blob_idx = pickle.dumps(idx)
    blob_chunks = pickle.dumps(chunks)
    blob_urls = pickle.dumps(urls)
    c.execute(
        "INSERT OR REPLACE INTO domains (domain, index_blob, chunks_blob, urls_blob) VALUES (?, ?, ?, ?)",
        (dom, blob_idx, blob_chunks, blob_urls)
    )
    conn.commit()

    return {
        "chatbot_url": f"https://chatbot-frontend-zeta-tawny.vercel.app/{dom.replace('.', '-')}",
        "indexed": True,
        "fetched_urls": urls
    }
