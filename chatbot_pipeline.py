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
# Create domains table
c.execute(
    '''CREATE TABLE IF NOT EXISTS domains (
         domain TEXT PRIMARY KEY,
         index_blob BLOB,
         chunks_blob BLOB
    )'''
)
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

# --- Load persisted indices at startup ---
index_store = {}
chunks_store = {}
urls_store = {}
for domain, ib, cb in c.execute("SELECT domain, index_blob, chunks_blob FROM domains"):  # noqa
    try:
        index_store[domain] = pickle.loads(ib)
        chunks_store[domain] = pickle.loads(cb)
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

def fetch_internal_links(base_url: str, max_links: int = 20) -> list[str]:
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
    # Short-circuit if already indexed
    if dom in index_store:
        return {
            "chatbot_url": f"https://chatbot-frontend-zeta-tawny.vercel.app/{dom.replace('.', '-')}",
            "indexed": False,
            "message": "Already indexed - using cached version"
        }

    base_url = f"https://{dom}"
    urls = fetch_internal_links(base_url, max_links=20)
    urls_store[dom] = urls

    idx = faiss.IndexFlatL2(1536)
    chunks = []
    for url in urls:
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
    blob_idx = pickle.dumps(idx)
    blob_chunks = pickle.dumps(chunks)
    c.execute(
        "INSERT OR REPLACE INTO domains (domain, index_blob, chunks_blob) VALUES (?, ?, ?)",
        (dom, blob_idx, blob_chunks)
    )
    conn.commit()

    return {
        "chatbot_url": f"https://chatbot-frontend-zeta-tawny.vercel.app/{dom.replace('.', '-')}",
        "indexed": True,
        "fetched_urls": urls
    }

# --- Protected endpoint: ask ---
@app.post("/ask")
async def ask_bot(req: QueryRequest, user: str = Depends(get_current_user)):
    dom = normalize(req.domain)
    # Load from DB if missing in memory
    if dom not in index_store:
        row = c.execute("SELECT index_blob, chunks_blob FROM domains WHERE domain = ?", (dom,)).fetchone()
        if row:
            index_store[dom] = pickle.loads(row[0])
            chunks_store[dom] = pickle.loads(row[1])

    idx = index_store.get(dom)
    chunks = chunks_store.get(dom, [])
    if not idx or not chunks:
        return {"answer": "No content indexed yet for this domain. Please create a chatbot first."}

    user_emb = openai.embeddings.create(input=req.question, model="text-embedding-3-small").data[0].embedding
    D, I = idx.search(np.array([user_emb]).astype('float32'), k=3)
    selected = [chunks[i] for i in I[0] if i < len(chunks)]
    if not selected:
        return {"answer": "Sorry, I couldn't find relevant content to answer your question."}

    context = "\n---\n".join(selected)
    prompt = f"Answer the question based only on the context below.\n\nContext:\n{context}\n\nQuestion: {req.question}"
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Du er en hjelpsom assistent som svarer på samme språk som spørsmålet."},
            {"role": "user", "content": prompt}
        ]
    )
    answer_text = completion.choices[0].message.content.strip()

    # Log query
    c.execute(
        "INSERT INTO queries (domain, question, answer) VALUES (?, ?, ?)",
        (dom, req.question, answer_text)
    )
    conn.commit()

    return {"answer": answer_text}

# --- Client-facing proxy endpoints ---
@app.post("/client/create-chatbot")
async def client_create_chatbot(req: DomainRequest):
    return await create_chatbot(req)

@app.post("/client/ask")
async def client_ask(req: QueryRequest):
    return await ask_bot(req)

# --- Other endpoints omitted for brevity ---
