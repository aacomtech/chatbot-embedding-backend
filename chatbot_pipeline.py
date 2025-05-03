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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Environment & API key ---
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OpenAI API key. Make sure OPENAI_API_KEY is set.")

# --- SQLite setup ---
DB_PATH = "chatbot_data.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS domains (
    domain      TEXT PRIMARY KEY,
    index_blob  BLOB,
    chunks_blob BLOB
)
''')
conn.commit()

# --- FastAPI setup ---
app = FastAPI()
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

# --- In-memory stores ---
index_store = {}
chunks_store = {}

# --- Utilities ---
def fetch_internal_links(base_url, max_links=20):
    try:
        resp = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        links, base_netloc = set(), urlparse(base_url).netloc
        for tag in soup.find_all("a", href=True):
            href = urljoin(base_url, tag["href"])
            p = urlparse(href)
            if p.netloc == base_netloc and p.scheme.startswith("http"):
                links.add(href)
            if len(links) >= max_links:
                break
        return list(links)
    except Exception:
        return [base_url]

# --- API Endpoints ---
@app.post("/create-chatbot")
async def create_chatbot(req: DomainRequest):
    # Normalize domain
    dom = req.domain.replace("https://", "").replace("http://", "").strip("/")
    base_url = f"https://{dom}"

    # Crawl up to 20 internal links
    urls = fetch_internal_links(base_url, max_links=20)

    # Build FAISS index and chunks
    domain_index = faiss.IndexFlatL2(1536)
    domain_chunks = []
    for url in urls:
        raw = trafilatura.fetch_url(url)
        text = trafilatura.extract(raw) if raw else None
        if not text:
            continue
        for i in range(0, len(text), 1000):
            chunk = text[i:i+1000]
            emb = openai.embeddings.create(input=chunk, model="text-embedding-3-small").data[0].embedding
            domain_index.add(np.array([emb]).astype('float32'))
            domain_chunks.append(chunk)

    # Store in-memory
    index_store[dom] = domain_index
    chunks_store[dom] = domain_chunks

    # Persist to SQLite
    blob_index = pickle.dumps(domain_index)
    blob_chunks = pickle.dumps(domain_chunks)
    c.execute(
        "INSERT OR REPLACE INTO domains (domain, index_blob, chunks_blob) VALUES (?, ?, ?)",
        (dom, blob_index, blob_chunks)
    )
    conn.commit()

    return {"chatbot_url": f"https://yourchatbotsite.com/{dom.replace('.', '-')}"}

@app.post("/ask")
async def ask_bot(req: QueryRequest):
    # Normalize domain
    dom = req.domain.replace("https://", "").replace("http://", "").strip("/")

    # Load from memory or SQLite
    if dom not in index_store:
        c.execute("SELECT index_blob, chunks_blob FROM domains WHERE domain = ?", (dom,))
        row = c.fetchone()
        if row:
            index_store[dom] = pickle.loads(row[0])
            chunks_store[dom] = pickle.loads(row[1])

    domain_index = index_store.get(dom)
    domain_chunks = chunks_store.get(dom)
    if not domain_index or not domain_chunks or domain_index.ntotal == 0:
        return {"answer": "No content indexed yet for this domain. Please create a chatbot first."}

    # Create embedding for question
    user_emb = openai.embeddings.create(input=req.question, model="text-embedding-3-small").data[0].embedding
    D, I = domain_index.search(np.array([user_emb]).astype('float32'), k=3)
    selected = [domain_chunks[i] for i in I[0] if i < len(domain_chunks)]
    if not selected:
        return {"answer": "Sorry, I couldn't find relevant content to answer your question."}

    # Construct prompt & get answer
    context = "\n---\n".join(selected)
    prompt = f"Answer the question based only on the context below.\n\nContext:\n{context}\n\nQuestion: {req.question}"
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"answer": completion.choices[0].message.content.strip()}
