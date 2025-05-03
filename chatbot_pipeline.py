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
PASS = os.getenv("API_PASS", "secret")

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
c.execute(
    '''CREATE TABLE IF NOT EXISTS domains (
         domain TEXT PRIMARY KEY,
         index_blob BLOB,
         chunks_blob BLOB
    )'''
)
conn.commit()

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

# --- OpenAPI Security Definitions ---
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Chatbot Embedding API",
        version="1.0.0",
        description="API for managing and querying website chatbots",
        routes=app.routes,
    )
    # Define HTTP Basic security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "basicAuth": {"type": "http", "scheme": "basic"}
    }
    # Apply basicAuth to protected endpoints
    protected_paths = ["/create-chatbot", "/ask", "/domains", "/domains/{domain}/info"]
    for path in openapi_schema.get("paths", {}):
        if path in protected_paths:
            for method in openapi_schema["paths"][path]:
                openapi_schema["paths"][path][method]["security"] = [{"basicAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Override default openapi
app.openapi = custom_openapi

# --- Models ---
class DomainRequest(BaseModel):
    domain: str

class QueryRequest(BaseModel):
    domain: str
    question: str

# --- In-memory stores ---
index_store = {}
chunks_store = {}
urls_store = {}

# --- Utility to normalize domain and load index ---
def normalize(domain: str) -> str:
    return domain.replace("https://", "").replace("http://", "").strip("/")

# --- Utility to fetch internal links ---
def fetch_internal_links(base_url: str, max_links: int = 20) -> list[str]:
    try:
        resp = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Always include the main page itself
        links = {base_url}
        base_netloc = urlparse(base_url).netloc
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

# --- Protected endpoint: create-chatbot ---
@app.post("/create-chatbot")
async def create_chatbot(req: DomainRequest, user: str = Depends(get_current_user)):
    dom = normalize(req.domain)
    base_url = f"https://{dom}"
    urls = fetch_internal_links(base_url, max_links=20)
    urls_store[dom] = urls
    idx = faiss.IndexFlatL2(1536)

    # Prepare per-URL and flat chunk storage
    url_chunks_map = {}
    flat_chunks = []
    for url in urls:
        raw = trafilatura.fetch_url(url)
        text = trafilatura.extract(raw) if raw else None
        if not text:
            continue
        # Split into chunks for this URL
        this_chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        # Add embeddings
        for chunk in this_chunks:
            emb = openai.embeddings.create(input=chunk, model="text-embedding-3-small").data[0].embedding
            idx.add(np.array([emb]).astype('float32'))
        # Store
        url_chunks_map[url] = this_chunks
        flat_chunks.extend(this_chunks)

    # Store in-memory
    index_store[dom] = idx
    chunks_store[dom] = { 'flat': flat_chunks, 'by_url': url_chunks_map }

    # Persist to SQLite (flat only)
    blob_idx = pickle.dumps(idx)
    blob_chunks = pickle.dumps(flat_chunks)
    c.execute(
        "INSERT OR REPLACE INTO domains (domain, index_blob, chunks_blob) VALUES (?, ?, ?)",
        (dom, blob_idx, blob_chunks)
    )
    conn.commit()

    return {"chatbot_url": f"https://yourchatbotsite.com/{dom.replace('.', '-')}", "indexed": True, "fetched_urls": urls}

# --- Protected endpoint: ask ---
@app.post("/ask")
async def ask_bot(req: QueryRequest, user: str = Depends(get_current_user)):
    dom = normalize(req.domain)
    if dom not in index_store:
        c.execute("SELECT index_blob, chunks_blob FROM domains WHERE domain = ?", (dom,))
        row = c.fetchone()
        if row:
            index_store[dom] = pickle.loads(row[0])
            # Load flat chunks
            chunks_store[dom] = {
                'flat': pickle.loads(row[1]),
                'by_url': chunks_store.get(dom, {})  # preserve by_url if exists
            }
    idx = index_store.get(dom)
    # Use flat chunk list for retrieval
    chunks = chunks_store.get(dom, {}).get('flat', [])
    if not idx or not chunks or idx.ntotal == 0:
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
    return {"answer": completion.choices[0].message.content.strip()}

# --- Client-facing proxy endpoints (no auth) ---
@app.post("/client/create-chatbot")
async def client_create_chatbot(req: DomainRequest):
    return await create_chatbot(req)

@app.post("/client/ask")
async def client_ask(req: QueryRequest):
    return await ask_bot(req)

# --- List indexed domains ---
@app.get("/domains")
async def list_indexed_domains(user: str = Depends(get_current_user)):
    c.execute("SELECT domain FROM domains")
    rows = [r[0] for r in c.fetchall()]
    return {"indexed_domains": rows}

# --- Domain info ---
@app.get("/domains/{domain}/info")
async def domain_info(domain: str, user: str = Depends(get_current_user)):
    dom = normalize(domain)
    urls = urls_store.get(dom, [])
    # Ensure chunks loaded
    store = chunks_store.get(dom, {'flat': [], 'by_url': {}})
    url_chunks = store.get('by_url', {})
    # Sample: first chunk of each URL
    sample_chunks = {url: chunks[0] for url, chunks in url_chunks.items() if chunks}
    return {
        "domain": dom,
        "fetched_urls": urls,
        "chunk_count": len(store.get('flat', [])),
        "sample_chunks": sample_chunks
    }
