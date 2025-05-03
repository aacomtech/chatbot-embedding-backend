# chatbot_pipeline.py

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import trafilatura
import openai
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("Missing OpenAI API key. Make sure OPENAI_API_KEY is set.")

embedding_model = "text-embedding-3-small"

# In-memory storage for multiple domains
index_store = {}
chunks_store = {}

class DomainRequest(BaseModel):
    domain: str

class QueryRequest(BaseModel):
    domain: str
    question: str

def fetch_internal_links(base_url, max_links=20):
    try:
        response = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        links = set()
        base_netloc = urlparse(base_url).netloc

        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            joined = urljoin(base_url, href)
            parsed = urlparse(joined)
            if parsed.netloc == base_netloc and parsed.scheme.startswith("http"):
                links.add(joined)
            if len(links) >= max_links:
                break

        return list(links)
    except Exception as e:
        print("Error fetching links:", e)
        return [base_url]

@app.post("/create-chatbot")
async def create_chatbot(req: DomainRequest):
    domain = req.domain.replace("https://", "").replace("http://", "").strip("/")
    base_url = f"https://{domain}"
    urls = fetch_internal_links(base_url, max_links=20)

    domain_index = faiss.IndexFlatL2(1536)
    domain_chunks = []

    for url in urls:
        raw = trafilatura.fetch_url(url)
        if not raw:
            continue
        text = trafilatura.extract(raw)
        if not text:
            continue

        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        for chunk in chunks:
            emb = openai.embeddings.create(input=chunk, model=embedding_model).data[0].embedding
            domain_index.add(np.array([emb]).astype('float32'))
            domain_chunks.append(chunk)

    index_store[domain] = domain_index
    chunks_store[domain] = domain_chunks

    return {"chatbot_url": f"https://yourchatbotsite.com/{domain.replace('.', '-')}"}

@app.post("/ask")
async def ask_bot(req: QueryRequest):
    domain = req.domain.replace("https://", "").replace("http://", "").strip("/")
    domain_index = index_store.get(domain)
    domain_chunks = chunks_store.get(domain)

    if not domain_index or not domain_chunks or domain_index.ntotal == 0:
        return {"answer": "No content indexed yet for this domain. Please create a chatbot first."}

    try:
        user_embedding = openai.embeddings.create(input=req.question, model=embedding_model).data[0].embedding
        D, I = domain_index.search(np.array([user_embedding]).astype('float32'), k=3)
        selected_chunks = [domain_chunks[i] for i in I[0] if i < len(domain_chunks)]

        if not selected_chunks:
            return {"answer": "Sorry, I couldn't find relevant content to answer your question."}

        context = "\n---\n".join(selected_chunks)
        prompt = f"Answer the question based only on the context below.\n\nContext:\n{context}\n\nQuestion: {req.question}"
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return {"answer": completion.choices[0].message.content.strip()}

    except Exception as e:
        return {"answer": f"Error during processing: {str(e)}"}
