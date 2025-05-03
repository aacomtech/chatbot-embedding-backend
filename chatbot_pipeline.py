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

index = faiss.IndexFlatL2(1536)
stored_chunks = []

class DomainRequest(BaseModel):
    domain: str

class QueryRequest(BaseModel):
    domain: str
    question: str

def fetch_internal_links(base_url, max_links=10):
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
    base_url = req.domain if req.domain.startswith("http") else f"https://{req.domain}"
    urls = fetch_internal_links(base_url, max_links=10)

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
            index.add(np.array([emb]).astype('float32'))
            stored_chunks.append(chunk)

    return {"chatbot_url": f"https://yourchatbotsite.com/{req.domain.replace('.', '-')}"}

@app.post("/ask")
async def ask_bot(req: QueryRequest):
    if not stored_chunks or index.ntotal == 0:
        return {"answer": "No content indexed yet. Please create a chatbot first."}

    try:
        user_embedding = openai.embeddings.create(input=req.question, model=embedding_model).data[0].embedding
        D, I = index.search(np.array([user_embedding]).astype('float32'), k=3)
        selected_chunks = [stored_chunks[i] for i in I[0] if i < len(stored_chunks)]

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

    except Exception as e:
        return {"answer": f"Error during processing: {str(e)}"}
