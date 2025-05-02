# chatbot_pipeline.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import trafilatura
import openai
import faiss
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = "YOUR_OPENAI_API_KEY"
embedding_model = "text-embedding-3-small"

index = faiss.IndexFlatL2(1536)  # For text-embedding-3-small
stored_chunks = []  # Matches vector order

class DomainRequest(BaseModel):
    domain: str

class QueryRequest(BaseModel):
    domain: str
    question: str

@app.post("/create-chatbot")
async def create_chatbot(req: DomainRequest):
    url = req.domain if req.domain.startswith("http") else f"https://{req.domain}"
    raw = trafilatura.fetch_url(url)
    if not raw:
        return {"message": "Could not fetch content."}, 400

    text = trafilatura.extract(raw)
    if not text:
        return {"message": "Could not extract content."}, 400

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    for chunk in chunks:
        emb = openai.embeddings.create(input=chunk, model=embedding_model).data[0].embedding
        index.add(np.array([emb]).astype('float32'))
        stored_chunks.append(chunk)

    return {"chatbot_url": f"https://yourchatbotsite.com/{req.domain.replace('.', '-')}"}

@app.post("/ask")
async def ask_bot(req: QueryRequest):
    if not stored_chunks or index.ntotal == 0:
        return {"message": "No content indexed yet. Please create a chatbot first."}, 400

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
        return {"message": f"Error during processing: {str(e)}"}, 500
