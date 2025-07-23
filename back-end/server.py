from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Optional
import lancedb
from sentence_transformers import SentenceTransformer
import openai
import numpy as np

# Connect to LanceDB Cloud
db = lancedb.connect(
    uri="db://researchgpt-17xoa0",
    api_key="",
    region="us-east-1"
)
table = db.open_table("reasearchgpt")

embed_model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
openai.api_key = ""

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def l2_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def rag_query(user_question, top_k=5):
    question_emb = embed_model.encode([user_question])[0].tolist()
    results = table.search(query=question_emb).distance_type("cosine").limit(top_k).to_list()
    context_lines = []
    for idx, r in enumerate(results, 1):
        sim_manual = cosine_similarity(question_emb, r["embeddings"])
        l2_manual = l2_distance(question_emb, r["embeddings"])
        cosine_lancedb = r["_distance"]
        context_lines.append(
            f"paper{idx}: title: {r['title']}\n"
            f"cosine_similarity (manual): {sim_manual:.6f}\n"
            f"l2_distance (manual): {l2_manual:.6f}\n"
            f"cosine_similarity (lancedb): {1 - cosine_lancedb:.6f}\n"
            f"diff (cosine): {abs(sim_manual - (1-cosine_lancedb)):.6e}\n"
        )
    context = "\n".join(context_lines)
    prompt = (
        f"You are a helpful AI research assistant. Here are the top {top_k} papers retrieved for the following question:\n"
        f"Question: {user_question}\n"
        f"{context}\n\n"
        f"For each paper, decide if you would recommend it (recommendation: true) or not (recommendation: false), based on its title and cosine distance. "
        f"ONLY include papers you recommend (recommendation: true) in your response.\n"
        f"For each recommended paper, use this format:\n"
        f"paperX: title: ..., cosine_similarity: ..., explanation: ...\n"
        f"Do not list or mention any papers you do not recommend."
    )

    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# ------------- FASTAPI SETUP -------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] for just local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RagRequest(BaseModel):
    user_question: str
    top_k: Optional[int] = 5

@app.post("/rag_cosine")
def rag_cosine_endpoint(req: RagRequest):
    output = rag_query(req.user_question, top_k=req.top_k)
    return {"output": output}