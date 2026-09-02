from fastapi import FastAPI
from pydantic import BaseModel

from app.rag.chunker import read_document,split_into_sentences,create_chunks
from app.rag.embeddings import create_embeddings
from app.rag.vector_store import create_vector_store
from app.rag.retriever import retrieve

app = FastAPI()
class chatRequest(BaseModel):
    question : str

# prepare the knowledge base
text = read_document("documents/refund_policy.txt")
sentances = split_into_sentences(text)
chunks = create_chunks(sentances, chunk_size=3, overlap=1)
embeddings = create_embeddings(chunks)
index = create_vector_store(embeddings)


@app.post('/chat')
def chat(request:chatRequest):
    results = retrieve(request.question, index, chunks, k=2)
    return {
        "question": request.question,
        "retrieved_chunks" : results
    }