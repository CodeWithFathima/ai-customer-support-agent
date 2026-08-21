from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

if __name__ == "__main__":
    chunks = [
       " Products can be returned within 30 days.",
        "Refunds are processed within 5 business days ."
    ]
    embeddings = create_embeddings(chunks)
    print(type(embeddings))
    print(embeddings.shape)