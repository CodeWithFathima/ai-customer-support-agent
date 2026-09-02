from app.rag.chunker import read_document,split_into_sentences,create_chunks
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

if __name__ == "__main__":
    text = read_document("documents/refund_policy.txt")
    sentences = split_into_sentences(text)
    chunks = create_chunks(sentences, chunk_size=3, overlap=1)
    embeddings = create_embeddings(chunks)
    print("Number of chunks:",len(chunks))
    print("Embedding Shape:",embeddings.shape)