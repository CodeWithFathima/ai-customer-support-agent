import faiss
import numpy as np

from app.rag.chunker import read_document, create_chunks, split_into_sentences
from app.rag.embeddings import create_embeddings

def create_vector_store(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index

def search_vectors(index,query_embedding,chunks,k=2):
    distances, indices = index.search(query_embedding,k)
    results =[]
    for index in indices[0]:
        results.append(chunks[index])
    return results

if __name__ == "__main__":
    text = read_document("documents/refund_policy.txt")
    sentences = split_into_sentences(text)
    chunks = create_chunks(sentences, chunk_size=3, overlap=1)
    embeddings = create_embeddings(chunks)
    index = create_vector_store(embeddings)
    # query = np.array([
    #     [0.12,0.21,0.31]
    # ], dtype='float32')

    # results = search_vectors(index,query,chunks,k=2)    
    # print('retrieved chunks:',results)
    print("Number of chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    print("Number of vectors:", index.ntotal)

