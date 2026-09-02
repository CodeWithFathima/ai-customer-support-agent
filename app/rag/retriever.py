import numpy as np
from app.rag.chunker import read_document,split_into_sentences,create_chunks
from app.rag.embeddings import create_embeddings
from app.rag.vector_store import create_vector_store

def retrieve(query, index, chunks, k=2):
    query_embedding = create_embeddings([query])  #convert the question into 384 - dimensional vectors
    query_embedding = np.array(query_embedding,dtype='float32') 

    distances, indices = index.search(query_embedding,k)  #asks FAISS : which stored document vectors are closest to this qn?

    results = []
    for i in indices[0]:       #convert those numbers back into actual text
        results.append(chunks[i])

    return results

if __name__ == "__main__":
    text = read_document("documents/refund_policy.txt")
    sentences = split_into_sentences(text)
    chunks = create_chunks(sentences,chunk_size=3,overlap=1)
    embeddings = create_embeddings(chunks)
    index = create_vector_store(embeddings)
    query = "How many days do i have to return a product?"
    results = retrieve(query, index, chunks, k=2)
    print("Query:", query)
    print("Retrieved Chunks: ")
    for result in results:
        print(result)

