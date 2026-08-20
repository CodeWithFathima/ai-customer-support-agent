from pathlib import Path
def read_document(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        return file.read()
def split_into_sentences(text):
    sentences = text.split(".")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

def create_chunks(sentences,chunk_size=3,overlap=1):
    chunks = []
    step = chunk_size - overlap
    for start in range(0,len(sentences),step):
        chunk = sentences[start:start + chunk_size]
        if chunk:
            chunks.append(f"\n".join(chunk))
    return chunks
if __name__ == "__main__":
    text = read_document('documents/refund_policy.txt')
    sentences = split_into_sentences(text)
    chunks = create_chunks(sentences,chunk_size=3,overlap=1)
    for i,chunk in enumerate(chunks):
        print(f"\nChunk{i+1}:")
        print(chunk)