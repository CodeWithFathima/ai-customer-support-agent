from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
sentence = 'How can i reset my password'
embedding = model.encode(sentence)
# print(embedding)
print(type(embedding))
print(len(embedding))
print(embedding[:10])   # Show only the first 10 numbers