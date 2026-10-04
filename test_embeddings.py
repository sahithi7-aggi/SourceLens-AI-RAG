from src.embeddings import get_embedding_model


embeddings = get_embedding_model()

text = "Python is a programming language."

vector = embeddings.embed_query(text)

print("Vector type:", type(vector))
print("Vector dimensions:", len(vector))

print("\nFirst 10 values:")
print(vector[:10])