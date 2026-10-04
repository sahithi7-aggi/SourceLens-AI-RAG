from src.loader import load_url
from src.splitter import split_documents


url = "https://physarum.ai/"

documents = load_url(url)

chunks = split_documents(documents)

print("Number of documents:", len(documents))
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n========== CHUNK {i + 1} ==========")

    print("Characters:", len(chunk.page_content))

    print("\nContent:")
    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)