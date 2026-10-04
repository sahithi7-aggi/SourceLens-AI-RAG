from src.loader import load_url

url = "https://example.com"

documents = load_url(url)

print("Number of documents:", len(documents))

for document in documents:
    print("\n--- CONTENT ---")
    print(document.page_content[:1000])

    print("\n--- METADATA ---")
    print(document.metadata)