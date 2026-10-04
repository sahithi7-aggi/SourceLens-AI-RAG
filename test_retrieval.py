from src.loader import load_url
from src.splitter import split_documents
from src.embeddings import get_embedding_model
from src.vectorstore import create_vectorstore


# ---------------------------------
# 1. Load webpage
# ---------------------------------

url = "https://docs.python.org/3/tutorial/index.html"

print("Loading webpage...")

documents = load_url(url)

print(f"Loaded {len(documents)} document(s)")


# ---------------------------------
# 2. Split into chunks
# ---------------------------------

print("\nSplitting documents...")

chunks = split_documents(documents)

print(f"Created {len(chunks)} chunks")


# ---------------------------------
# 3. Create embedding model
# ---------------------------------

print("\nLoading embedding model...")

embedding_model = get_embedding_model()

print("Embedding model loaded")


# ---------------------------------
# 4. Create Chroma vector store
# ---------------------------------

print("\nCreating ChromaDB...")

vectorstore = create_vectorstore(
    chunks,
    embedding_model
)

print("ChromaDB created")


# ---------------------------------
# 5. Create retriever
# ---------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# ---------------------------------
# 6. Ask a question
# ---------------------------------

question = "What is Python used for?"

print(f"\nQuestion: {question}")

results = retriever.invoke(question)


# ---------------------------------
# 7. Display results
# ---------------------------------

print("\n========== RETRIEVED DOCUMENTS ==========")

for i, document in enumerate(results):

    print(f"\n----- Result {i + 1} -----")

    print("\nContent:")
    print(document.page_content)

    print("\nSource:")
    print(document.metadata.get("source"))