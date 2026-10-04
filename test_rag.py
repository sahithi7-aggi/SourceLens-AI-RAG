from src.rag_pipeline import create_rag_pipeline


url = "https://docs.python.org/3/tutorial/index.html"

question = "What is Python used for?"

print("Creating RAG pipeline...")

retriever, llm, prompt = create_rag_pipeline(url)

print("RAG pipeline ready!")

# Retrieve relevant chunks
documents = retriever.invoke(question)

print(f"\nRetrieved {len(documents)} documents.")

# Combine retrieved content
context = "\n\n".join(
    document.page_content
    for document in documents
)

# Create the prompt
messages = prompt.invoke({
    "context": context,
    "question": question
})

# Ask Groq
response = llm.invoke(messages)

print("\n========== ANSWER ==========\n")
print(response.content)

print("\n========== SOURCES ==========\n")

for document in documents:
    print(document.metadata.get("source"))