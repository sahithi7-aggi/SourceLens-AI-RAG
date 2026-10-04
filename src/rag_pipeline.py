from src.loader import load_urls
from src.splitter import split_documents
from src.embeddings import get_embedding_model
from src.vectorstore import create_vectorstore
from src.llm import get_llm
from src.prompt import get_rag_prompt


def create_rag_pipeline(urls):
    """Build a RAG index while preserving per-source load status."""
    documents, errors, source_results = load_urls(urls)

    if not documents:
        raise ValueError(
            "None of the provided URLs could be loaded. "
            "Check the source errors and verify the URLs."
        )

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("The loaded sources did not contain usable text.")

    embedding_model = get_embedding_model()
    vectorstore = create_vectorstore(chunks, embedding_model)

    return {
        "vectorstore": vectorstore,
        "retriever": vectorstore.as_retriever(search_kwargs={"k": 5}),
        "embedding_model": embedding_model,
        "llm": get_llm(),
        "prompt": get_rag_prompt(),
        "documents": documents,
        "chunks": chunks,
        "errors": errors,
        "source_results": source_results,
    }
