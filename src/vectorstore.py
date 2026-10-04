from langchain_chroma import Chroma


def create_vectorstore(chunks, embedding_model):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )
