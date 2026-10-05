import platform
import tempfile

from chromadb.config import Settings
from langchain_chroma import Chroma


def create_vectorstore(chunks, embedding_model):
    if not chunks:
        raise ValueError("No chunks were provided to create the vector store.")

    # Give each indexing run its own persistent directory.
    # This avoids Chroma's in-memory SQLite state being lost
    # between Streamlit reruns.
    persist_directory = tempfile.mkdtemp(
        prefix="sourcelens_chroma_"
    )

    client_settings = None

    # Windows workaround for Chroma's Rust bindings.
    if platform.system() == "Windows":
        client_settings = Settings(
            anonymized_telemetry=False,
            chroma_api_impl="chromadb.api.segment.SegmentAPI",
        )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="source_lens",
        persist_directory=persist_directory,
        client_settings=client_settings,
    )

    return vectorstore