from pathlib import Path
import pickle

import numpy as np


VECTORSTORE_DIR = Path("vectorstore_data")


class LocalRetriever:
    """Small LangChain-compatible retriever wrapper."""

    def __init__(self, vectorstore, k=4):
        self.vectorstore = vectorstore
        self.k = k

    def invoke(self, query):
        return self.vectorstore.similarity_search(
            query,
            k=self.k,
        )

    def get_relevant_documents(self, query):
        return self.invoke(query)


class LocalVectorStore:
    """
    Lightweight local vector store using NumPy cosine similarity.

    This avoids the Windows Chroma/HNSW native crash.
    """

    def __init__(self, documents, embeddings, embedding_model):
        self.documents = documents
        self.embedding_model = embedding_model

        self.embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        # Normalize vectors for cosine similarity.
        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True,
        )

        norms[norms == 0] = 1.0

        self.embeddings = self.embeddings / norms

    def similarity_search_with_score(
        self,
        query,
        k=4,
        filter=None,
    ):
        query_embedding = np.asarray(
            self.embedding_model.embed_query(query),
            dtype=np.float32,
        )

        norm = np.linalg.norm(query_embedding)

        if norm != 0:
            query_embedding = query_embedding / norm

        scores = self.embeddings @ query_embedding

        candidate_indices = np.arange(
            len(self.documents)
        )

        # Metadata filtering.
        if filter:
            filtered_indices = []

            for idx in candidate_indices:
                metadata = self.documents[idx].metadata

                matches = all(
                    metadata.get(key) == value
                    for key, value in filter.items()
                )

                if matches:
                    filtered_indices.append(idx)

            candidate_indices = np.array(
                filtered_indices,
                dtype=int,
            )

            if len(candidate_indices) == 0:
                return []

        candidate_scores = scores[candidate_indices]

        k = min(
            k,
            len(candidate_indices),
        )

        if k <= 0:
            return []

        top_positions = np.argsort(
            candidate_scores
        )[-k:][::-1]

        top_indices = candidate_indices[
            top_positions
        ]

        return [
            (
                self.documents[idx],
                float(scores[idx]),
            )
            for idx in top_indices
        ]

    def similarity_search(
        self,
        query,
        k=4,
        filter=None,
    ):
        results = self.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
        )

        return [
            document
            for document, _ in results
        ]

    def as_retriever(
        self,
        search_kwargs=None,
        **kwargs,
    ):
        """
        Provide the same interface expected by the
        existing RAG pipeline.
        """

        search_kwargs = search_kwargs or {}

        k = search_kwargs.get(
            "k",
            4,
        )

        return LocalRetriever(
            self,
            k=k,
        )


def create_vectorstore(
    documents,
    embedding_model,
):
    """
    Create the local NumPy vector index.
    """

    if not documents:
        raise ValueError(
            "No documents were provided."
        )

    print(
        f"Creating embeddings for "
        f"{len(documents)} chunks..."
    )

    texts = [
        document.page_content
        for document in documents
    ]

    embeddings = (
        embedding_model.embed_documents(
            texts
        )
    )

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    vectorstore = LocalVectorStore(
        documents=documents,
        embeddings=embeddings,
        embedding_model=embedding_model,
    )

    # Save local vector data.
    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        VECTORSTORE_DIR / "documents.pkl",
        "wb",
    ) as f:
        pickle.dump(
            documents,
            f,
        )

    np.save(
        VECTORSTORE_DIR / "embeddings.npy",
        vectorstore.embeddings,
    )

    print(
        "Local vector store created successfully."
    )

    return vectorstore