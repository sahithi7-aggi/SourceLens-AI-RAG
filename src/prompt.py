from langchain_core.prompts import ChatPromptTemplate


def get_rag_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are SourceLens AI, a grounded document research assistant.
Answer using only the supplied context. If the context does not contain enough
information, say that clearly. Do not invent facts.

When useful, mention the source/page information included in the context.
""",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}",
        ),
    ])
