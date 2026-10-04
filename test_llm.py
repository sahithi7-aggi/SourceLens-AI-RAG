from src.llm import get_llm


llm = get_llm()

response = llm.invoke(
    "Explain what a vector database is in two sentences."
)

print(response.content)