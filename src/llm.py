import streamlit as st
from langchain_groq import ChatGroq


def get_llm():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from .streamlit/secrets.toml")
    return ChatGroq(
        api_key=api_key,
        model="openai/gpt-oss-20b",
        temperature=0,
    )
