import streamlit as st
from langchain_groq import ChatGroq


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"],
    )
