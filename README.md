# 🔎 SourceLens AI

A multi-source Retrieval-Augmented Generation research assistant built with Streamlit, LangChain, ChromaDB, local Hugging Face embeddings, and Groq.

## Features

- Add 1–10 URLs
- Multiple research modes: AI, Product, Research Papers, Documentation, Company Research
- Process sources once and ask multiple questions
- Grounded answers using only retrieved source context
- Source citations `[1]`, `[2]`, etc.
- **Answer mode**: global top-K retrieval
- **Compare Sources mode**: top-K retrieval independently from every readable source
- One citation number per unique source, even when multiple chunks are retrieved
- Retrieved evidence with relevance scores
- Per-URL failure handling with readable error messages
- Research history
- Modular RAG architecture

## Compare Sources retrieval

Normal questions use global similarity retrieval.

Comparison questions deliberately retrieve evidence independently from each readable URL:

```text
Source A → top K chunks
Source B → top K chunks
Source C → top K chunks
             ↓
        source-aware context
             ↓
          Groq LLM
```

This prevents one source from dominating the global top-K results and makes multi-source comparisons more reliable.

## Architecture

```text
URLs
 ↓
Per-URL reachability check
 ↓
Unstructured URL Loader
 ↓
Documents + source metadata
 ↓
Recursive Text Splitter
 ↓
Hugging Face Embeddings
 ↓
ChromaDB
 ↓
 ┌───────────────────────────────┐
 │ Answer: global top-K          │
 │ Compare: top-K per source     │
 └───────────────────────────────┘
 ↓
Source-numbered context
 ↓
Groq LLM
 ↓
Grounded Answer + Citations + Evidence
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Create `.streamlit/secrets.toml` using `.streamlit/secrets.toml.example` and add:

```toml
GROQ_API_KEY = "your-key-here"
```

## Project structure

```text
SourceLens_AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example
├── src/
│   ├── embeddings.py
│   ├── llm.py
│   ├── loader.py
│   ├── prompt.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── splitter.py
│   └── vectorstore.py
└── utils/
    └── helpers.py
```
