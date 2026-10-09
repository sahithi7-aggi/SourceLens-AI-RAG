import streamlit as st

st.set_page_config(page_title="SourceLens Architecture", page_icon="🧭", layout="wide")

st.title("🧭 SourceLens Architecture")

st.markdown("""
### Current pipeline

```text
Web URL / PDF / Scanned PDF / DOCX / TXT
                ↓
          Local extraction
                ↓
        OCR when required
                ↓
       LangChain Documents
                ↓
        Recursive chunking
                ↓
 Local Hugging Face embeddings
                ↓
          Local ChromaDB
                ↓
        Similarity retrieval
                ↓
     Relevant chunks only
                ↓
             Groq
                ↓
       Grounded answer
```

### Privacy boundary

- Document parsing happens locally in this application.
- OCR is local when Tesseract is installed.
- Embeddings are generated locally with `all-MiniLM-L6-v2`.
- ChromaDB is stored locally in `chroma_db/`.
- In the current cloud-LLM mode, only retrieved context is sent to Groq.
- For highly sensitive documents, do not use the public Streamlit deployment; use the local app or later add a fully local LLM mode.
""")
