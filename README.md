# SourceLens AI — Document & Web RAG V2

This version extends SourceLens from URL-only RAG to document RAG.

## Supported inputs

- Web URLs
- Text PDFs
- Scanned/image-only PDFs with local Tesseract OCR
- DOCX
- TXT

## Pipeline

Web/PDF/DOCX/TXT
→ local extraction
→ local OCR when needed
→ LangChain Documents
→ recursive chunking
→ local Hugging Face embeddings
→ local ChromaDB
→ similarity retrieval
→ relevant context
→ Groq answer generation

## Privacy

Document parsing, OCR, embeddings and ChromaDB are local when running the app locally.

In the default Groq mode, the retrieved context is sent to Groq for answer generation. For highly confidential material, use the app locally and do not deploy it publicly. A fully local LLM mode can be added later.

Never commit `.streamlit/secrets.toml` or private documents to GitHub.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create:

`.streamlit/secrets.toml`

with:

```toml
GROQ_API_KEY = "your-key"
```

Run:

```powershell
streamlit run app.py
```

## OCR setup on Windows

Scanned PDFs require Tesseract OCR.

1. Install Tesseract OCR.
2. Ensure `tesseract.exe` is on your PATH.
3. Restart PowerShell/VS Code.
4. Verify:

```powershell
tesseract --version
```

Then enable **OCR for scanned PDF pages** in SourceLens.

## Important

The `chroma_db/` folder is generated locally and is ignored by Git.
Private uploaded documents are also ignored by Git via `.gitignore`.
