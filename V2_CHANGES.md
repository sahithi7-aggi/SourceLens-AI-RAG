# What changed in V2

1. `src/loader.py`
   - Keeps existing URL ingestion.
   - Adds PDF page-by-page extraction.
   - Adds local OCR fallback for scanned PDF pages.
   - Adds DOCX and TXT loaders.
   - Preserves source/page/extraction metadata.

2. `src/rag_pipeline.py`
   - Accepts either URLs or pre-loaded documents.
   - Keeps the same splitter → embeddings → Chroma → LLM architecture.

3. `src/retriever.py`
   - Defines retrieval settings in one place.
   - Handles Answer and Compare Sources modes.
   - Preserves page metadata in evidence.

4. `app.py`
   - Adds Web URLs / Documents switch.
   - Adds PDF/DOCX/TXT upload.
   - Adds OCR toggle.
   - Fixes `top_k` and `readable_sources` scope issues.
   - Shows page and OCR information in evidence.

5. `src/vectorstore.py`
   - Rebuilds a fresh local Chroma collection for each processing run so old sessions do not contaminate a new upload.

6. Privacy
   - Local extraction.
   - Local OCR.
   - Local Hugging Face embeddings.
   - Local ChromaDB.
   - Only retrieved context is sent to Groq in the default mode.
