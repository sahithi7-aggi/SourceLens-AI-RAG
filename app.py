import streamlit as st

from src.rag_pipeline import create_rag_pipeline
from src.retriever import retrieve_documents, retrieve_per_source, build_context, source_label
from src.loader import normalize_urls, load_pdfs, load_docx, load_txt
from utils.helpers import MODE_CONFIG, hostname


st.set_page_config(
    page_title="SourceLens AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --sl-accent:#7c5cff; --sl-cyan:#22d3ee; --sl-muted:#94a3b8; --sl-border:rgba(255,255,255,.09); }
.block-container { max-width:1180px; padding-top:3.5rem !important; padding-bottom:3rem; }
[data-testid="stSidebar"] { border-right:1px solid var(--sl-border); }
.hero-title { font-size:3rem; font-weight:800; letter-spacing:-.055em; line-height:1.05; }
.hero-gradient { background:linear-gradient(90deg,#8b7cff,#35d5e8); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-subtitle { color:#9ca3af; font-size:1rem; margin-top:.55rem; line-height:1.5; }
.pill { display:inline-block; padding:.3rem .65rem; border-radius:999px; background:rgba(124,92,255,.11); color:#b9adff; border:1px solid rgba(124,92,255,.22); font-size:.78rem; margin-top:.7rem; }
.card { background:rgba(255,255,255,.03); border:1px solid var(--sl-border); border-radius:16px; padding:1rem 1.1rem; }
.answer-card { background:linear-gradient(135deg,rgba(124,92,255,.10),rgba(34,211,238,.045)); border:1px solid rgba(124,92,255,.20); border-radius:16px; padding:1.15rem 1.3rem; }
.source-chip { display:inline-block; border:1px solid rgba(255,255,255,.10); border-radius:999px; padding:.27rem .58rem; margin:.12rem .12rem .12rem 0; color:#cbd5e1; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "rag": None,
        "sources": [],
        "processed": False,
        "errors": [],
        "history": [],
        "last_results": [],
        "last_answer": "",
        "last_question": "",
        "last_mode": "Answer",
        "source_text": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


with st.sidebar:
    st.markdown("## 📚 Sources")

    source_type = st.radio(
        "Source type",
        ["🌐 Web URLs", "📄 Documents"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source_type == "🌐 Web URLs":
        st.caption("Up to 10 URLs")

        mode = st.selectbox("Research mode", list(MODE_CONFIG.keys()))
        config = MODE_CONFIG[mode]

        source_text = st.text_area(
            "URLs",
            value=st.session_state.source_text,
            placeholder="https://example.com\nhttps://example.org",
            height=145,
            label_visibility="collapsed",
        )
        st.session_state.source_text = source_text

        urls = normalize_urls(source_text.splitlines())
        st.caption(f"{len(urls)}/10 URLs")

        for url in urls:
            st.markdown(f'<span class="source-chip">🔗 {hostname(url)}</span>', unsafe_allow_html=True)

        if st.button("🚀 Process Sources", type="primary", use_container_width=True):
            if not urls:
                st.warning("Add at least one valid URL.")
            elif len(urls) > 10:
                st.warning("Maximum 10 URLs.")
            else:
                with st.spinner("Building your source index..."):
                    try:
                        rag = create_rag_pipeline(urls=urls)
                        st.session_state.rag = rag
                        st.session_state.sources = urls
                        st.session_state.errors = rag["errors"]
                        st.session_state.processed = True
                        st.session_state.history = []
                        st.session_state.last_results = []
                        st.session_state.last_answer = ""
                        st.session_state.last_question = ""
                        st.rerun()
                    except Exception as exc:
                        st.session_state.rag = None
                        st.session_state.processed = False
                        st.error(f"Could not process sources: {exc}")

    else:
        st.caption("Upload documents. Native PDF text is processed locally; scanned PDF pages can use local OCR.")

        file_types = ["pdf", "docx", "txt"]
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=file_types,
            accept_multiple_files=True,
            help="PDF, DOCX and TXT. Scanned PDFs use Tesseract OCR when installed.",
        )

        enable_ocr = st.checkbox("Enable OCR for scanned PDF pages", value=True)

        if uploaded_files:
            for f in uploaded_files:
                st.markdown(f'<span class="source-chip">📄 {f.name}</span>', unsafe_allow_html=True)

            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                with st.spinner("Extracting, embedding and indexing documents..."):
                    try:
                        pdfs = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]
                        docxs = [f for f in uploaded_files if f.name.lower().endswith(".docx")]
                        txts = [f for f in uploaded_files if f.name.lower().endswith(".txt")]

                        documents, errors, source_results = [], [], []

                        if pdfs:
                            d, e, s = load_pdfs(pdfs, enable_ocr=enable_ocr)
                            documents.extend(d); errors.extend(e); source_results.extend(s)
                        if docxs:
                            d, e, s = load_docx(docxs)
                            documents.extend(d); errors.extend(e); source_results.extend(s)
                        if txts:
                            d, e, s = load_txt(txts)
                            documents.extend(d); errors.extend(e); source_results.extend(s)

                        if not documents:
                            raise ValueError(
                                "No readable content was extracted. For scanned PDFs, install Tesseract OCR and add it to PATH."
                            )

                        rag = create_rag_pipeline(
                            documents=documents,
                            errors=errors,
                            source_results=source_results,
                        )

                        st.session_state.rag = rag
                        st.session_state.sources = [f.name for f in uploaded_files]
                        st.session_state.errors = errors
                        st.session_state.processed = True
                        st.session_state.history = []
                        st.session_state.last_results = []
                        st.session_state.last_answer = ""
                        st.session_state.last_question = ""
                        st.session_state.last_mode = "Answer"

                        st.success(
                            f"Indexed {len(documents)} document sections into {len(rag['chunks'])} chunks."
                        )
                        st.rerun()

                    except Exception as exc:
                        st.session_state.rag = None
                        st.session_state.processed = False
                        st.error(f"Could not process documents: {exc}")

    st.divider()
    st.markdown("### 🔎 Retrieval")
    top_k = st.slider(
        "Evidence chunks",
        min_value=2,
        max_value=8,
        value=5,
        help="Number of relevant chunks retrieved for each question.",
    )


st.markdown("""
<div class="hero">
    <div class="hero-title">🔎 <span class="hero-gradient">SourceLens AI</span></div>
    <div class="hero-subtitle">Ask across your sources. See the evidence behind every answer.</div>
    <div class="pill">RAG · Retrieval · Citations · Documents</div>
</div>
""", unsafe_allow_html=True)


if not st.session_state.processed:
    if source_type == "🌐 Web URLs":
        st.markdown(
            f'<div class="card"><strong>{mode}</strong> · '
            f'<span>{config["description"]}</span><br><br>'
            f'<span>Try: {config["examples"][0]}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="card"><strong>📄 Document Analysis</strong> · '
            '<span>Upload PDF, scanned PDF, DOCX or TXT files. '
            'Text extraction and embeddings are local; OCR can also run locally.</span></div>',
            unsafe_allow_html=True,
        )


if st.session_state.processed:
    rag = st.session_state.rag
    readable = len([r for r in rag.get("source_results", []) if r["status"] == "success"])
    failed = len(rag.get("source_results", [])) - readable

    c1, c2, c3 = st.columns(3)
    c1.metric("Sources", readable)
    c2.metric("Chunks", len(rag["chunks"]))
    c3.metric("Questions", len(st.session_state.history))

    if failed:
        with st.expander(f"⚠️ {failed} source(s) had issues"):
            for item in rag.get("source_results", []):
                if item["status"] == "failed":
                    st.write(f"**{item['url']}** — {item['reason']}")


st.markdown("## 💬 Ask")

if st.session_state.processed:
    research_mode = st.segmented_control(
        "Mode",
        ["Answer", "Compare Sources"],
        default="Answer",
        label_visibility="collapsed",
    )

    question_placeholder = (
        config["question"] if source_type == "🌐 Web URLs"
        else "Ask a question about your uploaded documents..."
    )

    question = st.text_area(
        "Question",
        placeholder=question_placeholder,
        height=90,
        label_visibility="collapsed",
    )

    ask = st.button("✨ Ask SourceLens", type="primary")

    if ask:
        if not question.strip():
            st.warning("Ask a question first.")
        else:
            rag = st.session_state.rag

            readable_sources = [
                item["url"]
                for item in rag.get("source_results", [])
                if item["status"] == "success"
            ]

            if research_mode == "Compare Sources" and len(readable_sources) < 2:
                st.warning("Compare needs at least 2 readable sources.")
            else:
                with st.spinner("Finding evidence..."):
                    try:
                        if research_mode == "Compare Sources":
                            results = retrieve_per_source(
                                rag["vectorstore"],
                                question,
                                readable_sources,
                                k_per_source=top_k,
                            )
                        else:
                            results = retrieve_documents(
                                rag["vectorstore"],
                                question,
                                top_k,
                            )

                        context, source_numbers = build_context(results)
                    except Exception as exc:
                        st.error(f"Retrieval failed: {exc}")
                        st.stop()

                if not results:
                    st.warning("No useful evidence was found.")
                else:
                    llm_question = question
                    if research_mode == "Compare Sources":
                        llm_question = (
                            f"Compare the supplied sources for this question:\n\n{question}\n\n"
                            "Separate similarities, differences, source-specific findings, and takeaway."
                        )

                    with st.spinner("Synthesizing..."):
                        try:
                            messages = rag["prompt"].invoke(
                                {"context": context, "question": llm_question}
                            )
                            response = rag["llm"].invoke(messages)
                        except Exception as exc:
                            st.error(f"Answer generation failed: {exc}")
                            st.stop()

                    st.session_state.last_results = results
                    st.session_state.last_answer = response.content
                    st.session_state.last_question = question
                    st.session_state.last_mode = research_mode
                    st.session_state.history.append({
                        "question": question,
                        "answer": response.content,
                        "mode": research_mode,
                    })

                    st.markdown("### 🤖 Answer")
                    st.markdown(
                        f'<div class="answer-card">{response.content}</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("### 📚 Sources")
                    cols = st.columns(min(3, max(1, len(source_numbers))))
                    for idx, (source, number) in enumerate(
                        sorted(source_numbers.items(), key=lambda x: x[1])
                    ):
                        with cols[idx % len(cols)]:
                            st.markdown(f"**[{number}] {source_label(source)}**")
                            st.caption(source)

                    st.markdown("### 🔎 Evidence")
                    st.caption("These are the retrieved chunks sent to the model.")

                    for item in results:
                        document = item["document"]
                        score = item["score"]
                        number = item["source_number"]
                        source = document.metadata.get("source", "Unknown source")
                        page = document.metadata.get("page")
                        method = document.metadata.get("extraction_method")

                        label = f"[{number}] {source_label(source)} · {score:.0%}"
                        if page:
                            label += f" · Page {page}"
                        if method == "ocr":
                            label += " · OCR"

                        with st.expander(label):
                            st.write(document.page_content)
                            if page:
                                st.caption(
                                    f"{source} · Page {page} · Extraction: {method}"
                                )
                            else:
                                st.caption(source)

else:
    st.markdown(
        '<div class="card"><strong>👈 Add your sources</strong><br>'
        '<span>Then ask a question.</span></div>',
        unsafe_allow_html=True,
    )


if st.session_state.history:
    with st.expander("🕘 History"):
        for item in reversed(st.session_state.history):
            st.markdown(f"**{item['mode']}** · {item['question']}")
            st.markdown(item["answer"])
            st.divider()

st.divider()
st.caption("SourceLens AI · grounded answers from the sources you provide.")
