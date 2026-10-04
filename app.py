import streamlit as st

from src.rag_pipeline import create_rag_pipeline
from src.retriever import retrieve_documents, retrieve_per_source, build_context, source_label
from src.loader import normalize_urls
from utils.helpers import MODE_CONFIG, hostname


st.set_page_config(
    page_title="SourceLens AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
:root { --sl-accent:#7c5cff; --sl-cyan:#22d3ee; --sl-green:#34d399; --sl-red:#fb7185; --sl-muted:#94a3b8; --sl-border:rgba(255,255,255,.09); }
.block-container { max-width:1180px; padding-top:3.5rem !important; padding-bottom:3rem; }
[data-testid="stSidebar"] { border-right:1px solid var(--sl-border); }
.hero { padding:.4rem 0 1.1rem; }
.hero-title { font-size:3rem; font-weight:800; letter-spacing:-.055em; line-height:1.05; }
.hero-gradient { background:linear-gradient(90deg,#8b7cff,#35d5e8); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-subtitle { color:#9ca3af; font-size:1rem; margin-top:.55rem; line-height:1.5; }
.pill { display:inline-block; padding:.3rem .65rem; border-radius:999px; background:rgba(124,92,255,.11); color:#b9adff; border:1px solid rgba(124,92,255,.22); font-size:.78rem; margin-top:.7rem; }
.card { background:rgba(255,255,255,.03); border:1px solid var(--sl-border); border-radius:16px; padding:1rem 1.1rem; }
.answer-card { background:linear-gradient(135deg,rgba(124,92,255,.10),rgba(34,211,238,.045)); border:1px solid rgba(124,92,255,.20); border-radius:16px; padding:1.15rem 1.3rem; }
.source-chip { display:inline-block; border:1px solid rgba(255,255,255,.10); border-radius:999px; padding:.27rem .58rem; margin:.12rem .12rem .12rem 0; color:#cbd5e1; font-size:.8rem; }
.small-muted { color:var(--sl-muted); font-size:.84rem; }
[data-testid="stMetric"] { background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.07); padding:.65rem .8rem; border-radius:13px; }
div.stButton > button { border-radius:10px; font-weight:650; }
.step-chip { border:1px solid var(--sl-border); border-radius:12px; padding:.55rem .7rem; text-align:center; background:rgba(255,255,255,.02); }
</style>
""",
    unsafe_allow_html=True,
)


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


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("## 📚 Sources")
    st.caption("Up to 10 URLs")

    mode = st.selectbox("Research mode", list(MODE_CONFIG.keys()))
    config = MODE_CONFIG[mode]

    source_text = st.text_area(
        "URLs",
        value=st.session_state.source_text,
        placeholder=(
            "https://platform.openai.com/docs/guides/function-calling\n"
            "https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use"
        ),
        height=145,
        label_visibility="collapsed",
    )
    st.session_state.source_text = source_text

    urls = normalize_urls(source_text.splitlines())
    st.caption(f"{len(urls)}/10 URLs")

    if urls:
        for url in urls:
            st.markdown(
                f'<span class="source-chip">🔗 {hostname(url)}</span>',
                unsafe_allow_html=True,
            )

    if st.button("🚀 Process Sources", type="primary", use_container_width=True):
        if not urls:
            st.warning("Add at least one valid URL.")
        elif len(urls) > 10:
            st.warning("Maximum 10 URLs.")
        else:
            with st.spinner("Building your source index..."):
                try:
                    rag = create_rag_pipeline(urls)
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

    if st.session_state.processed:
        readable = len([r for r in st.session_state.rag.get("source_results", []) if r["status"] == "success"])
        failed = len(st.session_state.rag.get("source_results", [])) - readable
        st.success(f"{readable} ready" + (f" · {failed} failed" if failed else ""))

        if st.button("↻ Replace Sources", use_container_width=True):
            for key, value in {
                "rag": None,
                "processed": False,
                "sources": [],
                "errors": [],
                "last_results": [],
                "last_answer": "",
                "last_question": "",
            }.items():
                st.session_state[key] = value
            st.rerun()

    st.divider()
    st.markdown("### Retrieval")
    top_k = st.slider("Chunks", 2, 8, 5, help="Number of evidence chunks retrieved.")

    if st.session_state.processed:
        st.divider()
        if st.button("🧭 Visualize RAG", use_container_width=True):
            st.switch_page("pages/2_Architecture.py")


# ---------------------------
# Hero
# ---------------------------
st.markdown(
    """
<div class="hero">
    <div class="hero-title">🔎 <span class="hero-gradient">SourceLens AI</span></div>
    <div class="hero-subtitle">Ask across your sources. See the evidence behind every answer.</div>
    <div class="pill">RAG · Retrieval · Citations</div>
</div>
""",
    unsafe_allow_html=True,
)


if not st.session_state.processed:
    st.markdown(
        f'<div class="card"><strong>{mode}</strong> · <span class="small-muted">{config["description"]}</span><br><br>'
        f'<span class="small-muted">Try: {config["examples"][0]}</span></div>',
        unsafe_allow_html=True,
    )


# ---------------------------
# Compact status
# ---------------------------
if st.session_state.processed:
    rag = st.session_state.rag
    readable = len([r for r in rag.get("source_results", []) if r["status"] == "success"])
    failed = len(rag.get("source_results", [])) - readable

    c1, c2, c3 = st.columns(3)
    c1.metric("Sources", readable)
    c2.metric("Chunks", len(rag["chunks"]))
    c3.metric("Questions", len(st.session_state.history))

    if failed:
        with st.expander(f"⚠️ {failed} source{'s' if failed != 1 else ''} failed"):
            for item in rag.get("source_results", []):
                if item["status"] == "failed":
                    st.write(f"**{hostname(item['url'])}** — {item['reason']}")


# ---------------------------
# Ask
# ---------------------------
st.markdown("## 💬 Ask")

if st.session_state.processed:
    research_mode = st.segmented_control(
        "Mode",
        ["Answer", "Compare Sources"],
        default="Answer",
        label_visibility="collapsed",
    )

    question = st.text_area(
        "Question",
        placeholder=config["question"],
        height=90,
        label_visibility="collapsed",
    )

    ask = st.button("✨ Ask SourceLens", type="primary")

    if ask:
        if not question.strip():
            st.warning("Ask a question first.")
        elif research_mode == "Compare Sources" and len(
            [r for r in st.session_state.rag.get("source_results", []) if r["status"] == "success"]
        ) < 2:
            st.warning("Compare needs at least 2 readable sources.")
        else:
            rag = st.session_state.rag
            with st.spinner("Finding evidence..."):
                try:
                    if research_mode == "Compare Sources":
                        readable_sources = [
                            item["url"]
                            for item in rag.get("source_results", [])
                            if item["status"] == "success"
                        ]
                        results = retrieve_per_source(
                            rag["vectorstore"], question, readable_sources, k_per_source=top_k
                        )
                    else:
                        results = retrieve_documents(rag["vectorstore"], question, top_k)

                    context, source_numbers = build_context(results)
                except Exception as exc:
                    st.error(f"Retrieval failed: {exc}")
                    st.stop()

            if not results:
                st.warning("No useful evidence was found in the provided sources.")
            else:
                llm_question = question
                if research_mode == "Compare Sources":
                    llm_question = (
                        f"Compare the supplied sources for this question:\n\n{question}\n\n"
                        "Separate similarities, differences, source-specific findings, and takeaway."
                    )

                with st.spinner("Synthesizing..."):
                    try:
                        messages = rag["prompt"].invoke({"context": context, "question": llm_question})
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
                st.markdown(f'<div class="answer-card">{response.content}</div>', unsafe_allow_html=True)

                st.markdown("### 📚 Sources")
                cols = st.columns(min(3, max(1, len(source_numbers))))
                for idx, (source, number) in enumerate(sorted(source_numbers.items(), key=lambda x: x[1])):
                    with cols[idx % len(cols)]:
                        st.markdown(f"**[{number}] {source_label(source)}**")
                        st.caption(source)

                st.markdown("### 🔎 Evidence")
                st.caption("These are the exact chunks sent to the model.")
                for item in results:
                    document = item["document"]
                    score = item["score"]
                    number = item["source_number"]
                    source = document.metadata.get("source", "Unknown source")
                    with st.expander(f"[{number}] {source_label(source)} · {score:.0%}"):
                        st.write(document.page_content)
                        st.caption(source)

                st.button(
                    "🧭 See how this answer was built",
                    on_click=lambda: st.switch_page("pages/2_Architecture.py"),
                )
else:
    st.markdown(
        '<div class="card"><strong>👈 Add your sources</strong><br><span class="small-muted">Then ask a question.</span></div>',
        unsafe_allow_html=True,
    )


# ---------------------------
# History
# ---------------------------
if st.session_state.history:
    with st.expander("🕘 History"):
        for item in reversed(st.session_state.history):
            st.markdown(f"**{item['mode']}** · {item['question']}")
            st.markdown(item["answer"])
            st.divider()

st.divider()
st.caption("SourceLens AI · grounded answers from the sources you provide.")
