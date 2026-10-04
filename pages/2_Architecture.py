import html
import numpy as np
import streamlit as st

from src.retriever import source_label

st.set_page_config(page_title="SourceLens · RAG Explorer", page_icon="🧭", layout="wide")

st.markdown("""
<style>
.block-container{max-width:1220px;padding-top:3.4rem!important;padding-bottom:3rem}
.rag-title{font-size:2.55rem;font-weight:850;letter-spacing:-.05em;line-height:1.05}.rag-sub{color:#94a3b8;margin:.4rem 0 1rem}
.trace{border:1px solid rgba(124,92,255,.18);background:linear-gradient(90deg,rgba(124,92,255,.08),rgba(34,211,238,.035));border-radius:16px;padding:1rem;margin:.8rem 0 1.2rem}
.trace-row{display:flex;align-items:center;gap:.45rem;overflow-x:auto}.trace-node{min-width:105px;text-align:center;border:1px solid rgba(255,255,255,.09);border-radius:12px;padding:.65rem .5rem;background:rgba(255,255,255,.025)}.trace-node strong{display:block;font-size:.82rem}.trace-node span{font-size:.67rem;color:#94a3b8}.trace-arrow{color:#7c5cff;font-size:1.1rem}
.stage-note{border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.025);border-radius:14px;padding:.9rem 1rem;margin:.7rem 0}.muted{color:#94a3b8}.chunk-card{border:1px solid #292e3e;border-left:3px solid #7c5cff;border-radius:0 12px 12px 0;background:rgba(124,92,255,.045);padding:.75rem;margin:.5rem 0}.chunk-meta{font-size:.72rem;color:#a78bfa;font-weight:800}.chunk-text{font-size:.82rem;color:#cbd5e1;line-height:1.5;margin-top:.25rem}
.vector{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;background:#0b0e15;border:1px solid #252a39;border-radius:12px;padding:.8rem;color:#a5b4fc;font-size:.78rem;overflow-x:auto}.packet{background:#0b0e15;border:1px solid #252a39;border-radius:14px;padding:1rem}.packet-label{font-size:.68rem;color:#a78bfa;font-weight:800;letter-spacing:.08em}.packet-body{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.76rem;line-height:1.55;color:#cbd5e1;white-space:pre-wrap}.answer-box{border:1px solid rgba(124,92,255,.2);border-radius:15px;background:linear-gradient(135deg,rgba(124,92,255,.1),rgba(34,211,238,.04));padding:1rem;line-height:1.6}
</style>
""", unsafe_allow_html=True)

if st.button("← Back to SourceLens"):
    st.switch_page("app.py")

rag=st.session_state.get("rag"); processed=st.session_state.get("processed",False)
if not processed or not rag:
    st.markdown('<div class="rag-title">🧭 Explore your RAG pipeline</div><div class="rag-sub">Process sources on the main page first.</div>',unsafe_allow_html=True)
    st.info("Process at least one source on the main page to see the live pipeline.")
    st.stop()

chunks=rag.get("chunks",[]); results=st.session_state.get("last_results",[]); question=st.session_state.get("last_question","")
answer=st.session_state.get("last_answer",""); mode=st.session_state.get("last_mode","Answer")
sources=rag.get("source_results",[]); ready=[x for x in sources if x.get("status")=="success"]

def short(s,n=520):
    s=" ".join(str(s).split()); return s if len(s)<=n else s[:n].rstrip()+"…"

def esc(s): return html.escape(str(s))

retrieved=[]
for i,item in enumerate(results,1):
    doc=item["document"]; src=doc.metadata.get("source","Unknown source")
    retrieved.append({"rank":i,"number":item.get("source_number",1),"source":source_label(src),"url":src,"score":float(item.get("score",0)),"text":short(doc.page_content,700)})

st.markdown('<div class="rag-title">🧭 Explore your RAG pipeline</div><div class="rag-sub">Click through the pipeline like an interactive explainer — using your actual sources, retrieved chunks and query.</div>',unsafe_allow_html=True)
st.markdown(f'<div class="trace"><div class="trace-row"><div class="trace-node">📚<strong>{len(ready)} sources</strong><span>loaded</span></div><div class="trace-arrow">→</div><div class="trace-node">✂️<strong>{len(chunks):,} chunks</strong><span>split</span></div><div class="trace-arrow">→</div><div class="trace-node">🔢<strong>384-D</strong><span>embeddings</span></div><div class="trace-arrow">→</div><div class="trace-node">🎯<strong>{len(results)} selected</strong><span>retrieval</span></div><div class="trace-arrow">→</div><div class="trace-node">🧠<strong>GPT-OSS 20B</strong><span>Groq API</span></div><div class="trace-arrow">→</div><div class="trace-node">✨<strong>Answer</strong><span>citations</span></div></div></div>',unsafe_allow_html=True)

st.caption("This explains SourceLens' RAG pipeline — not the internal neural layers of GPT-OSS 20B.")

stages=["01 Sources","02 Chunking","03 Embeddings","04 Retrieval","05 LLM input","06 Answer"]
current=st.session_state.get("viz_stage", "04 Retrieval" if results else "01 Sources")
cols=st.columns(6)
for i,label in enumerate(stages):
    with cols[i]:
        if st.button(label, key=f"viz_{i}", use_container_width=True, type="primary" if current==label else "secondary"):
            st.session_state.viz_stage=label; st.rerun()
current=st.session_state.get("viz_stage",current)

if current=="01 Sources":
    st.markdown("### 01 · Sources enter the system")
    st.write("SourceLens loads the URLs you supplied. A failed page is isolated; readable pages continue into the pipeline.")
    cols=st.columns(min(4,max(1,len(sources))))
    for i,item in enumerate(sources):
        with cols[i%len(cols)]:
            icon="🟢" if item["status"]=="success" else "🔴"
            st.markdown(f'<div class="stage-note"><strong>{icon} {esc(source_label(item["url"]))}</strong><br><span class="muted">{item.get("documents",0)} document(s)</span></div>',unsafe_allow_html=True)

elif current=="02 Chunking":
    st.markdown("### 02 · Documents become searchable chunks")
    st.write(f"The index currently contains **{len(chunks):,} chunks**. Below are a few real chunks from this run.")
    for i,c in enumerate(chunks[:8],1):
        src=source_label(c.metadata.get("source","Unknown"))
        st.markdown(f'<div class="chunk-card"><div class="chunk-meta">CHUNK {i} · {esc(src)} · {len(c.page_content):,} characters</div><div class="chunk-text">{esc(short(c.page_content,420))}</div></div>',unsafe_allow_html=True)
    if len(chunks)>8: st.caption(f"Showing 8 of {len(chunks):,}. The remaining chunks are also stored in Chroma.")

elif current=="03 Embeddings":
    st.markdown("### 03 · Text becomes vectors")
    st.write("SourceLens uses **all-MiniLM-L6-v2 locally**. Each chunk becomes a 384-dimensional embedding before it is stored in Chroma.")
    if chunks and rag.get("embedding_model"):
        try:
            vec=rag["embedding_model"].embed_query(chunks[0].page_content)
            st.markdown('<div class="vector">['+", ".join(f"{x:.5f}" for x in vec[:16])+', …]</div>',unsafe_allow_html=True)
            a,b=st.columns(2); a.metric("Dimensions",len(vec)); b.metric("Embedding location","Local")
        except Exception as exc: st.warning(f"Could not preview an embedding: {exc}")
    st.markdown('<div class="stage-note"><strong>Why?</strong><br><span class="muted">The vector lets SourceLens compare meaning between your question and stored chunks instead of relying only on exact keyword matches.</span></div>',unsafe_allow_html=True)

elif current=="04 Retrieval":
    st.markdown("### 04 · Retrieval decides what the model gets")
    if not results:
        st.info("Run a question on the main page first. Then return here to see the exact retrieval trace.")
    else:
        st.markdown(f'<div class="stage-note"><strong>Question</strong><br>{esc(question)}<br><span class="muted">Mode: {esc(mode)} · {len(results)} chunks selected</span></div>',unsafe_allow_html=True)
        # Real 2-D PCA projection of the retrieved embeddings + query.
        try:
            texts=[x["document"].page_content for x in results]
            vectors=np.asarray(rag["embedding_model"].embed_documents(texts),dtype=float)
            q=np.asarray(rag["embedding_model"].embed_query(question),dtype=float)
            m=np.vstack([vectors,q]); m-=m.mean(axis=0,keepdims=True)
            _,_,vt=np.linalg.svd(m,full_matrices=False); coords=m@vt[:2].T
            lo=coords.min(axis=0); hi=coords.max(axis=0); span=np.maximum(hi-lo,1e-9)
            svg=[]
            for i,item in enumerate(retrieved):
                x=55+260*((coords[i,0]-lo[0])/span[0]); y=30+230*(1-(coords[i,1]-lo[1])/span[1]);
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="#211b3d" stroke="#8b7cff" stroke-width="2"/><text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" fill="#c4b5fd" font-size="10">#{item["rank"]}</text>')
            x=55+260*((coords[-1,0]-lo[0])/span[0]); y=30+230*(1-(coords[-1,1]-lo[1])/span[1]); svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="21" fill="#17313a" stroke="#22d3ee" stroke-width="3"/><text x="{x:.1f}" y="{y+5:.1f}" text-anchor="middle" fill="#67e8f9" font-size="11" font-weight="800">Q</text>')
            st.markdown('<svg viewBox="0 0 370 285" style="width:100%;max-height:330px;background:radial-gradient(circle,#151a29,#0b0e15);border:1px solid #252a39;border-radius:14px">'+''.join(svg)+'<text x="12" y="270" fill="#64748b" font-size="10">2-D PCA projection of actual retrieved embeddings · Q = query</text></svg>',unsafe_allow_html=True)
        except Exception:
            st.info("Vector projection unavailable for this run.")
        for r in retrieved:
            st.markdown(f'<div class="stage-note"><strong>#{r["rank"]} · [{r["number"]}] {esc(r["source"])}</strong> <span class="muted">· similarity {r["score"]:.0%}</span><br><span class="muted">{esc(short(r["text"],300))}</span></div>',unsafe_allow_html=True)

elif current=="05 LLM input":
    st.markdown("### 05 · The exact context sent to GPT-OSS 20B")
    st.write("Retrieval happens first. The hosted model receives the question plus the selected source passages; it does not fetch the webpages itself.")
    st.markdown(f'<div class="packet"><div class="packet-label">USER QUESTION</div><div class="packet-body">{esc(question or "No question yet")}</div></div>',unsafe_allow_html=True)
    for r in retrieved:
        st.markdown(f'<div class="packet" style="margin-top:8px"><div class="packet-label">SOURCE [{r["number"]}] · {esc(r["source"])} · {r["score"]:.0%}</div><div class="packet-body">{esc(r["text"])}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="stage-note"><strong>Model endpoint</strong><br><span class="muted">GPT-OSS 20B · Groq API · not downloaded locally</span></div>',unsafe_allow_html=True)

elif current=="06 Answer":
    st.markdown("### 06 · Grounded answer")
    if answer:
        st.markdown(f'<div class="answer-box">{esc(answer)}</div>',unsafe_allow_html=True)
        st.markdown('<div class="stage-note"><strong>Trace complete</strong><br><span class="muted">The answer above was generated after retrieval and can cite the source numbers attached to the context.</span></div>',unsafe_allow_html=True)
    else:
        st.info("Run a question on the main page first.")

with st.expander("🔬 See the exact retrieved chunks"):
    if results:
        for item in results:
            doc=item["document"]; src=doc.metadata.get("source","Unknown")
            st.markdown(f'**[{item["source_number"]}] {source_label(src)} · {item["score"]:.0%} relevance**')
            st.write(doc.page_content)
            st.divider()
    else: st.info("No retrieval run yet.")
