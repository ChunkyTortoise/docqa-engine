"""DocQA Engine: Upload documents, ask questions, get cited answers."""

from pathlib import Path

import numpy as np
import streamlit as st

from docqa_engine.answer import generate_answer
from docqa_engine.cost_tracker import CostTracker
from docqa_engine.ingest import DocumentChunk, ingest_file, ingest_txt
from docqa_engine.prompt_lab import PromptLab
from docqa_engine.retriever import HybridRetriever

DEMO_DIR = Path(__file__).parent / "demo_data"


def init_session_state():
    if "retriever" not in st.session_state:
        st.session_state.retriever = HybridRetriever()
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "cost_tracker" not in st.session_state:
        st.session_state.cost_tracker = CostTracker()
    if "prompt_lab" not in st.session_state:
        st.session_state.prompt_lab = PromptLab()
        st.session_state.prompt_lab.create_version(
            "default",
            "Answer based on the context. Cite sources.\n\nContext:\n{context}\n\nQ: {question}\nA:",
        )
    if "history" not in st.session_state:
        st.session_state.history = []


def load_demo_documents():
    """Load demo documents into the retriever."""
    for f in DEMO_DIR.glob("*.txt"):
        content = f.read_text()
        result = ingest_txt(content, filename=f.name)
        st.session_state.retriever.add_chunks(result.chunks)
        st.session_state.chunks.extend(result.chunks)


def main():
    st.set_page_config(page_title="DocQA Engine", page_icon="📄", layout="wide")
    st.title("📄 DocQA Engine")
    st.caption("Upload documents, ask questions, get cited answers")

    init_session_state()

    # Sidebar: document management
    st.sidebar.header("Documents")

    if st.sidebar.button("Load Demo Documents"):
        load_demo_documents()
        st.sidebar.success(f"Loaded {len(st.session_state.chunks)} chunks from demo docs")

    uploaded = st.sidebar.file_uploader(
        "Upload documents", type=["pdf", "docx", "txt", "csv"], accept_multiple_files=True
    )
    if uploaded:
        for f in uploaded:
            # Save temporarily and ingest
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{f.name}") as tmp:
                tmp.write(f.read())
                result = ingest_file(tmp.name)
            st.session_state.retriever.add_chunks(result.chunks)
            st.session_state.chunks.extend(result.chunks)
        st.sidebar.success(f"Ingested {len(uploaded)} file(s), {len(st.session_state.chunks)} total chunks")

    st.sidebar.metric("Indexed Chunks", len(st.session_state.chunks))

    # Tabs
    tab_qa, tab_lab, tab_costs = st.tabs(["💬 Q&A", "🧪 Prompt Lab", "💰 Costs"])

    # Q&A tab
    with tab_qa:
        question = st.text_input("Ask a question about your documents:")
        if question and st.button("Get Answer"):
            if not st.session_state.chunks:
                st.warning("Load demo documents or upload files first.")
            else:
                import asyncio
                results = asyncio.get_event_loop().run_until_complete(
                    st.session_state.retriever.search(question, top_k=5)
                )
                answer = asyncio.get_event_loop().run_until_complete(
                    generate_answer(question, results)
                )

                st.session_state.cost_tracker.record_query(
                    answer.question[:20], answer.question, answer.provider,
                    answer.model, answer.tokens_used // 2, answer.tokens_used // 2,
                )
                st.session_state.history.append(answer)

                st.markdown(f"**Answer:**\n\n{answer.answer_text}")

                if answer.citations:
                    st.markdown("**Sources:**")
                    for i, c in enumerate(answer.citations, 1):
                        page_ref = f" (p.{c.page_number})" if c.page_number else ""
                        st.markdown(f"{i}. **{c.source}**{page_ref} — _{c.content_snippet[:100]}..._")

        if st.session_state.history:
            st.divider()
            st.subheader("History")
            for ans in reversed(st.session_state.history[-10:]):
                with st.expander(f"Q: {ans.question[:60]}"):
                    st.write(ans.answer_text)
                    st.caption(f"Provider: {ans.provider} | Tokens: {ans.tokens_used}")

    # Prompt Lab tab
    with tab_lab:
        st.subheader("Prompt Versions")
        lab = st.session_state.prompt_lab

        versions = lab.list_versions()
        for v in versions:
            with st.expander(f"{v.name} ({v.version_id})"):
                st.code(v.template, language="text")
                st.caption(f"Temperature: {v.temperature} | Max tokens: {v.max_tokens}")
                stats = lab.get_version_stats(v.version_id)
                if stats["runs"] > 0:
                    st.json(stats)

        st.divider()
        st.subheader("Create Version")
        name = st.text_input("Version name:")
        template = st.text_area("Template (use {context} and {question}):")
        temp = st.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
        if st.button("Create") and name and template:
            v = lab.create_version(name, template, temperature=temp)
            st.success(f"Created version: {v.version_id}")

    # Cost tab
    with tab_costs:
        st.subheader("Cost Dashboard")
        tracker = st.session_state.cost_tracker
        summary = tracker.summary()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Queries", summary["total_queries"])
        col2.metric("Total Tokens", f"{summary['total_tokens']:,}")
        col3.metric("Total Cost", f"${summary['total_cost']:.4f}")

        if summary["by_provider"]:
            st.subheader("By Provider")
            for provider, data in summary["by_provider"].items():
                st.write(f"**{provider}**: {data['queries']} queries, "
                         f"{data['tokens']:,} tokens, ${data['cost']:.4f}")


if __name__ == "__main__":
    main()
