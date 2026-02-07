"""DocQA Engine: Upload documents, ask questions, get cited answers with a prompt lab."""

from __future__ import annotations

import asyncio
from pathlib import Path

import streamlit as st

from docqa_engine.pipeline import DocQAPipeline

DEMO_DIR = Path(__file__).parent / "demo_docs"


def get_pipeline() -> DocQAPipeline:
    """Get or create the pipeline in session state."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = DocQAPipeline()
    return st.session_state.pipeline


def run_async(coro):
    """Run an async coroutine from synchronous Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def render_documents_tab(pipeline: DocQAPipeline) -> None:
    """Render the Documents tab for ingestion and overview."""
    st.subheader("Document Ingestion")

    source = st.sidebar.radio("Document Source", ["Demo Documents", "Upload Files"])

    if source == "Demo Documents":
        st.info("Demo mode: loads sample documents from the demo_docs/ directory.")
        if st.button("Load Demo Documents"):
            count = 0
            if DEMO_DIR.exists():
                for ext in ("*.md", "*.txt"):
                    for f in sorted(DEMO_DIR.glob(ext)):
                        pipeline.ingest(str(f))
                        count += 1
            if count:
                st.success(f"Loaded {count} demo document(s)")
            else:
                st.warning("No demo documents found in demo_docs/")
    else:
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["txt", "md", "pdf", "docx", "csv"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Ingest Uploaded Files"):
            for uf in uploaded_files:
                content = uf.read()
                if uf.name.endswith((".txt", ".md", ".csv")):
                    text = content.decode("utf-8", errors="replace")
                    pipeline.ingest_text(text, filename=uf.name)
                else:
                    import tempfile

                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}") as tmp:
                        tmp.write(content)
                        pipeline.ingest(tmp.name)
            st.success(f"Ingested {len(uploaded_files)} file(s)")

    # Display ingested documents
    stats = pipeline.get_stats()
    doc_count = stats.get("documents", 0)
    chunk_count = stats.get("chunk_count", 0)

    if doc_count > 0:
        st.divider()
        st.subheader("Ingested Documents")
        col1, col2, col3 = st.columns(3)
        col1.metric("Documents", doc_count)
        col2.metric("Chunks", chunk_count)
        col3.metric("Total Chars", f"{stats.get('total_chars', 0):,}")
    else:
        st.caption("No documents ingested yet. Load demo docs or upload files to get started.")


def render_ask_tab(pipeline: DocQAPipeline) -> None:
    """Render the Ask Questions tab."""
    st.subheader("Ask a Question")

    stats = pipeline.get_stats()
    if stats.get("chunk_count", 0) == 0:
        st.warning("No documents loaded. Go to the Documents tab first.")
        return

    question = st.text_input("Your question:", placeholder="What is a list comprehension in Python?")

    # Template selector
    templates = pipeline.prompt_library.list_templates()
    template_names = [t.name for t in templates]
    selected_template = st.selectbox("Prompt template:", template_names)

    if st.button("Ask") and question:
        with st.spinner("Searching and generating answer..."):
            answer = run_async(pipeline.ask(question, template=selected_template))

        st.markdown(f"**Answer:**\n\n{answer.answer_text}")

        if answer.citations:
            st.divider()
            st.markdown("**Citations:**")
            for i, cit in enumerate(answer.citations, 1):
                page_ref = f" (p.{cit.page_number})" if cit.page_number else ""
                source_name = cit.source or "unknown"
                st.markdown(
                    f"{i}. **{source_name}**{page_ref} "
                    f"(score: {cit.relevance_score:.3f}) -- "
                    f"_{cit.content_snippet[:120]}..._"
                )

        st.caption(f"Provider: {answer.provider} | Model: {answer.model} | Tokens: {answer.tokens_used}")


def render_prompt_lab_tab(pipeline: DocQAPipeline) -> None:
    """Render the Prompt Lab tab for A/B comparison."""
    st.subheader("Prompt A/B Comparison")

    stats = pipeline.get_stats()
    if stats.get("chunk_count", 0) == 0:
        st.warning("No documents loaded. Go to the Documents tab first.")
        return

    templates = pipeline.prompt_library.list_templates()
    template_names = [t.name for t in templates]

    if len(template_names) < 2:
        st.info("Need at least 2 prompt templates for comparison.")
        return

    question = st.text_input(
        "Question for comparison:",
        key="lab_question",
        placeholder="Explain overfitting in machine learning",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Template A**")
        template_a = st.selectbox("Select Template A:", template_names, key="tmpl_a")
        tmpl_a = pipeline.prompt_library.get_template(template_a)
        st.code(tmpl_a.template, language="text")
        if tmpl_a.description:
            st.caption(tmpl_a.description)

    with col_b:
        st.markdown("**Template B**")
        template_b = st.selectbox(
            "Select Template B:",
            template_names,
            index=min(1, len(template_names) - 1),
            key="tmpl_b",
        )
        tmpl_b = pipeline.prompt_library.get_template(template_b)
        st.code(tmpl_b.template, language="text")
        if tmpl_b.description:
            st.caption(tmpl_b.description)

    if st.button("Compare") and question:
        with st.spinner("Running A/B comparison..."):
            comparison = run_async(pipeline.compare_templates(question, template_a, template_b))

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"**Result A ({template_a}):**")
            st.markdown(comparison.answer_a.answer_text)
            st.caption(f"Tokens: {comparison.answer_a.tokens_used}")

        with col_b:
            st.markdown(f"**Result B ({template_b}):**")
            st.markdown(comparison.answer_b.answer_text)
            st.caption(f"Tokens: {comparison.answer_b.tokens_used}")


def render_stats_tab(pipeline: DocQAPipeline) -> None:
    """Render the Stats tab with pipeline metrics."""
    st.subheader("Pipeline Statistics")

    stats = pipeline.get_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", stats.get("documents", 0))
    col2.metric("Chunks", stats.get("chunk_count", 0))
    col3.metric("Total Chars", f"{stats.get('total_chars', 0):,}")

    # Template overview
    st.divider()
    st.subheader("Prompt Templates")
    for tmpl in pipeline.prompt_library.list_templates():
        with st.expander(f"{tmpl.name}"):
            st.code(tmpl.template, language="text")
            if tmpl.description:
                st.caption(tmpl.description)
            if tmpl.variables:
                st.caption(f"Variables: {', '.join(tmpl.variables)}")


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(page_title="DocQA Engine", layout="wide")
    st.title("DocQA Engine")
    st.caption("Upload documents, ask questions -- get cited answers with a prompt engineering lab")

    pipeline = get_pipeline()

    tab_docs, tab_ask, tab_lab, tab_stats = st.tabs(["Documents", "Ask Questions", "Prompt Lab", "Stats"])

    with tab_docs:
        render_documents_tab(pipeline)

    with tab_ask:
        render_ask_tab(pipeline)

    with tab_lab:
        render_prompt_lab_tab(pipeline)

    with tab_stats:
        render_stats_tab(pipeline)


if __name__ == "__main__":
    main()
