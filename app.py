# app.py

import streamlit as st
from rag_pipeline import build_rag_from_pdf_bytes


st.set_page_config(page_title="Multi-Modal RAG QA", layout="wide")


def main():
    st.title("Multi-Modal RAG QA – PDF Question Answering & Summary")

    st.write(
        "Upload an economic report PDF, then either:\n"
        "- Ask questions about it (Q&A mode), or\n"
        "- Generate a summary (Summarize mode).\n\n"
        "The system supports **text**, **tables**, and **OCR (images)**."
    )

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if not uploaded_file:
        st.info("Please upload a PDF to continue.")
        return

    pdf_bytes = uploaded_file.read()

    @st.cache_resource(show_spinner=True)
    def get_pipeline(data: bytes):
        return build_rag_from_pdf_bytes(data)

    rag = get_pipeline(pdf_bytes)

    st.success(f"Loaded: {uploaded_file.name}")

    st.markdown("---")
    st.markdown("Ask a Question")

    question = st.text_input("Enter your question:")
    top_k = st.slider("Context chunks to retrieve", 5, 20, 8)

    col1, col2 = st.columns(2)
    ask_clicked = col1.button("Ask")
    summarize_clicked = col2.button("Summarize Document")

    # ---------- Q&A Mode ----------
    if ask_clicked and question.strip():
        with st.spinner("Processing..."):
            result = rag.answer_question(question, top_k)

        st.subheader("Answer")
        st.write(result["answer"])

        # 📌 Display Metrics Dashboard
        st.subheader("📈 Retrieval Metrics")
        metrics = result["metrics"]

        colA, colB, colC = st.columns(3)
        colA.metric("Total Latency (ms)", f"{metrics['total_time_ms']:.2f}")
        colB.metric("Retrieval Time (ms)", f"{metrics['retrieval_time_ms']:.2f}")
        colC.metric("Generation Time (ms)", f"{metrics['generation_time_ms']:.2f}")

        colD, colE, colF = st.columns(3)
        colD.metric("Avg Distance", f"{metrics['avg_distance']:.2f}")
        colE.metric("Min Distance", f"{metrics['min_distance']:.2f}")
        colF.metric("Max Distance", f"{metrics['max_distance']:.2f}")

        # 📌 Source Viewer
        st.subheader("Sources Used")
        for i, src in enumerate(result["sources"], start=1):
            st.markdown(
                f"**Source {i}** – Page {src['page']} ({src['modality']})\n"
                f"> {src['snippet']}"
            )

    # ---------- Summarization Mode ----------
    if summarize_clicked:
        with st.spinner("Summarizing document..."):
            summary = rag.summarize_document()

        st.subheader("Executive Summary")
        st.write(summary)


if __name__ == "__main__":
    main()
