# main.py — Streamlit app with healthcheck, DEBUG diagnostics, and expandable source snippets

from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

# Core extraction deps (used internally by HybridExtractor)
# Note: for OCR paths, install system binaries (tesseract-ocr, poppler-utils) at build time.
import pytesseract  # noqa: F401
from PIL import Image  # noqa: F401
import pdf2image  # noqa: F401
import docx  # noqa: F401
import PyPDF2  # noqa: F401
from striprtf.striprtf import rtf_to_text  # noqa: F401

from legal_bert_rag import LegalBERTRAG
from utils.extractors import HybridExtractor


# ──────────────────────────── Page config ─────────────────────────────
st.set_page_config(
    page_title="Legal Document Scanner • Streamlit + Tesseract + Legal AI",
    page_icon="⚖️",
    layout="wide",
)

# ──────────────────────────── Session state ───────────────────────────
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "legal_doc" not in st.session_state:
    st.session_state.legal_doc = None
if "legal_rag" not in st.session_state:
    st.session_state.legal_rag = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

DEBUG = bool(st.secrets.get("DEBUG", False)) if hasattr(st, "secrets") else bool(os.getenv("DEBUG"))

# ──────────────────────────── UI: Uploader ────────────────────────────
st.title("⚖️ Legal Document Scanner • Streamlit + Tesseract + Legal AI")
st.markdown(
    "Upload a **PDF, DOCX, RTF or TXT** file. The app tries native text "
    "extraction first, then falls back to OCR with Tesseract if needed. "
    "Plus, get advanced Legal AI analysis with InLegalBERT!"
)

uploaded = st.file_uploader(
    "Upload legal document",
    type=["pdf", "docx", "doc", "rtf", "txt"],
    help="Maximum 200 MB per file",
)

if uploaded:
    st.success(f"File received: {uploaded.name}")

# ──────────────────────────── Basic extraction ────────────────────────
if st.button("🔍 Extract Text", type="primary"):
    if not uploaded:
        st.info("Please upload a document first.")
    else:
        extractor = HybridExtractor()
        with st.spinner("Processing …"):
            text = extractor.extract(uploaded.name, uploaded)
        if text:
            st.success("✅ Extraction complete")
            wc, cc = len(text.split()), len(text)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Words", wc)
            with col2:
                st.metric("Characters", cc)
            st.download_button(
                "📥 Download text",
                text,
                file_name=f"{Path(uploaded.name).stem}_extracted.txt",
                mime="text/plain",
            )
            with st.expander("📄 View text"):
                st.text_area("Extracted content", text, height=400)
        else:
            st.error("No text could be extracted.")

# ──────────────────────────── Legal AI Section ────────────────────────
st.divider()
st.subheader("🤖 Advanced Legal AI Analysis")

if st.button("🔍 Analyze with Legal AI", type="primary", key="legal_analysis_btn"):
    if not uploaded:
        st.info("Please upload a document before analysis.")
    else:
        # Reset state for new analysis
        st.session_state.analysis_complete = False
        st.session_state.legal_doc = None
        st.session_state.legal_rag = None
        st.session_state.retriever = None

        # Initialize Legal RAG (reads secrets/env internally)
        legal_rag = LegalBERTRAG()

        # Healthcheck once
        hc = legal_rag.healthcheck()
        if hc != "OK":
            st.warning(f"Gemini healthcheck: {hc}")

        with st.spinner("Processing with InLegalBERT + spaCy..."):
            doc = legal_rag.process_document(uploaded.name, uploaded)
        if doc:
            st.session_state.legal_doc = doc
            st.session_state.legal_rag = legal_rag
            st.session_state.retriever = legal_rag.setup_retriever(doc)
            st.session_state.analysis_complete = True
            st.rerun()
        else:
            st.error("Failed to process document. Please check the file format.")

# ──────────────────────────── Results & Q/A ───────────────────────────
if st.session_state.analysis_complete and st.session_state.legal_doc:
    doc = st.session_state.legal_doc
    st.success("✅ Legal AI analysis complete!")

    # Dashboard
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Document Type", doc.meta["document_type"])
        st.metric("Companies", doc.meta["entity_counts"]["companies"])
    with col2:
        st.metric("People", doc.meta["entity_counts"]["people"])
        st.metric("Dates", doc.meta["entity_counts"]["dates"])
    with col3:
        st.metric("Money References", doc.meta["entity_counts"]["money"])
        st.metric("Word Count", len(doc.content.split()))

    # Entities
    with st.expander("📋 Legal Entities Detected"):
        entities = doc.meta["legal_entities"]
        if entities.get("companies"):
            st.write("**🏢 Companies:**")
            for company in entities["companies"][:8]:
                st.write(f"• {company}")
        if entities.get("people"):
            st.write("**👤 People:**")
            for person in entities["people"][:8]:
                st.write(f"• {person}")
        if entities.get("dates"):
            st.write("**📅 Important Dates:**")
            for date in entities["dates"][:8]:
                st.write(f"• {date}")
        if entities.get("money"):
            st.write("**💰 Financial References:**")
            for money in entities["money"][:8]:
                st.write(f"• {money}")

    # Q&A
    st.subheader("💬 Legal Document Q&A")
    st.write("Ask specific questions about this legal document:")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Who are the parties involved?", key="parties_btn"):
            st.session_state.current_question = "Who are the parties involved in this document?"
        if st.button("What are the key dates?", key="dates_btn"):
            st.session_state.current_question = "What are the important dates mentioned in this document?"
    with c2:
        if st.button("What are the financial terms?", key="financial_btn"):
            st.session_state.current_question = "What are the financial terms and amounts mentioned?"
        if st.button("What are the main obligations?", key="obligations_btn"):
            st.session_state.current_question = "What are the main obligations and responsibilities outlined?"

    query = st.text_input(
        "Your question:",
        value=st.session_state.get("current_question", ""),
        key="legal_query_input",
    )

    if st.button("Get Answer", key="answer_btn") and query:
        with st.spinner("Generating legal analysis..."):
            answer = st.session_state.legal_rag.query_document(st.session_state.retriever, query)
        st.markdown("**🤖 Legal AI Answer:**")
        st.write(answer)

        # DEBUG diagnostics if the provider returned an empty/error response
        if DEBUG and (str(answer).startswith("Empty model content") or str(answer).startswith("Error generating response")):
            with st.expander("Diagnostics"):
                st.code(str(answer))

        # Show source snippets for trust and traceability
        retrieved = getattr(st.session_state.legal_rag, "_last_retrieved", [])
        if retrieved:
            st.subheader("🔎 Sources")
            for rc in retrieved[:5]:
                with st.expander(f"Chunk {rc.idx}  • score={rc.score:.2f} • chars {rc.start}-{rc.end}"):
                    st.write(rc.text)

        if "current_question" in st.session_state:
            del st.session_state.current_question
else:
    if not uploaded:
        st.info("👆 Upload a document to begin.")

# ──────────────────────────── Footer ──────────────────────────────────
st.divider()
st.markdown(
    """
### Features:
- **Basic Mode**: Hybrid OCR + text extraction for all document types
- **Legal AI Mode**: InLegalBERT embeddings + entity extraction + intelligent Q&A
- **Document Types**: PDF, DOCX, RTF, TXT with OCR fallback
- **Legal Entities**: Companies, people, dates, financial amounts
- **Smart Classification**: Contract, filing, corporate, insurance, real estate documents
"""
)
