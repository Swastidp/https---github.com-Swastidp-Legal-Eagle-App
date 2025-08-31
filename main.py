# main.py - COMPLETE FIXED VERSION with session state
import io, os, tempfile
from pathlib import Path

import streamlit as st
import pytesseract
from PIL import Image
import pdf2image
import docx
import PyPDF2
from striprtf.striprtf import rtf_to_text

# New imports for Legal AI
from legal_bert_rag import LegalBERTRAG
from utils.extractors import HybridExtractor

# ──────────────────────────── Page config ─────────────────────────────

st.set_page_config(
    page_title="Legal Document Scanner • Streamlit + Tesseract + Legal AI",
    page_icon="⚖️",
    layout="wide",
)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'legal_doc' not in st.session_state:
    st.session_state.legal_doc = None
if 'legal_rag' not in st.session_state:
    st.session_state.legal_rag = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# ──────────────────────────── UI ──────────────────────────────────────

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
    
    # Basic extraction
    if st.button("🔍 Extract Text", type="primary"):
        extractor = HybridExtractor()
        with st.spinner("Processing …"):
            text = extractor.extract(uploaded.name, uploaded)

        if text:
            st.success("✅ Extraction complete")
            wc, cc = len(text.split()), len(text)
            col1, col2 = st.columns(2)
            col1.metric("Words", wc)
            col2.metric("Characters", cc)

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

    # ──────────────────────────── Legal AI Section ─────────────────────────────
    
    st.divider()
    st.subheader("🤖 Advanced Legal AI Analysis")

    # Legal AI Analysis Button
    if st.button("🔍 Analyze with Legal AI", type="primary", key="legal_analysis_btn"):
        # Reset session state for new analysis
        st.session_state.analysis_complete = False
        st.session_state.legal_doc = None
        st.session_state.legal_rag = None
        st.session_state.retriever = None
        
        # Initialize Legal RAG (with hardcoded API key)
        legal_rag = LegalBERTRAG()  # No API key needed - it's hardcoded
        
        with st.spinner("Processing with InLegalBERT + spaCy..."):
            doc = legal_rag.process_document(uploaded.name, uploaded)
            
        if doc:
            # Store in session state
            st.session_state.legal_doc = doc
            st.session_state.legal_rag = legal_rag
            st.session_state.retriever = legal_rag.setup_retriever(doc)
            st.session_state.analysis_complete = True
            st.rerun()  # Refresh to show results
        else:
            st.error("Failed to process document. Please check the file format.")

    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.legal_doc:
        doc = st.session_state.legal_doc
        
        st.success("✅ Legal AI analysis complete!")
        
        # Document insights dashboard
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
        
        # Detailed entity breakdown
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
        
        # Legal Q&A Interface
        st.subheader("💬 Legal Document Q&A")
        st.write("Ask specific questions about this legal document:")
        
        # Suggested questions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Who are the parties involved?", key="parties_btn"):
                st.session_state.current_question = "Who are the parties involved in this document?"
            if st.button("What are the key dates?", key="dates_btn"):
                st.session_state.current_question = "What are the important dates mentioned in this document?"
        with col2:
            if st.button("What are the financial terms?", key="financial_btn"):
                st.session_state.current_question = "What are the financial terms and amounts mentioned?"
            if st.button("What are the main obligations?", key="obligations_btn"):
                st.session_state.current_question = "What are the main obligations and responsibilities outlined?"
        
        # Query input
        query = st.text_input(
            "Your question:", 
            value=st.session_state.get('current_question', ''),
            key="legal_query_input"
        )
        
        # Answer question button
        if st.button("Get Answer", key="answer_btn") and query:
            with st.spinner("Generating legal analysis..."):
                answer = st.session_state.legal_rag.query_document(st.session_state.retriever, query)
                
            st.markdown("**🤖 Legal AI Answer:**")
            st.write(answer)
            
            # Clear the current question
            if 'current_question' in st.session_state:
                del st.session_state.current_question

else:
    st.info("👆 Upload a document to begin.")

# ──────────────────────────── Footer ─────────────────────────────
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
