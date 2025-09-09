# utils/model_cache.py
from __future__ import annotations
import streamlit as st

@st.cache_resource(show_spinner="Loading InLegalBERT embeddings…")
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("nlpaueb/legal-bert-base-uncased")

@st.cache_resource(show_spinner="Loading spaCy pipeline…")
def get_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None
