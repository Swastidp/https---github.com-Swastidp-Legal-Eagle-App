# utils/model_cache.py

from __future__ import annotations

import streamlit as st


@st.cache_resource(show_spinner="Loading InLegalBERT embeddings…")
def get_embedder():
    """
    Returns a cached SentenceTransformer for legal-domain embeddings.
    """
    from sentence_transformers import SentenceTransformer  # sbert loader [docs: SentenceTransformer]
    # Hugging Face model: nlpaueb/legal-bert-base-uncased
    # Good balance for legal text similarity and retrieval
    return SentenceTransformer("nlpaueb/legal-bert-base-uncased")


@st.cache_resource(show_spinner="Loading spaCy pipeline…")
def get_spacy():
    """
    Loads a spaCy English pipeline with a 'lg' > 'sm' preference:
    - Try en_core_web_lg first (better vectors & NER precision).
    - Fallback to en_core_web_sm if lg isn't installed.
    - If spaCy unavailable or no model present, return None.
    """
    try:
        import spacy
        # Prefer the large pipeline for higher NER precision if present. [spaCy models]
        try:
            return spacy.load("en_core_web_lg")  # requires model to be installed
        except Exception:
            try:
                return spacy.load("en_core_web_sm")
            except Exception:
                return None
    except Exception:
        return None
