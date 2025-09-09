# utils/legal_processors.py
from __future__ import annotations
import re
from typing import Dict, List, Any
import streamlit as st
from utils.model_cache import get_embedder, get_spacy

class LegalNLPProcessor:
    def __init__(self):
        self._embedder = get_embedder()
        self._nlp = get_spacy()

    @property
    def embedder(self):
        return self._embedder

    @property
    def nlp(self):
        return self._nlp

    def extract_legal_entities(self, text: str) -> Dict[str, List[Any]]:
        companies: List[str] = []
        people: List[str] = []
        spacy_dates: List[str] = []

        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        companies.append(ent.text)
                    elif ent.label_ == "PERSON":
                        people.append(ent.text)
                    elif ent.label_ == "DATE":
                        spacy_dates.append(ent.text)
            except Exception as e:
                st.warning(f"spaCy processing failed: {e}")

        company_patterns = [
            r"\b[A-Z][A-Za-z&,\s.]+(?:Inc\.?|LLC|Corp\.?|Company|Co\.?|Ltd\.?)\b",
            r"\b[A-Z][A-Za-z&,\s.]+(?:Corporation|Partnership|LLP|LP)\b",
            r"\b[A-Z][A-Za-z\s]+(?:Bank|Trust|Group|Holdings|Enterprises)\b",
        ]
        for pat in company_patterns:
            companies.extend(re.findall(pat, text))

        people_patterns = [
            r"\b(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
            r"\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Jr\.?|Sr\.?|III|IV))?\b",
        ]
        for pat in people_patterns:
            people.extend(re.findall(pat, text))

        money_patterns = [
            r"\$[\d,]+\.?\d*",
            r"USD\s?[\d,]+\.?\d*",
            r"[\d,]+\.?\d*\s?dollars?",
            r"[\d,]+\.?\d*\s?USD",
        ]
        money: List[str] = []
        for pat in money_patterns:
            money.extend(re.findall(pat, text, flags=re.IGNORECASE))

        date_patterns = [
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            r"\b\d{1,2}-\d{1,2}-\d{4}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        ]
        regex_dates: List[str] = []
        for pat in date_patterns:
            regex_dates.extend(re.findall(pat, text, flags=re.IGNORECASE))

        def uniq(seq: List[str]) -> List[str]:
            seen, out = set(), []
            for s in seq:
                if s not in seen:
                    seen.add(s); out.append(s)
            return out

        all_dates = spacy_dates + regex_dates
        return {
            "dates": uniq(all_dates)[:10],
            "money": uniq(money)[:10],
            "companies": uniq(companies)[:10],
            "people": uniq(people)[:10],
        }

    def get_legal_embeddings(self, text: str):
        return self.embedder.encode(text)

    def analyze_document_type(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["whereas", "party agrees", "consideration", "terms and conditions", "this agreement", "contract", "covenant", "hereby agree"]):
            return "📄 Contract/Agreement"
        if any(k in t for k in ["plaintiff", "defendant", "court", "motion", "your honor", "civil action", "case no", "complaint", "docket"]):
            return "⚖️ Legal Filing"
        if any(k in t for k in ["board of directors", "shareholders", "bylaws", "articles of incorporation", "annual report", "sec filing", "proxy statement"]):
            return "🏢 Corporate Document"
        if any(k in t for k in ["policy", "coverage", "premium", "claim", "deductible", "insured", "policyholder", "beneficiary"]):
            return "🛡️ Insurance Document"
        if any(k in t for k in ["deed", "mortgage", "lease", "rental agreement", "property", "landlord", "tenant", "escrow"]):
            return "🏠 Real Estate Document"
        return "📋 General Legal Document"
