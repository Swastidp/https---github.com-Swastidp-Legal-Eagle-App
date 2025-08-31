# utils/legal_processors.py - STREAMLIT CLOUD COMPATIBLE
import streamlit as st
import re
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any
import subprocess
import sys

class LegalNLPProcessor:
    def __init__(self):
        self._embedder = None
        self._nlp = None
    
    @property
    def embedder(self):
        if self._embedder is None:
            with st.spinner("Loading InLegalBERT model..."):
                self._embedder = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
        return self._embedder
    
    @property
    def nlp(self):
        if self._nlp is None:
            try:
                import spacy
                with st.spinner("Loading spaCy model..."):
                    self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                # Fallback: Download and install spaCy model
                st.info("Installing spaCy model for first-time use...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
                    ])
                    import spacy
                    self._nlp = spacy.load("en_core_web_sm")
                except Exception as e:
                    st.warning(f"spaCy model installation failed: {e}")
                    st.warning("Using regex-only entity extraction...")
                    self._nlp = None
        return self._nlp
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[Any]]:
        """Extract legal entities using spaCy + enhanced regex patterns (with fallback)"""
        
        # Try spaCy first
        companies = []
        people = []
        spacy_dates = []
        
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
        
        # Enhanced regex patterns (works with or without spaCy)
        company_patterns = [
            r'\b[A-Z][a-zA-Z\s&,.]+(Inc\.?|LLC|Corp\.?|Company|Co\.?|Ltd\.?)\b',
            r'\b[A-Z][a-zA-Z\s&,.]+(?:Corporation|Partnership|LLP|LP)\b',
            r'\b[A-Z][a-zA-Z\s]+(Bank|Trust|Group|Holdings|Enterprises)\b'
        ]
        
        # Add regex-found companies
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            companies.extend(matches)
        
        # People patterns (regex fallback)
        people_patterns = [
            r'\b(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Jr\.?|Sr\.?|III|IV))?\b'
        ]
        
        for pattern in people_patterns:
            matches = re.findall(pattern, text)
            people.extend(matches)
        
        # Money patterns
        money_patterns = [
            r'\$[\d,]+\.?\d*',  # $1,000.00
            r'USD\s?[\d,]+\.?\d*',  # USD 1000
            r'[\d,]+\.?\d*\s?dollars?',  # 1000 dollars
            r'[\d,]+\.?\d*\s?USD',  # 1000 USD
        ]
        
        money = []
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            money.extend(matches)
        
        # Date patterns (regex fallback)
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        regex_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            regex_dates.extend(matches)
        
        all_dates = spacy_dates + regex_dates
        
        return {
            'dates': list(set(all_dates))[:10],
            'money': list(set(money))[:10], 
            'companies': list(set(companies))[:10],
            'people': list(set(people))[:10],
        }
    
    def get_legal_embeddings(self, text: str):
        return self.embedder.encode(text)
    
    def analyze_document_type(self, text: str) -> str:
        """Enhanced document classification"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in [
            'whereas', 'party agrees', 'consideration', 'terms and conditions',
            'this agreement', 'contract', 'covenant', 'hereby agree'
        ]):
            return "📄 Contract/Agreement"
            
        elif any(term in text_lower for term in [
            'plaintiff', 'defendant', 'court', 'motion', 'your honor',
            'civil action', 'case no', 'complaint', 'docket'
        ]):
            return "⚖️ Legal Filing"
            
        elif any(term in text_lower for term in [
            'board of directors', 'shareholders', 'bylaws', 'articles of incorporation',
            'annual report', 'sec filing', 'proxy statement'
        ]):
            return "🏢 Corporate Document"
            
        elif any(term in text_lower for term in [
            'policy', 'coverage', 'premium', 'claim', 'deductible', 'insured',
            'policyholder', 'beneficiary'
        ]):
            return "🛡️ Insurance Document"
            
        elif any(term in text_lower for term in [
            'deed', 'mortgage', 'lease', 'rental agreement', 'property',
            'landlord', 'tenant', 'escrow'
        ]):
            return "🏠 Real Estate Document"
            
        else:
            return "📋 General Legal Document"
