# utils/legal_processors.py
import streamlit as st
import spacy
import re
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any

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
            with st.spinner("Loading spaCy model..."):
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[Any]]:
        """Extract legal entities using spaCy + enhanced regex patterns"""
        doc = self.nlp(text)
        
        # Extract organizations and people using spaCy
        companies = []
        people = []
        spacy_dates = []
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                companies.append(ent.text)
            elif ent.label_ == "PERSON":
                people.append(ent.text)
            elif ent.label_ == "DATE":
                spacy_dates.append(ent.text)
        
        # Enhanced regex patterns for legal entities
        company_patterns = [
            r'\b[A-Z][a-zA-Z\s&,.]+(Inc\.?|LLC|Corp\.?|Company|Co\.?|Ltd\.?)\b',
            r'\b[A-Z][a-zA-Z\s&,.]+(?:Corporation|Partnership|LLP|LP)\b',
            r'\b[A-Z][a-zA-Z\s]+(Bank|Trust|Group|Holdings|Enterprises)\b'
        ]
        
        # Add regex-found companies
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            companies.extend(matches)
        
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
        
        # Combine spaCy dates with additional regex patterns
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
        ]
        
        regex_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
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
