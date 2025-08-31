# legal_bert_rag.py - GEMINI VERSION
import streamlit as st
from haystack import Document
from sentence_transformers import SentenceTransformer
from utils.extractors import HybridExtractor
from utils.legal_processors import LegalNLPProcessor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai

class LegalBERTRAG:
    def __init__(self, api_key: str = None):
        self.legal_processor = LegalNLPProcessor()
        # GEMINI API KEY - Get from https://aistudio.google.com/app/apikey
        self.api_key = api_key or "AIzaSyALMpDypmii_yg6-E1gvo1SUbeLXb4I4dg"  # Replace with your actual key
        self.document_chunks = []
        self.chunk_embeddings = []
        self._sentence_model = None
        
        # Gemini client using OpenAI compatibility
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    
    @property
    def sentence_model(self):
        """Lazy load sentence transformer model"""
        if self._sentence_model is None:
            with st.spinner("Loading InLegalBERT model..."):
                self._sentence_model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
        return self._sentence_model
        
    def process_document(self, file_name: str, file_buffer) -> Document:
        """Process document with legal AI analysis"""
        extractor = HybridExtractor()
        text = extractor.extract(file_name, file_buffer)
        
        if not text:
            return None
            
        # Extract legal entities
        entities = self.legal_processor.extract_legal_entities(text)
        doc_type = self.legal_processor.analyze_document_type(text)
        
        # Create enhanced document
        doc = Document(
            content=text,
            meta={
                "filename": file_name,
                "document_type": doc_type,
                "legal_entities": entities,
                "entity_counts": {
                    "companies": len(entities.get('companies', [])),
                    "dates": len(entities.get('dates', [])),
                    "money": len(entities.get('money', [])),
                    "people": len(entities.get('people', []))
                }
            }
        )
        return doc
    
    def setup_retriever(self, doc: Document):
        """Create simple retrieval setup using direct sentence-transformers"""
        # Split text into chunks
        words = doc.content.split()
        chunk_size = 300
        overlap = 50
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 10:  # Only add non-empty chunks
                chunks.append(chunk)
        
        self.document_chunks = chunks
        
        # Create embeddings for chunks using sentence-transformers directly
        with st.spinner("Creating document embeddings..."):
            self.chunk_embeddings = self.sentence_model.encode(chunks)
        
        return self  # Return self as "retriever"
    
    def query_document(self, retriever, query: str) -> str:
        """Query document using Gemini AI"""
        if not self.document_chunks:
            return "No document content available for querying."
        
        # Get query embedding using sentence-transformers directly
        query_embedding = self.sentence_model.encode([query])
        
        # Find most similar chunks
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top 3 most similar chunks
        top_indices = similarities.argsort()[-3:][::-1]
        context = "\n\n".join([self.document_chunks[i] for i in top_indices if similarities[i] > 0.1])
        
        if not context.strip():
            return "I don't have enough relevant information in the document to answer that question."
        
        # Gemini API call (using OpenAI compatibility)
        try:
            response = self.client.chat.completions.create(
                model="gemini-2.0-flash-exp",  # Latest Gemini model
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a legal AI assistant specialized in document analysis. "
                            "Use ONLY the provided context to answer questions about legal documents. "
                            "If information is not in the context, say you don't know. "
                            "Be precise, cite specific sections when possible, and focus on legal implications.\n\n"
                            f"CONTEXT:\n{context}"
                        )
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response with Gemini: {str(e)}"
