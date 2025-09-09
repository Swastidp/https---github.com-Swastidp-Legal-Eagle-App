# legal_bert_rag.py — FAISS HNSW + persistent index + section-aware chunking (fixed) + robust Gemini parsing + citations

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import streamlit as st
import faiss  # faiss-cpu
from sklearn.metrics.pairwise import cosine_similarity  # fallback for tiny docs
from haystack import Document
from openai import OpenAI

from utils.extractors import HybridExtractor
from utils.legal_processors import LegalNLPProcessor
from utils.model_cache import get_embedder  # unified cached SentenceTransformer


def _get_chat_model_name() -> str:
    return (
        (st.secrets.get("GEMINI_CHAT_MODEL") if hasattr(st, "secrets") else None)
        or os.getenv("GEMINI_CHAT_MODEL")
        or "gemini-2.5-flash"
    )


@dataclass
class RetrievedChunk:
    idx: int
    score: float
    start: int
    end: int
    text: str


class LegalBERTRAG:
    def __init__(self, api_key: Optional[str] = None):
        self.legal_processor = LegalNLPProcessor()

        key_from_secrets = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
        self.api_key = api_key or key_from_secrets or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY. Set it in Streamlit secrets or as an env var.")

        self.document_chunks: List[str] = []
        self.chunk_embeddings: np.ndarray = np.zeros((0, 768), dtype=np.float32)

        self._sentence_model = None
        self._faiss = None
        self._last_retrieved: List[RetrievedChunk] = []

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    @property
    def sentence_model(self):
        if self._sentence_model is None:
            with st.spinner("Loading InLegalBERT model..."):
                self._sentence_model = get_embedder()
        return self._sentence_model

    # ---------------- Healthcheck ----------------
    def healthcheck(self) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=_get_chat_model_name(),
                messages=[{"role": "user", "content": "Reply with OK"}],
                max_tokens=5,
                temperature=0.0,
            )
            choice0 = resp.choices[0]
            msg = getattr(choice0, "message", None) or (choice0.get("message") if isinstance(choice0, dict) else None)
            content = (getattr(msg, "content", None) if msg else None) or (msg.get("content") if isinstance(msg, dict) else None)
            if not content:
                content = getattr(choice0, "text", None) or (choice0.get("text") if isinstance(choice0, dict) else None)
            return (content or "").strip() or "EMPTY"
        except Exception as e:
            return f"ERR:{e}"

    # ---------------- Document processing ----------------
    def process_document(self, file_name: str, file_buffer) -> Document | None:
        text = HybridExtractor().extract(file_name, file_buffer)
        if not text:
            return None

        entities = self.legal_processor.extract_legal_entities(text)
        doc_type = self.legal_processor.analyze_document_type(text)

        return Document(
            content=text,
            meta={
                "filename": file_name,
                "document_type": doc_type,
                "legal_entities": entities,
                "entity_counts": {
                    "companies": len(entities.get("companies", [])),
                    "dates": len(entities.get("dates", [])),
                    "money": len(entities.get("money", [])),
                    "people": len(entities.get("people", [])),
                },
            },
        )

    # ---------------- FAISS persistence helpers ----------------
    def _index_path(self, doc: Document) -> str:
        fname = doc.meta.get("filename", "doc")
        digest = hashlib.sha1((doc.content or "")[:2048].encode("utf-8", "ignore")).hexdigest()[:12]
        cache_dir = os.path.join(".cache_faiss")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{Path(fname).stem}-{digest}.index")

    def try_load_index(self, doc: Document) -> bool:
        path = self._index_path(doc)
        if os.path.exists(path):
            try:
                self._faiss = faiss.read_index(path)
                return True
            except Exception:
                return False
        return False

    # ---------------- Section-aware chunking (fixed) ----------------
    def _smart_chunk(self, text: str) -> List[str]:
        # Defensive: ensure text is a usable string
        if not isinstance(text, str) or not text.strip():
            return []

        # Use non-capturing groups so split doesn't inject labels; filter strictly to strings
        pattern = r"(?im)^\s*(?:section|article|clause)\s+\d+[.:)]|\bwhereas\b"
        sec_split = re.split(pattern, text)
        parts = [p.strip() for p in sec_split if isinstance(p, str) and p is not None and p.strip()]

        if parts and len(" ".join(parts)) > 0:
            base_units = parts
        else:
            base_units = [p.strip() for p in re.split(r"\n{2,}", text) if isinstance(p, str) and p.strip()]

        chunks, buf = [], ""
        for unit in base_units:
            # Light sentence split; ensure strings only
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", unit) if isinstance(s, str) and s]
            for sent in sentences:
                if len(buf) + len(sent) + 1 <= 1400:
                    buf = f"{buf} {sent}".strip()
                else:
                    if len(buf) > 50:
                        chunks.append(buf)
                    buf = sent
            if len(buf) > 800:
                chunks.append(buf)
                buf = ""
        if len(buf) > 50:
            chunks.append(buf)
        return chunks

    # ---------------- Build FAISS HNSW ----------------
    def setup_retriever(self, doc: Document) -> "LegalBERTRAG":
        # Attempt to load a persisted index first
        loaded = self.try_load_index(doc)

        # Always rebuild chunks/embeddings to align with loaded index
        chunks = self._smart_chunk((doc.content or ""))
        self.document_chunks = chunks

        if not chunks:
            self.chunk_embeddings = np.zeros((0, 768), dtype=np.float32)
            self._faiss = None
            return self

        with st.spinner("Creating document embeddings..."):
            embs = self.sentence_model.encode(chunks).astype("float32")
            # Normalize vectors for cosine via inner product
            faiss.normalize_L2(embs)
            self.chunk_embeddings = embs

        if loaded and self._faiss is not None and self._faiss.ntotal == len(chunks):
            # Index already on disk for this content fingerprint
            return self

        d = self.chunk_embeddings.shape[1]
        M = 64
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 128
        index.hnsw.efSearch = 64
        index.add(self.chunk_embeddings)
        self._faiss = index

        # Persist to disk
        faiss.write_index(self._faiss, self._index_path(doc))
        return self

    # ---------------- Query with citations ----------------
    def query_document(self, retriever: "LegalBERTRAG", query: str) -> str:
        if not self.document_chunks:
            return "No document content available for querying."
        if getattr(self, "_faiss", None) is None or getattr(self.chunk_embeddings, "size", 0) == 0:
            # Fallback to exact cosine over RAM (small docs) to remain functional
            q = self.sentence_model.encode([query]).astype("float32")
            sims = cosine_similarity(q, self.chunk_embeddings).ravel() if self.chunk_embeddings.size else np.array([])
            if sims.size == 0:
                return "Document is too short for retrieval or index not ready."
            k = min(8, sims.size)
            top = np.argsort(sims)[-k:][::-1]
            retrieved = []
            for ii in top:
                sc = float(sims[ii])
                if sc < 0.25:
                    continue
                txt = self.document_chunks[int(ii)]
                retrieved.append(RetrievedChunk(idx=int(ii), score=sc, start=0, end=len(txt), text=txt))
        else:
            q = self.sentence_model.encode([query]).astype("float32")
            faiss.normalize_L2(q)
            top_k = 8
            D, I = self._faiss.search(q, top_k)
            sims = D.ravel()
            ids = I.ravel()
            retrieved = []
            for ii, sc in zip(ids, sims):
                if ii < 0 or sc < 0.25:
                    continue
                txt = self.document_chunks[int(ii)]
                retrieved.append(RetrievedChunk(idx=int(ii), score=float(sc), start=0, end=len(txt), text=txt))

        if not retrieved:
            return "I don't have enough relevant information in the document to answer that question."

        # Build bounded context with chunk tags
        max_chars = 12000
        context_parts, used = [], 0
        for rc in retrieved:
            piece = f"[Chunk {rc.idx}] {rc.text}"
            if used + len(piece) + 2 > max_chars:
                break
            context_parts.append(piece)
            used += len(piece) + 2
        context = "\n\n".join(context_parts).strip()
        if not context:
            return "I don't have enough relevant information in the document to answer that question."

        try:
            response = self.client.chat.completions.create(
                model=_get_chat_model_name(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a legal AI assistant specialized in document analysis. "
                            "Use ONLY the provided context to answer questions about legal documents. "
                            "If information is not in the context, say you don't know. "
                            "After the answer, append a 'Citations:' section listing [Chunk id] references used."
                            "\n\nCONTEXT:\n" + context
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                max_tokens=800,
                temperature=0.1,
            )

            # Robust parse
            choice0 = response.choices[0]
            msg = getattr(choice0, "message", None) or (choice0.get("message") if isinstance(choice0, dict) else None)
            content = (getattr(msg, "content", None) if msg else None) or (msg.get("content") if isinstance(msg, dict) else None)
            if not content:
                content = getattr(choice0, "text", None) or (choice0.get("text") if isinstance(choice0, dict) else None)
            finish_reason = getattr(choice0, "finish_reason", None) or (choice0.get("finish_reason") if isinstance(choice0, dict) else None)
            if not content:
                return f"Empty model content. finish_reason={finish_reason!r}. Raw preview: {str(response)[:500]}"

            # Attach machine-readable citations for UI
            cite_lines = [f"- [Chunk {rc.idx}] chars {rc.start}-{rc.end} (score={rc.score:.2f})" for rc in retrieved[:5]]
            content += "\n\nCitations:\n" + "\n".join(cite_lines)
            self._last_retrieved = retrieved
            return content

        except Exception as e:
            return f"Error generating response with Gemini: {str(e)}"
