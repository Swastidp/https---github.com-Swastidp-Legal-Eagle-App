# legal_bert_rag.py — Native Gemini primary + OpenAI-compat fallback
# FAISS HNSW (persistent) + section-aware chunking + robust compat parsing fix
# Healthcheck/client guards + citations + snippet fallback

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Any

import numpy as np
import streamlit as st
import faiss  # faiss-cpu
from sklearn.metrics.pairwise import cosine_similarity
from haystack import Document

from utils.extractors import HybridExtractor
from utils.legal_processors import LegalNLPProcessor
from utils.model_cache import get_embedder

# Native Gemini SDK (preferred on Cloud)
from google import genai
from google.genai.types import HttpOptions

# OpenAI-compat (fallback)
try:
    from openai import OpenAI  # pinned >= 1.101.0 recommended
except Exception:
    OpenAI = None


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

        # RAG state
        self.document_chunks: List[str] = []
        self.chunk_embeddings: np.ndarray = np.zeros((0, 768), dtype=np.float32)
        self._sentence_model = None
        self._faiss = None
        self._last_retrieved: List[RetrievedChunk] = []

        # Clients and errors
        self.gclient = None
        self.client = None
        self._client_error = None

        # Native client first
        if not self.api_key:
            self._client_error = "Missing GEMINI_API_KEY in secrets/env."
        else:
            try:
                self.gclient = genai.Client(api_key=self.api_key, http_options=HttpOptions(api_version="v1"))
            except Exception as e:
                self._client_error = f"Native client init failed: {e}"

            if OpenAI is not None:
                try:
                    self.client = OpenAI(
                        api_key=self.api_key,
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    )
                except Exception as e:
                    if self._client_error:
                        self._client_error += f" | Compat init failed: {e}"
                    else:
                        self._client_error = f"Compat init failed: {e}"

    @property
    def sentence_model(self):
        if self._sentence_model is None:
            with st.spinner("Loading InLegalBERT model..."):
                self._sentence_model = get_embedder()
        return self._sentence_model

    # ---------- Utilities: strict compat parser ----------
    @staticmethod
    def _parse_compat_content(resp: Any) -> str | None:
        """
        Strict, ordered extraction for OpenAI-compat responses.
        Prefers choices[0].message.content, then dict fallbacks, then .text.
        Returns a stripped string or None if not present.
        """
        try:
            choices = getattr(resp, "choices", None)
            if choices is None and isinstance(resp, dict):
                choices = resp.get("choices")
            if not choices:
                return None

            c0 = choices[0]

            # 1) Attribute path: choices[0].message.content
            if hasattr(c0, "message") and getattr(c0, "message") is not None:
                m = getattr(c0, "message")
                content = getattr(m, "content", None)
                if isinstance(content, str) and content.strip():
                    return content.strip()
                # Some SDKs return list parts; join strings
                if isinstance(content, list):
                    parts = [p for p in content if isinstance(p, str)]
                    if parts:
                        return "".join(parts).strip()

            # 2) Dict-style path
            if isinstance(c0, dict):
                msg = c0.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                    if isinstance(content, list):
                        parts = [p for p in content if isinstance(p, str)]
                        if parts:
                            return "".join(parts).strip()

            # 3) Non-chat fallback: choices[0].text
            txt = getattr(c0, "text", None)
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
            if isinstance(c0, dict):
                txt = c0.get("text")
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()

            return None
        except Exception:
            return None

    # ---------------- Healthcheck ----------------
    def healthcheck(self) -> str:
        model = _get_chat_model_name()
        if self.gclient:
            try:
                r = self.gclient.models.generate_content(model=model, contents="Reply with OK")
                return (getattr(r, "text", "") or "").strip() or "EMPTY"
            except Exception as e:
                return f"ERR:{e}"
        if self.client:
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Reply with OK"}],
                    max_tokens=5,
                    temperature=0.0,
                )
                content = self._parse_compat_content(resp)
                return content or "EMPTY"
            except Exception as e:
                return f"ERR:{e}"
        return f"ERR:{self._client_error or 'Clients not initialized'}"

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
        if not isinstance(text, str) or not text.strip():
            return []
        pattern = r"(?im)^\s*(?:section|article|clause)\s+\d+[.:)]|\bwhereas\b"
        sec_split = re.split(pattern, text)
        parts = [p.strip() for p in sec_split if isinstance(p, str) and p is not None and p.strip()]

        if parts and len(" ".join(parts)) > 0:
            base_units = parts
        else:
            base_units = [p.strip() for p in re.split(r"\n{2,}", text) if isinstance(p, str) and p.strip()]

        chunks, buf = [], ""
        for unit in base_units:
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
        loaded = self.try_load_index(doc)

        chunks = self._smart_chunk((doc.content or ""))
        self.document_chunks = chunks

        if not chunks:
            self.chunk_embeddings = np.zeros((0, 768), dtype=np.float32)
            self._faiss = None
            return self

        with st.spinner("Creating document embeddings..."):
            embs = self.sentence_model.encode(chunks).astype("float32")
            faiss.normalize_L2(embs)  # cosine via inner product
            self.chunk_embeddings = embs

        if loaded and self._faiss is not None and self._faiss.ntotal == len(chunks):
            return self

        d = self.chunk_embeddings.shape[1]
        M = 64
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 128
        index.hnsw.efSearch = 64
        index.add(self.chunk_embeddings)
        self._faiss = index

        faiss.write_index(self._faiss, self._index_path(doc))
        return self

    # ---------------- Query with citations ----------------
    def query_document(self, retriever: "LegalBERTRAG", query: str) -> str:
        if not self.document_chunks:
            return "No document content available for querying."

        # Retrieval
        if getattr(self, "_faiss", None) is None or getattr(self.chunk_embeddings, "size", 0) == 0:
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

        # Build bounded context
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

        model = _get_chat_model_name()

        # Native Gemini first
        if self.gclient:
            try:
                r = self.gclient.models.generate_content(
                    model=model,
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                (
                                    "You are a legal AI assistant specialized in document analysis. "
                                    "Use ONLY the provided context to answer questions about legal documents. "
                                    "If information is not in the context, say you don't know. "
                                    "After the answer, append a 'Citations:' section listing [Chunk id] references.\n\n"
                                    f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
                                )
                            ],
                        }
                    ],
                )
                content = (getattr(r, "text", "") or "").strip()
                if content:
                    cite_lines = [f"- [Chunk {rc.idx}] chars {rc.start}-{rc.end} (score={rc.score:.2f})" for rc in retrieved[:5]]
                    content += "\n\nCitations:\n" + "\n".join(cite_lines)
                    self._last_retrieved = retrieved
                    return content
            except Exception:
                pass  # fall through

        # OpenAI-compat fallback (with fixed parser)
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=model,
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
                content = self._parse_compat_content(response)
                if content and content.strip():
                    cite_lines = [f"- [Chunk {rc.idx}] chars {rc.start}-{rc.end} (score={rc.score:.2f})" for rc in retrieved[:5]]
                    content = content.strip() + "\n\nCitations:\n" + "\n".join(cite_lines)
                    self._last_retrieved = retrieved
                    return content
                # If parser failed, show diagnostic preview
                return f"Compat parse error: raw preview: {str(response)[:500]}"
            except Exception as e:
                # fall through to snippet mode
                pass

        # Snippet fallback
        self._last_retrieved = retrieved
        bullets = []
        for rc in retrieved[:5]:
            sents = re.split(r"(?<=[.!?])\s+", rc.text)
            snippet = " ".join(sents[:2]).strip()
            bullets.append(f"- [Chunk {rc.idx}] {snippet[:400]}")
        return (
            "Gemini response unavailable; returning extractive snippet summary from top sources.\n\n"
            "Summary:\n" + "\n".join(bullets) + "\n\n" +
            "Citations:\n" + "\n".join([f"- [Chunk {rc.idx}] chars {rc.start}-{rc.end} (score={rc.score:.2f})" for rc in retrieved[:5]])
        )
