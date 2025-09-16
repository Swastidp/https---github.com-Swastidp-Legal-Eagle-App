# legal_bert_rag.py — Gemini primary + OpenAI-compat fallback (hardened)
# FAISS HNSW (persistent) + section-aware chunking + robust compat parsing + retries
# Healthcheck/client guards + citations + snippet fallback
# Chunk metadata: page and paragraph numbers in citations and Sources
# Resilient retrieval: fallback to document paragraphs if embeddings/index are empty
# Page mapping: detect "=== PAGE n ===" delimiters and map chunk start offsets to page numbers

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Any, Tuple, Dict

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
    page: int = -1
    para: int = -1


class LegalBERTRAG:
    def __init__(self, api_key: Optional[str] = None):
        self.legal_processor = LegalNLPProcessor()
        key_from_secrets = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
        self.api_key = api_key or key_from_secrets or os.getenv("GEMINI_API_KEY")

        # RAG state
        self.document_chunks: List[str] = []
        self.chunk_meta: List[Dict[str, int]] = []  # {"page": int, "para": int}
        self.chunk_embeddings: np.ndarray = np.zeros((0, 768), dtype=np.float32)
        self._sentence_model = None
        self._faiss = None
        self._last_retrieved: List[RetrievedChunk] = []

        # Keep a raw head for last-ditch fallback
        self.raw_doc_text: str = ""

        # Track page start offsets (absolute char positions) for mapping chunk starts to pages
        self._page_starts: List[int] = [0]

        # Clients and errors
        self.gclient = None
        self.client = None
        self._client_error = None

        # Retry/timeout config
        self._timeout = float(os.getenv("GENAI_TIMEOUT", "30"))
        self._retries = int(os.getenv("GENAI_RETRIES", "2"))

        # Native client first
        if not self.api_key:
            self._client_error = "Missing GEMINI_API_KEY in secrets/env."
        else:
            try:
                self.gclient = genai.Client(api_key=self.api_key, http_options=HttpOptions(api_version="v1"))
            except Exception as e:
                self._client_error = f"Native client init failed: {e}"

        # OpenAI-compat client (fallback)
        if OpenAI is not None and self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    timeout=self._timeout,
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

    # ---------- Utilities: fully permissive compat parser ----------
    @staticmethod
    def _parse_compat_content(resp: Any) -> str | None:
        try:
            choices = getattr(resp, "choices", None)
            if choices is None and isinstance(resp, dict):
                choices = resp.get("choices")
            if not choices or len(choices) == 0:
                return None
            c0 = choices[0]

            msg = getattr(c0, "message", None)
            if msg is None and isinstance(c0, dict):
                msg = c0.get("message")

            content = None
            if msg is not None:
                content = getattr(msg, "content", None)
                if content is None and isinstance(msg, dict):
                    content = msg.get("content")

            if isinstance(content, str) and content.strip():
                return content.strip()

            if isinstance(content, list):
                parts = [p for p in content if isinstance(p, str)]
                if parts:
                    return "".join(parts).strip()

            txt = getattr(c0, "text", None)
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
            if isinstance(c0, dict):
                txt = c0.get("text")
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()

            try:
                to_dict = getattr(msg, "to_dict", None)
                if callable(to_dict):
                    d = to_dict()
                    if isinstance(d, dict):
                        c = d.get("content")
                        if isinstance(c, str) and c.strip():
                            return c.strip()
                        if isinstance(c, list):
                            parts = [p for p in c if isinstance(p, str)]
                            if parts:
                                return "".join(parts).strip()
            except Exception:
                pass

            return None
        except Exception:
            return None

    # ---------------- Page start offsets from extractor delimiters ----------------
    def _compute_page_starts(self, text: str) -> List[int]:
        """
        Detect "=== PAGE n ===" delimiters (inserted by extractor) and return absolute
        char offsets where each page's content starts. Always include offset 0 for page 1.
        """
        starts = [0]
        if not isinstance(text, str) or not text:
            return starts
        # Match delimiter lines like "\n\n=== PAGE 3 ===\n\n" robustly
        for m in re.finditer(r"\n\s*===\s*PAGE\s+(\d+)\s*===\s*\n", text, flags=re.IGNORECASE):
            # Content starts immediately after the delimiter
            pos = m.end()
            starts.append(pos)
        starts = sorted(set(starts))
        return starts

    # ---------------- Healthcheck ----------------
    def healthcheck(self) -> str:
        model = _get_chat_model_name()
        if self.gclient:
            try:
                r = self.gclient.models.generate_content(model=model, contents="Reply with OK")
                txt = (getattr(r, "text", "") or "").strip().upper()
                return "OK" if "OK" in txt else (txt or "EMPTY")
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
                content = (self._parse_compat_content(resp) or "").strip().upper()
                return "OK" if "OK" in content else (content or "EMPTY")
            except Exception as e:
                return f"ERR:{e}"
        return f"ERR:{self._client_error or 'Clients not initialized'}"

    # ---------------- Document processing ----------------
    def process_document(self, file_name: str, file_buffer) -> Document | None:
        text = HybridExtractor().extract(file_name, file_buffer)
        if not text:
            return None
        # Keep raw text head and page starts for fallbacks/mapping
        self.raw_doc_text = text[:4000] if isinstance(text, str) else ""
        self._page_starts = self._compute_page_starts(text)

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

    # ---------------- Section-aware chunking with paragraph + page mapping ----------------
    def _smart_chunk(self, text: str) -> Tuple[List[str], List[Dict[str, int]]]:
        """
        Returns chunks and metadata: [{"page": int, "para": int}, ...]
        Page is derived by mapping the chunk's absolute start char offset
        to the latest page start offset (1-based page number).
        Paragraph is the base unit index.
        """
        if not isinstance(text, str) or not text.strip():
            return [], []

        page_starts = getattr(self, "_page_starts", [0])

        def offset_to_page(off: int) -> int:
            if not page_starts:
                return 1
            lo, hi = 0, len(page_starts) - 1
            ans = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if page_starts[mid] <= off:
                    ans = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            return ans + 1  # 1-based

        # Section/paragraph segmentation while tracking absolute spans
        pattern = r"(?im)^\s*(?:section|article|clause)\s+\d+[.:)]|\bwhereas\b"
        parts: List[str] = []
        spans: List[Tuple[int, int]] = []  # absolute (start, end)

        last_end = 0
        for m in re.finditer(pattern, text):
            seg = text[last_end:m.start()]
            if seg.strip():
                start_off = last_end
                end_off = m.start()
                parts.append(seg.strip())
                spans.append((start_off, end_off))
            last_end = m.start()
        tail = text[last_end:]
        if tail.strip():
            parts.append(tail.strip())
            spans.append((last_end, len(text)))

        if not parts:
            # Fallback: double-newline paragraphs with approximate spans
            parts = [p.strip() for p in re.split(r"\n{2,}", text) if isinstance(p, str) and p.strip()]
            spans = []
            walk = 0
            for p in parts:
                idx = text.find(p, walk)
                if idx < 0:
                    idx = walk
                spans.append((idx, idx + len(p)))
                walk = idx + len(p)

        chunks: List[str] = []
        metas: List[Dict[str, int]] = []
        buf, buf_start_off = "", None

        for para_idx, (unit, (u_start, u_end)) in enumerate(zip(parts, spans)):
            unit_cursor = u_start
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", unit) if isinstance(s, str) and s]
            for sent in sentences:
                s_idx = text.find(sent, unit_cursor)
                if s_idx < 0:
                    s_idx = unit_cursor
                if not buf:
                    buf_start_off = s_idx
                if len(buf) + len(sent) + 1 <= 1400:
                    buf = f"{buf} {sent}".strip()
                else:
                    if len(buf) > 50:
                        page = offset_to_page(buf_start_off or u_start)
                        chunks.append(buf)
                        metas.append({"page": page, "para": para_idx})
                    buf = sent
                    buf_start_off = s_idx
                if len(buf) > 800:
                    page = offset_to_page(buf_start_off or s_idx)
                    chunks.append(buf)
                    metas.append({"page": page, "para": para_idx})
                    buf = ""
                    buf_start_off = None
                unit_cursor = s_idx + len(sent)

        if len(buf) > 50:
            page = offset_to_page(buf_start_off or (spans[-1][0] if spans else 0))
            chunks.append(buf)
            metas.append({"page": page, "para": len(parts) - 1 if parts else 0})

        return chunks, metas

    # ---------------- Build FAISS HNSW ----------------
    def setup_retriever(self, doc: Document) -> "LegalBERTRAG":
        loaded = self.try_load_index(doc)
        chunks, metas = self._smart_chunk((doc.content or ""))
        self.document_chunks = chunks
        self.chunk_meta = metas

        # If nothing to index, clear state and return early
        if not chunks:
            self.chunk_embeddings = np.zeros((0, 768), dtype=np.float32)
            self._faiss = None
            return self

        # Keep meta aligned with chunks length
        if len(self.chunk_meta) != len(self.document_chunks):
            if len(self.chunk_meta) < len(self.document_chunks):
                pad_count = len(self.document_chunks) - len(self.chunk_meta)
                self.chunk_meta.extend([{"page": -1, "para": -1}] * pad_count)
            else:
                self.chunk_meta = self.chunk_meta[:len(self.document_chunks)]

        with st.spinner("Creating document embeddings..."):
            embs = self.sentence_model.encode(chunks).astype("float32")
            embs = np.asarray(embs, dtype="float32")
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)
            if embs.ndim != 2 or embs.shape[0] != len(chunks) or embs.shape[1] <= 0:
                self.chunk_embeddings = np.zeros((0, 768), dtype="float32")
                self._faiss = None
                return self

            faiss.normalize_L2(embs)
            self.chunk_embeddings = embs

        # Reuse on-disk index only if it matches the chunk count
        if loaded and self._faiss is not None and self._faiss.ntotal == len(chunks):
            return self

        # Correct embedding dimensionality (vector dim is axis 1)
        d = int(self.chunk_embeddings.shape[1])
        if d <= 0:
            self._faiss = None
            return self

        M = 64
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 128
        index.hnsw.efSearch = 64
        index.add(self.chunk_embeddings)
        self._faiss = index
        try:
            faiss.write_index(self._faiss, self._index_path(doc))
        except Exception:
            pass
        return self

    # ---------------- Internal: compat chat with retries ----------------
    def _compat_chat(self, model: str, system_prompt: str, user_prompt: str) -> str | None:
        if not self.client:
            return None
        last_err: Any = None
        for _ in range(self._retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=800,
                    temperature=0.1,
                )
                content = self._parse_compat_content(resp)
                if isinstance(content, str) and content.strip():
                    return content.strip()

                try:
                    c0 = resp.choices[0]
                    if hasattr(c0, "message") and getattr(c0.message, "content", None):
                        alt = c0.message.content
                        if isinstance(alt, str) and alt.strip():
                            return alt.strip()
                        if isinstance(alt, list):
                            parts = [p for p in alt if isinstance(p, str)]
                            if parts:
                                return "".join(parts).strip()
                    if hasattr(c0, "text") and isinstance(c0.text, str) and c0.text.strip():
                        return c0.text.strip()
                except Exception:
                    pass
                last_err = ValueError("Empty compat content")
            except Exception as e:
                last_err = e
        return None  # triggers snippet fallback

    # ---------------- Query with citations (resilient) ----------------
    def query_document(self, retriever: "LegalBERTRAG", query: str) -> str:
        if not self.document_chunks:
            # Last-ditch: try to chunk raw head for some context
            if self.raw_doc_text:
                fallback_chunk = self.raw_doc_text[:1400]
                self.document_chunks = [fallback_chunk]
                self.chunk_meta = [{"page": 1, "para": 0}]
            else:
                return "I don't have enough relevant information in the document to answer that question."

        # Retrieval
        retrieved: List[RetrievedChunk] = []

        if getattr(self, "_faiss", None) is None or getattr(self.chunk_embeddings, "size", 0) == 0:
            # No FAISS or no embeddings: attempt cosine with whatever is loaded
            if self.chunk_embeddings.size > 0:
                q = self.sentence_model.encode([query]).astype("float32")
                sims = cosine_similarity(q, self.chunk_embeddings).ravel()
            else:
                sims = np.array([])

            # Fallback path if similarities are empty -> use first paragraphs
            if sims.size == 0:
                max_fallback = min(3, len(self.document_chunks))
                for i in range(max_fallback):
                    txt = self.document_chunks[i]
                    meta = self.chunk_meta[i] if 0 <= i < len(self.chunk_meta) else {"page": 1, "para": i}
                    retrieved.append(RetrievedChunk(
                        idx=i, score=0.0, start=0, end=len(txt), text=txt,
                        page=int(meta.get("page", 1)), para=int(meta.get("para", i))
                    ))
            else:
                k = min(8, sims.size)
                top = np.argsort(sims)[-k:][::-1]
                for ii in top:
                    sc = float(sims[ii])
                    if sc < 0.15:
                        continue
                    txt = self.document_chunks[int(ii)]
                    meta = self.chunk_meta[int(ii)] if 0 <= int(ii) < len(self.chunk_meta) else {"page": 1, "para": -1}
                    retrieved.append(RetrievedChunk(
                        idx=int(ii), score=sc, start=0, end=len(txt), text=txt,
                        page=int(meta.get("page", 1)), para=int(meta.get("para", -1))
                    ))
        else:
            q = self.sentence_model.encode([query]).astype("float32")
            faiss.normalize_L2(q)
            top_k = 8
            D, I = self._faiss.search(q, top_k)
            sims = D.ravel()
            ids = I.ravel()
            for ii, sc in zip(ids, sims):
                if ii < 0 or sc < 0.15:
                    continue
                txt = self.document_chunks[int(ii)]
                meta = self.chunk_meta[int(ii)] if 0 <= int(ii) < len(self.chunk_meta) else {"page": 1, "para": -1}
                retrieved.append(RetrievedChunk(
                    idx=int(ii), score=float(sc), start=0, end=len(txt), text=txt,
                    page=int(meta.get("page", 1)), para=int(meta.get("para", -1))
                ))

        if not retrieved:
            # Absolute final fallback: use raw_doc_text head
            if self.raw_doc_text:
                txt = self.raw_doc_text[:1400]
                retrieved = [RetrievedChunk(idx=0, score=0.0, start=0, end=len(txt), text=txt, page=1, para=0)]
            else:
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
            # Final guard
            first_text = retrieved[0].text if retrieved else (self.raw_doc_text[:1400] if self.raw_doc_text else "")
            if not first_text:
                return "I don't have enough relevant information in the document to answer that question."
            context = f"[Chunk 0] {first_text}"

        model = _get_chat_model_name()

        # Native Gemini first — simple string prompt
        if self.gclient:
            try:
                prompt = (
                    "You are a legal AI assistant specialized in document analysis. "
                    "Use ONLY the provided context to answer questions about legal documents. "
                    "If information is not in the context, say you don't know. "
                    "After the answer, append a 'Citations:' section listing [Chunk id] references.\n\n"
                    f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
                )
                r = self.gclient.models.generate_content(model=model, contents=prompt)
                content = (getattr(r, "text", "") or "").strip()
                if content:
                    cite_lines = [
                        f"- [Chunk {rc.idx}] page {rc.page if rc.page>=0 else '?'} • para {rc.para if rc.para>=0 else '?'} "
                        f"• chars {rc.start}-{rc.end} (score={rc.score:.2f})"
                        for rc in retrieved[:5]
                    ]
                    content += "\n\nCitations:\n" + "\n".join(cite_lines)
                    self._last_retrieved = retrieved
                    return content
            except Exception:
                pass  # fall through

        # OpenAI-compat fallback
        if self.client:
            system_prompt = (
                "You are a legal AI assistant specialized in document analysis. "
                "Use ONLY the provided context to answer questions about legal documents. "
                "If information is not in the context, say you don't know. "
                "After the answer, append a 'Citations:' section listing [Chunk id] references used.\n\n"
                f"CONTEXT:\n{context}"
            )
            content = self._compat_chat(model=model, system_prompt=system_prompt, user_prompt=query)
            if content:
                cite_lines = [
                    f"- [Chunk {rc.idx}] page {rc.page if rc.page>=0 else '?'} • para {rc.para if rc.para>=0 else '?'} "
                    f"• chars {rc.start}-{rc.end} (score={rc.score:.2f})"
                    for rc in retrieved[:5]
                ]
                content = content.strip() + "\n\nCitations:\n" + "\n".join(cite_lines)
                self._last_retrieved = retrieved
                return content

        # Snippet fallback — guaranteed non-error output
        self._last_retrieved = retrieved
        bullets = []
        for rc in retrieved[:5]:
            sents = re.split(r"(?<=[.!?])\s+", rc.text)
            snippet = " ".join(sents[:2]).strip()
            bullets.append(f"- [Chunk {rc.idx}] {snippet[:400]}")
        return (
            "Gemini response unavailable; returning extractive snippet summary from top sources.\n\n"
            "Summary:\n" + "\n".join(bullets) + "\n\n" +
            "Citations:\n" + "\n".join([
                f"- [Chunk {rc.idx}] page {rc.page if rc.page>=0 else '?'} • para {rc.para if rc.para>=0 else '?'} "
                f"• chars {rc.start}-{rc.end} (score={rc.score:.2f})"
                for rc in retrieved[:5]
            ])
        )
