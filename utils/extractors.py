# utils/extractors.py

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import streamlit as st
import pytesseract
from PIL import Image
import pdf2image
import docx
import PyPDF2
from striprtf.striprtf import rtf_to_text

# Optional: legacy .doc extraction via textract (antiword)
try:
    import textract  # requires system antiword for .doc
    _HAS_TEXTRACT = True
except Exception:
    _HAS_TEXTRACT = False


def _normalize_text_blocks(lines: list[str]) -> str:
    """
    Light cleanup:
    - strip leading/trailing whitespace on lines
    - remove repeated blank lines beyond two
    - join with single newlines within paragraphs and double newlines between
    """
    cleaned: list[str] = []
    blank = 0
    for ln in lines:
        s = (ln or "").rstrip()
        if not s.strip():
            blank += 1
            if blank <= 2:
                cleaned.append("")
            continue
        blank = 0
        cleaned.append(s)
    # Collapse runs of >2 blanks to exactly 2 newlines
    text = "\n".join(cleaned)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


class HybridExtractor:
    """Shared document extraction logic."""

    def extract(self, file_name: str, file_buf: io.BytesIO) -> str | None:
        ext = Path(file_name).suffix.lower()
        match ext:
            case ".pdf":
                return self._pdf(file_buf)
            case ".docx":
                return self._docx(file_buf)
            case ".doc":
                return self._doc_legacy(file_buf)
            case ".rtf":
                return self._rtf(file_buf)
            case ".txt":
                return self._txt(file_buf)
            case _:
                st.error(f"Unsupported file type: {ext}")
                return None

    # ---------------- PDF (native text -> OCR fallback) ----------------
    def _pdf(self, buf: io.BytesIO) -> str:
        """
        Try PyPDF2 native extraction per-page, insert page delimiters to help downstream
        paragraph/page-aware chunking. If no text, fallback to OCR via Tesseract.
        """
        # Attempt native text
        try:
            reader = PyPDF2.PdfReader(buf)
            page_texts: list[str] = []
            for i, page in enumerate(reader.pages):
                t = page.extract_text() or ""
                t = t.replace("\r", "\n")
                # Normalize intra-page whitespace minimally
                t = _normalize_text_blocks(t.split("\n"))
                page_texts.append(t)
            if any(pt.strip() for pt in page_texts):
                # Insert explicit page separators
                pieces = []
                for idx, pt in enumerate(page_texts, start=1):
                    if idx > 1:
                        pieces.append(f"\n\n=== PAGE {idx} ===\n\n")
                    pieces.append(pt)
                return "".join(pieces).strip()
        except Exception as e:
            st.warning(f"PDF text extraction failed, falling back to OCR: {e}")

        # OCR fallback for scanned PDFs
        st.info("PDF appears scanned; running OCR …")
        buf.seek(0)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(buf.read())
                tmp.flush()
                tmp_path = tmp.name
            pages = pdf2image.convert_from_path(tmp_path, dpi=300)
            result: list[str] = []
            for i, img in enumerate(pages, 1):
                st.write(f"🖼️ OCR page {i}/{len(pages)}")
                result.append(pytesseract.image_to_string(img))
            # Insert the same page separators to align with native path
            cleaned: list[str] = []
            for idx, page_str in enumerate(result, start=1):
                if idx > 1:
                    cleaned.append(f"\n\n=== PAGE {idx} ===\n\n")
                cleaned.append(_normalize_text_blocks((page_str or "").split("\n")))
            return "".join(cleaned).strip()
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ---------------- DOCX ----------------
    def _docx(self, buf: io.BytesIO) -> str:
        doc = docx.Document(buf)
        paras = [p.text for p in doc.paragraphs]
        text = _normalize_text_blocks(paras)
        return text

    # ---------------- Legacy .DOC (textract/antiword -> OCR last) ----------------
    def _doc_legacy(self, buf: io.BytesIO) -> str:
        # Save to a temp .doc file for textract/antiword
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
            tmp.write(buf.read())
            tmp.flush()
            path = tmp.name
        try:
            if _HAS_TEXTRACT:
                try:
                    raw = textract.process(path, encoding="utf-8")
                    text = raw.decode("utf-8", "ignore")
                    if text.strip():
                        return _normalize_text_blocks(text.split("\n"))
                    st.warning("Empty text via antiword; attempting OCR fallback.")
                except Exception as e:
                    st.warning(f".doc extraction via textract failed: {e}. Attempting OCR fallback.")
            else:
                st.warning("textract/antiword not available; attempting OCR fallback for .doc.")

            # OCR fallback (best-effort; not all .doc formats are image-renderable)
            try:
                img = Image.open(path)
                text = pytesseract.image_to_string(img)
                return _normalize_text_blocks(text.split("\n"))
            except Exception:
                st.error("Unable to OCR .doc file. Please convert to DOCX or PDF.")
                return ""
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    # ---------------- RTF ----------------
    def _rtf(self, buf: io.BytesIO) -> str:
        # Preserve unknown bytes and normalize output paragraphs
        data = buf.read()
        for enc in ("utf-8", "latin-1"):
            try:
                txt = rtf_to_text(data.decode(enc))
                return _normalize_text_blocks(txt.split("\n"))
            except Exception:
                continue
        st.error("Failed to decode RTF content.")
        return ""

    # ---------------- TXT ----------------
    def _txt(self, buf: io.BytesIO) -> str:
        data = buf.read()
        for enc in ("utf-8", "latin-1"):
            try:
                txt = data.decode(enc)
                # Normalize to stable paragraphs
                return _normalize_text_blocks(txt.splitlines())
            except Exception:
                continue
        st.error("Failed to decode TXT content.")
        return ""
