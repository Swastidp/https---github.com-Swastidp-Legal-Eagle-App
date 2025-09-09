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
        try:
            reader = PyPDF2.PdfReader(buf)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                return text
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
            result = []
            for i, img in enumerate(pages, 1):
                st.write(f"🖼️ OCR page {i}/{len(pages)}")
                result.append(pytesseract.image_to_string(img))
            return "\n".join(result)
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ---------------- DOCX ----------------
    def _docx(self, buf: io.BytesIO) -> str:
        doc = docx.Document(buf)
        return "\n".join(p.text for p in doc.paragraphs)

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
                        return text
                    st.warning("Empty text via antiword; attempting OCR fallback.")
                except Exception as e:
                    st.warning(f".doc extraction via textract failed: {e}. Attempting OCR fallback.")
            else:
                st.warning("textract/antiword not available; attempting OCR fallback for .doc.")

            # OCR fallback only as last resort (works only if file is an image rendering)
            try:
                # Convert first page to image if possible; many .doc files are not images.
                img = Image.open(path)
                return pytesseract.image_to_string(img)
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
        # Avoid silently dropping content; preserve unknown bytes
        data = buf.read()
        try:
            return rtf_to_text(data.decode("utf-8"))
        except Exception:
            # Retry with latin-1 as a fallback
            try:
                return rtf_to_text(data.decode("latin-1"))
            except Exception:
                st.error("Failed to decode RTF content.")
                return ""

    # ---------------- TXT ----------------
    def _txt(self, buf: io.BytesIO) -> str:
        data = buf.read()
        try:
            return data.decode("utf-8")
        except Exception:
            try:
                return data.decode("latin-1")
            except Exception:
                st.error("Failed to decode TXT content.")
                return ""
