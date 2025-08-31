# utils/extractors.py
import io, os, tempfile
from pathlib import Path
import streamlit as st
import pytesseract
from PIL import Image
import pdf2image, docx, PyPDF2
from striprtf.striprtf import rtf_to_text

class HybridExtractor:
    """Shared document extraction logic"""
    
    def extract(self, file_name: str, file_buf: io.BytesIO) -> str | None:
        ext = Path(file_name).suffix.lower()
        match ext:
            case ".pdf": return self._pdf(file_buf)
            case ".docx": return self._docx(file_buf)
            case ".doc": return self._doc_legacy(file_buf)
            case ".rtf": return self._rtf(file_buf)
            case ".txt": return self._txt(file_buf)
            case _:
                st.error(f"Unsupported file type: {ext}")
                return None
    
    def _pdf(self, buf: io.BytesIO) -> str:
        reader = PyPDF2.PdfReader(buf)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            return text
        
        st.info("PDF is scanned; running OCR …")
        buf.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(buf.read())
            tmp.flush()
            pages = pdf2image.convert_from_path(tmp.name, dpi=300)
            result = []
            for i, img in enumerate(pages, 1):
                st.write(f"🖼️ OCR page {i}/{len(pages)}")
                result.append(pytesseract.image_to_string(img))
            os.unlink(tmp.name)
            return "\n".join(result)
    
    def _docx(self, buf: io.BytesIO) -> str:
        doc = docx.Document(buf)
        return "\n".join(p.text for p in doc.paragraphs)
    
    def _doc_legacy(self, buf: io.BytesIO) -> str:
        st.warning("Legacy .doc detected - OCR fallback only.")
        return pytesseract.image_to_string(Image.open(buf))
    
    def _rtf(self, buf: io.BytesIO) -> str:
        return rtf_to_text(buf.read().decode("utf-8", errors="ignore"))
    
    def _txt(self, buf: io.BytesIO) -> str:
        return buf.read().decode("utf-8", errors="ignore")
