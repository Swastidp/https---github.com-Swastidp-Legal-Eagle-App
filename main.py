# frontend.py  ── Streamlit + Hybrid OCR/Text extractor
import io, os, tempfile
from pathlib import Path

import streamlit as st
import pytesseract
from PIL import Image
import pdf2image
import docx
import PyPDF2
from striprtf.striprtf import rtf_to_text

# ────────────────────────────  Page config  ─────────────────────────────
st.set_page_config(
    page_title="Legal Document Scanner • Streamlit + Tesseract",
    page_icon="⚖️",
    layout="wide",
)

# ────────────────────────────  Helper class  ────────────────────────────
class HybridExtractor:
    """Native text extraction first; fallback to Tesseract OCR."""

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

    # -------------------- individual format handlers --------------------
    def _pdf(self, buf: io.BytesIO) -> str:
        """PDF → try PyPDF2, else OCR per-page."""
        reader = PyPDF2.PdfReader(buf)
        text = "".join(page.extract_text() or "" for page in reader.pages)

        if text.strip():
            return text

        # Fallback: OCR each rendered page
        st.info("PDF is scanned; running OCR …")
        buf.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(buf.read())
            tmp.flush()

            pages = pdf2image.convert_from_path(tmp.name, dpi=300)
            result = []
            for i, img in enumerate(pages, 1):
                st.write(f"🖼️  OCR page {i}/{len(pages)}")
                result.append(pytesseract.image_to_string(img))
            os.unlink(tmp.name)

        return "\n".join(result)

    def _docx(self, buf: io.BytesIO) -> str:
        doc = docx.Document(buf)
        return "\n".join(p.text for p in doc.paragraphs)

    def _doc_legacy(self, buf: io.BytesIO) -> str:
        st.warning("Legacy .doc detected - OCR fallback only.")
        return self._image(Image.open(buf))

    def _rtf(self, buf: io.BytesIO) -> str:
        return rtf_to_text(buf.read().decode("utf-8", errors="ignore"))

    def _txt(self, buf: io.BytesIO) -> str:
        return buf.read().decode("utf-8", errors="ignore")

    def _image(self, img: Image.Image) -> str:
        return pytesseract.image_to_string(img)

# ────────────────────────────  UI  ──────────────────────────────────────
st.title("⚖️  Legal Document Scanner • Streamlit + Tesseract")
st.markdown(
    "Upload a **PDF, DOCX, RTF or TXT** file. The app tries native text "
    "extraction first, then falls back to OCR with Tesseract if needed."
)

uploaded = st.file_uploader(
    "Upload legal document",
    type=["pdf", "docx", "doc", "rtf", "txt"],
    help="Maximum 200 MB per file",
)

if uploaded:
    st.success(f"File received: {uploaded.name}")
    if st.button("🔍  Extract Text", type="primary"):
        extractor = HybridExtractor()
        with st.spinner("Processing …"):
            text = extractor.extract(uploaded.name, uploaded)

        if text:
            st.success("✅  Extraction complete")

            wc, cc = len(text.split()), len(text)
            col1, col2 = st.columns(2)
            col1.metric("Words", wc)
            col2.metric("Characters", cc)

            st.download_button(
                "📥  Download text",
                text,
                file_name=f"{Path(uploaded.name).stem}_extracted.txt",
                mime="text/plain",
            )

            with st.expander("📄  View text"):
                st.text_area("Extracted content", text, height=400)
        else:
            st.error("No text could be extracted.")
else:
    st.info("👆  Upload a document to begin.")
