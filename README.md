# ⚖️ Legal Eagle: AI-Powered Legal Document Analysis Platform

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-blue?style=for-the-badge)](https://swastidip-legal-eagle-app.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-API-orange?style=flat-square&logo=google)](https://ai.google.dev/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-green?style=flat-square&logo=meta)](https://github.com/facebookresearch/faiss)

> **Legal Eagle** transforms complex legal document analysis with cutting-edge AI, making legal information accessible to everyone through intelligent OCR, entity extraction, and conversational Q&A.

---

## 🎯 **Problem Statement**

Legal documents are notoriously complex, time-consuming to analyze, and often inaccessible to individuals and small businesses. Key challenges include:

- **Language Barrier**: Legal jargon is difficult for non-lawyers to understand
- **Time Intensive**: Manual document review takes hours or days
- **Limited Access**: Expensive legal consultation for simple document analysis
- **Format Diversity**: Documents come in various formats (PDF, scanned images, DOCX)
- **Information Extraction**: Finding specific entities, dates, and financial terms manually

## 🚀 **Solution: Legal Eagle**

An **AI-powered legal document analysis platform** that combines:
- ✅ **Hybrid OCR + Text Extraction** for any document format
- ✅ **Legal-specific AI models** (InLegalBERT) for accurate understanding
- ✅ **Intelligent entity extraction** for companies, people, dates, money
- ✅ **Conversational Q&A** with cited sources
- ✅ **Smart document classification** across legal categories
- ✅ **Real-time processing** with persistent vector indexing

---

## 🎥 **Live Demo**

**🌟 [Try Legal Eagle Now →](https://swastidip-legal-eagle-app.streamlit.app/)**

*Upload any legal document (PDF, DOCX, RTF, TXT) and experience AI-powered legal analysis in seconds!*

---

## ✨ **Key Features**

### 🔍 **Hybrid Document Processing**
- **Multi-format Support**: PDF, DOCX, DOC, RTF, TXT
- **OCR Fallback**: Tesseract-powered text extraction from scanned documents
- **Smart Text Extraction**: Native text extraction with OCR backup

### 🤖 **AI-Powered Analysis**
- **InLegalBERT Integration**: Specialized legal language understanding
- **Google Gemini API**: Advanced reasoning and conversation
- **FAISS Vector Search**: Lightning-fast similarity search with HNSW indexing
- **Section-Aware Chunking**: Intelligent document segmentation

### 🏢 **Legal Entity Extraction**
- **Companies**: Automatic detection of organizations, LLCs, corporations
- **People**: Names, titles, and roles identification
- **Financial Terms**: Money amounts, currencies, financial references
- **Important Dates**: Contract dates, deadlines, milestones
- **Real-time Dashboard**: Visual entity counts and statistics

### 💬 **Intelligent Q&A System**
- **Conversational Interface**: Natural language questions
- **Cited Responses**: Every answer includes source references
- **Context-Aware**: Understanding of legal document structure
- **Quick Actions**: Pre-built questions for common legal queries

### 📋 **Document Classification**
- **Contract/Agreement**: Terms, conditions, parties involved
- **Legal Filing**: Court documents, motions, complaints
- **Corporate Document**: Board resolutions, bylaws, SEC filings
- **Insurance Document**: Policies, claims, coverage terms
- **Real Estate Document**: Deeds, mortgages, lease agreements

---

## 🛠️ **Technology Stack**

| Category | Technologies |
|----------|-------------|
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) |
| **AI/ML** | ![Google Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=flat&logo=google&logoColor=white) ![HuggingFace](https://img.shields.io/badge/🤗_InLegalBERT-yellow?style=flat) ![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat&logo=spacy&logoColor=white) |
| **Vector DB** | ![FAISS](https://img.shields.io/badge/FAISS-1877F2?style=flat&logo=meta&logoColor=white) |
| **Document Processing** | ![PyPDF2](https://img.shields.io/badge/PyPDF2-FF0000?style=flat) ![pytesseract](https://img.shields.io/badge/Tesseract_OCR-4B8BBE?style=flat) ![python-docx](https://img.shields.io/badge/python--docx-4B8BBE?style=flat) |
| **Deployment** | ![Streamlit Cloud](https://img.shields.io/badge/Streamlit_Cloud-FF4B4B?style=flat&logo=streamlit&logoColor=white) |

---

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   AI Processing  │    │   User Interface│
│   Ingestion     │───▶│      Engine      │───▶│   & Responses   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ • PDF/DOCX/RTF  │    │ • InLegalBERT    │    │ • Q&A Interface │
│ • OCR Fallback  │    │ • Gemini API     │    │ • Entity Dashboard│
│ • Text Extract  │    │ • FAISS Search   │    │ • Source Citations│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+
- Google Gemini API key
- System dependencies for OCR

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/legal-eagle.git
   cd legal-eagle
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies (Ubuntu/Debian)**
   ```bash
   sudo apt-get update
   sudo apt-get install tesseract-ocr poppler-utils
   ```

4. **Set up environment variables**
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key_here"
   ```
   Or create a `.streamlit/secrets.toml` file:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   GEMINI_CHAT_MODEL = "gemini-2.5-flash"
   DEBUG = false
   ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

6. **Open in browser**: `http://localhost:8501`

---

## 📁 **Project Structure**

```
legal-eagle/
├── main.py                 # 🏠 Main Streamlit application
├── legal_bert_rag.py      # 🧠 Core RAG system with Gemini integration
├── utils/
│   ├── extractors.py      # 📄 Document extraction (PDF/DOCX/OCR)
│   ├── legal_processors.py # ⚖️ Legal NLP processing & entity extraction
│   └── model_cache.py     # 🚀 Model caching for performance
├── requirements.txt        # 📦 Python dependencies
├── .streamlit/
│   └── secrets.toml       # 🔐 API keys and configuration
└── .cache_faiss/          # 💾 Persistent vector index storage
```

---

## 💡 **Usage Examples**

### 1. **Contract Analysis**
```
📤 Upload: employment_contract.pdf
🤖 Ask: "What are the key terms and conditions?"
📋 Result: Detailed analysis with salary, benefits, termination clauses
```

### 2. **Entity Extraction**
```
📤 Upload: merger_agreement.docx
🏢 Extract: Companies, financial amounts, important dates
📊 Dashboard: Visual summary of all entities
```

### 3. **Legal Document Q&A**
```
📤 Upload: lease_agreement.pdf
❓ Questions:
  • "Who are the parties involved?"
  • "What are the financial terms?"
  • "What are the tenant's obligations?"
```

---

## 🎯 **Hackathon Highlights**

### 🏆 **Innovation Points**
- **Legal-Specific AI**: First implementation using InLegalBERT for legal document understanding
- **Hybrid Architecture**: Seamless fallback from native Gemini to OpenAI compatibility
- **Real-World Ready**: Production-grade error handling and user experience
- **Scalable Design**: FAISS indexing for enterprise-level document volumes

### ⚡ **Technical Excellence**
- **Performance**: Persistent FAISS indexing with HNSW for sub-second search
- **Reliability**: Multi-layer fallback systems for OCR and AI processing
- **User Experience**: Intuitive interface with real-time feedback
- **Deployment**: Cloud-ready with Streamlit Cloud integration

### 🌍 **Social Impact**
- **Democratizing Legal Access**: Making legal analysis accessible to everyone
- **Time Efficiency**: Reducing hours of manual review to minutes
- **Cost Reduction**: Eliminating expensive consultation for basic document analysis
- **Education**: Helping users understand complex legal concepts

---

## 🔮 **Future Roadmap**

### Phase 1: Enhanced AI Capabilities
- [ ] Multi-language legal document support
- [ ] Advanced contract risk analysis
- [ ] Legal precedent matching
- [ ] Automated compliance checking

### Phase 2: Enterprise Features
- [ ] Bulk document processing
- [ ] Team collaboration tools
- [ ] API for third-party integration
- [ ] Advanced security and encryption

### Phase 3: Specialized Modules
- [ ] Intellectual property analysis
- [ ] Regulatory compliance checker
- [ ] Legal document drafting assistance
- [ ] Court filing preparation tools

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---


## 🙏 **Acknowledgments**

- **Google Gemini API** for advanced AI capabilities
- **nlpaueb/legal-bert-base-uncased** for legal language understanding
- **Facebook Research FAISS** for efficient vector search
- **Streamlit** for rapid prototyping and deployment
- **Open source community** for amazing libraries and tools

---

## 📊 **Project Stats**

![Lines of Code](https://img.shields.io/badge/Lines_of_Code-2000+-blue?style=flat-square)
![Files](https://img.shields.io/badge/Files-15+-green?style=flat-square)
![Languages](https://img.shields.io/badge/Languages-Python-yellow?style=flat-square)

---

<div align="center">

**🌟 [Try Legal Eagle Live Demo](https://swastidip-legal-eagle-app.streamlit.app/) 🌟**

*Making legal document analysis accessible to everyone, one document at a time.*

</div>

