🧠 Multi-Modal RAG Document Intelligence System

📌 Overview

This project implements a Multi-Modal Retrieval-Augmented Generation (RAG) pipeline capable of extracting insights from complex PDF documents containing:

✔ Text
✔ Tables
✔ Scanned images (OCR)

Users can upload a PDF through a Streamlit UI, ask natural language questions, and receive fact-grounded answers with citations to the document.
A summarization feature is also included to provide high-level document insights.

🚀 Features
Feature	Description
📄 Multi-modal ingestion	Extracts text, tables & OCR image text
🔍 Vector Search with FAISS	Fast top-K retrieval
🧩 Smart Chunking	Improves semantic context & accuracy
🤖 Gemini LLM Integration	Page-cited answers only from retrieved context
📊 Performance Metrics	Retrieval + Generation latency
📌 Summarization	5-bullet policy briefing from document

## 🧩 System Architecture

**PDF Upload**  
⬇  
**Text & Table Extraction** (pdfplumber)  
+  
**OCR for Images/Scanned Pages** (Tesseract + Poppler)  
⬇  
**Smart Chunking** (overlapping word windows)  
⬇  
**Embeddings Generation**  
*SentenceTransformer — all-MiniLM-L6-v2*  
⬇  
**FAISS Vector Indexing**  
⬇  
**Top-K Semantic Retrieval**  
⬇  
**LLM Response Generation**  
*Gemini 2.5 Flash*  
⬇  
📌 **Grounded Answer with Page-Level Citations**


## 🛠 Tech Stack

| Component          | Tool |
|-------------------|------|
| LLM               | Gemini-2.5-Flash |
| Embedding Model   | all-MiniLM-L6-v2 |
| Vector Store      | FAISS |
| OCR               | Tesseract + Poppler |
| PDF Extraction    | pdfplumber, pdf2image |
| Frontend          | Streamlit |
| Language          | Python |


▶️ How to Run the App
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Add API Key (Important)

Create a .env file in root folder:

GEMINI_API_KEY=YOUR_KEY_HERE

3️⃣ Run Streamlit
streamlit run app.py


Upload a PDF → Ask questions → View results with citations.

📈 Results & Performance Metrics
Metric	Value
Avg Retrieval Time	< 200 ms
Avg Answer Generation	~2–4 sec
Modalities supported	Text, Tables, OCR
Citation accuracy	High

📌 Performance tested using Qatar Economic PDF report.

📌 Deliverables

✔ Full Multi-Modal RAG pipeline

✔ Streamlit demo application

✔ Summarization bonus feature

✔ Secure environment variable handling for API key

🔒 Security

API keys are loaded from .env and not included in the repository.

.env
__pycache__/
*.pyc


are ignored via .gitignore.

📚 Future Enhancements (Optional)

Cross-modal reranking (RRF)

Evaluation dashboard with quality metrics

Support for multiple PDF uploads

Chat history memory
