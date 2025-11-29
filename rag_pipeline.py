# rag_pipeline.py

import io
import uuid
from pathlib import Path
from typing import List, Dict

import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import google.generativeai as genai
import os
import time  # for latency metrics

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()



# ---------------- Gemini Configuration ----------------

# For local testing: put your real key string here (DO NOT commit this to Git / share it)
genai.configure(api_key="YOUR API KEY")


# ---------------- External Tool Paths ----------------

# Set Tesseract OCR command path (update if needed for your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Poppler path for pdf2image (for OCR of PDF pages)
POPPLER_PATH = r"\poppler-24.08.0\Library\bin"


# ---------- PDF → raw docs (text + tables + OCR) ----------

def extract_text_and_tables_from_bytes(pdf_bytes: bytes) -> List[Dict]:
    """
    Extracts:
      - page text (modality='text')
      - tables as text (modality='table')
      - OCR text from rendered pages (modality='ocr')
    Returns a list of dicts with: id, page, modality, content.
    """
    docs: List[Dict] = []
    file_obj = io.BytesIO(pdf_bytes)

    # --- Extract TEXT + TABLES via pdfplumber ---
    with pdfplumber.open(file_obj) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            # Text extraction
            text = page.extract_text() or ""
            if text.strip():
                docs.append({
                    "id": str(uuid.uuid4()),
                    "page": page_num,
                    "modality": "text",
                    "content": text.strip()
                })

            # Table extraction
            tables = page.extract_tables() or []
            for table in tables:
                rows = [", ".join([(cell or "").strip() for cell in row]) for row in table]
                table_str = "\n".join(rows)
                if table_str.strip():
                    docs.append({
                        "id": str(uuid.uuid4()),
                        "page": page_num,
                        "modality": "table",
                        "content": table_str.strip()
                    })

    # --- OCR for scanned pages / images (convert PDF pages to images) ---
    images = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
    for page_num, img in enumerate(images, start=1):
        ocr_text = pytesseract.image_to_string(img)
        if ocr_text.strip():
            docs.append({
                "id": str(uuid.uuid4()),
                "page": page_num,
                "modality": "ocr",
                "content": ocr_text.strip()
            })

    return docs


# ---------- Chunking ----------

def chunk_text(text: str, max_words: int = 200, overlap: int = 40) -> List[str]:
    """
    Split a long text into overlapping word chunks.
    """
    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap

    return chunks


def build_chunks(docs: List[Dict], max_words: int = 200, overlap: int = 40) -> List[Dict]:
    """
    Convert raw docs into chunked docs with:
      - page
      - modality
      - chunk text
    """
    chunks: List[Dict] = []
    for doc in docs:
        for ch in chunk_text(doc["content"], max_words, overlap):
            chunks.append({
                "page": doc["page"],
                "modality": doc["modality"],
                "chunk": ch
            })
    return chunks


# ---------- RAG Class ----------

class RAGPipeline:
    def __init__(self, model: SentenceTransformer, index: faiss.Index, chunks: List[Dict]):
        self.model = model
        self.index = index
        self.chunks = chunks
        self.llm = genai.GenerativeModel("gemini-2.5-flash")

    # -------- Embeddings & Retrieval --------

    def embed_query(self, query: str):
        emb = self.model.encode([query])
        return np.array(emb).astype("float32")

    def retrieve(self, query: str, top_k: int = 5):
        q_emb = self.embed_query(query)
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            m = dict(self.chunks[idx])
            m["distance"] = float(dist)
            results.append(m)
        return results

    def build_context(self, retrieved: List[Dict]):
        """
        Build the context string sent to the LLM, with explicit source headers.
        """
        blocks = []
        for i, r in enumerate(retrieved, start=1):
            header = f"[Source {i} | page {r['page']} | {r['modality']}]"
            blocks.append(header + "\n" + r["chunk"])
        return "\n\n".join(blocks)

    # -------- LLM Calls --------

    def ask_llm(self, question: str, context: str):
        prompt = f"""
Use only the context to answer.
Cite pages for your answer.
If not found, respond:
"I couldn't find this information in the document."

Question: {question}
Context:
{context}
"""
        resp = self.llm.generate_content(prompt)
        return resp.text.strip()

    def summarize_document(self, max_chunks: int = 60) -> str:
        """
        Summarize the uploaded PDF into a short briefing.
        We take up to `max_chunks` chunks from the document as context.
        """
        pieces = []
        for c in self.chunks[:max_chunks]:
            header = f"[Page {c['page']} | {c['modality']}]"
            pieces.append(header + "\n" + c["chunk"])
        context = "\n\n".join(pieces)

        prompt = f"""
You are summarizing an economic report (similar to an IMF Article IV staff report).

Using ONLY the context below, produce a concise briefing for a policymaker.

Requirements:
- At most 5 bullet points.
- Mention key numbers if available (growth, inflation, fiscal balance, current account).
- Mention key medium-term drivers (e.g., LNG expansion, reforms) if present.
- Do not invent facts that are not in the context.

Context:
{context}
"""

        resp = self.llm.generate_content(prompt)
        return resp.text.strip()

    def answer_question(self, question: str, top_k: int = 5):
        """
        End-to-end QA:
        - measure retrieval & generation latency
        - compute simple retrieval metrics
        - return answer, sources, and metrics
        """
        t0 = time.perf_counter()
        retrieved = self.retrieve(question, top_k)
        t1 = time.perf_counter()

        context = self.build_context(retrieved)
        t2 = time.perf_counter()

        answer = self.ask_llm(question, context)
        t3 = time.perf_counter()

        distances = [r.get("distance", 0.0) for r in retrieved] or [0.0]

        metrics = {
            "k": len(retrieved),
            "retrieval_time_ms": (t1 - t0) * 1000,
            "context_build_time_ms": (t2 - t1) * 1000,
            "generation_time_ms": (t3 - t2) * 1000,
            "total_time_ms": (t3 - t0) * 1000,
            "avg_distance": float(np.mean(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
        }

        sources = [
            {
                "page": r["page"],
                "modality": r["modality"],
                "snippet": r["chunk"][:300] + "..."
            }
            for r in retrieved
        ]

        return {
            "answer": answer,
            "sources": sources,
            "metrics": metrics,
        }


# ---------- Build RAG from Uploaded PDF ----------

def build_rag_from_pdf_bytes(pdf_bytes: bytes) -> RAGPipeline:
    """
    High-level helper: from raw PDF bytes → RAGPipeline.
    Everything is in memory; no files are written.
    """
    docs = extract_text_and_tables_from_bytes(pdf_bytes)
    chunks = build_chunks(docs)
    texts = [c["chunk"] for c in chunks]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return RAGPipeline(model, index, chunks)
