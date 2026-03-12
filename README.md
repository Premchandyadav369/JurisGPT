# ⚖️ JurisGPT — Indian Legal AI Assistant

[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Premchan369/JurisGPT-blue)](https://huggingface.co/Premchan369/JurisGPT)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow)](LICENSE)

> Last updated: 2026-03-12 05:06

AI-powered Indian legal assistant using **Qwen2-7B + LegalBERT + RAG** on **5027 legal documents**.

---

## 🚀 Features
- 💬 **Legal Chat** — Ask any question about Indian law
- 📄 **PDF Analyzer** — Upload legal documents for instant AI analysis
- ⚖️ **Outcome Prediction** — LegalBERT case outcome classifier
- 🔍 **RAG Retrieval** — Semantic search over Indian legal corpus
- 📚 **IPC & Constitution Explorer** — Look up any section or article

---

## 🏗️ Architecture
```
User Query
    ↓
LegalBERT Outcome Predictor
    ↓
FAISS RAG Retrieval (5027 docs)
    ↓
Qwen2-7B-Instruct LLM (4-bit)
    ↓
Structured Legal Analysis
```

---

## ⚙️ Models Used
| Component | Model |
|-----------|-------|
| Main LLM | Qwen2-7B-Instruct (4-bit quantized) |
| Embeddings | BAAI/bge-large-en |
| Classifier | LegalBERT (nlpaueb/legal-bert-base-uncased) |
| Vector DB | FAISS IndexFlatIP |

---

## 📚 Legal Corpus (5027 documents)
| Dataset | Docs |
|---------|------|
| LexGLUE Case Law | 3000 |
| SCOTUS Court Decisions | 2000 |
| Human Rights Cases | 0 |
| Indian Legal QA | 0 |
| MultiLexSum Summaries | 0 |
| EUR-Lex Legal Texts | 0 |
| LegalBench Tasks | 0 |
| IPC Sections (hardcoded) | 17 |
| Constitution Articles | 10 |

---

## 💻 Install & Run
```bash
git clone https://github.com/Premchandyadav369/JurisGPT
cd JurisGPT
pip install -r requirements.txt
python app.py
```

---

## 🤗 Model on HuggingFace
➡️ https://huggingface.co/Premchan369/JurisGPT

---

## ⚠️ Disclaimer
JurisGPT provides general legal information for **educational purposes only**.
Always consult a qualified lawyer for serious legal matters.

---

## 📜 License
Apache 2.0
