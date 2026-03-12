# ⚖️ JurisGPT — Indian Legal AI Assistant

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Premchan369/JurisGPT-blue)](https://huggingface.co/Premchan369/JurisGPT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)

An AI-powered Indian legal assistant using **Qwen2-7B + LegalBERT + RAG** on a corpus of Indian legal documents.

---

## 🚀 Features

- 💬 **Legal Chat** — Ask any question about Indian law
- 📄 **PDF Analyzer** — Upload legal documents for AI analysis
- ⚖️ **Outcome Prediction** — LegalBERT case outcome classifier
- 🔍 **RAG Retrieval** — Semantic search over Indian legal corpus
- 📚 **IPC Explorer** — Look up any IPC section or Constitutional article

---

## 🏗️ Architecture

```
User Query
    ↓
LegalBERT Outcome Predictor
    ↓
FAISS RAG (Indian Constitution + IPC + Case Law)
    ↓
Qwen2-7B-Instruct LLM
    ↓
Legal Analysis
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Premchandyadav369/JurisGPT.git
cd JurisGPT
pip install -r requirements.txt
```

---

## 💻 Usage

### Run Gradio Web App
```bash
python app.py
```

### Python API
```python
from inference import load_jurisgpt, ask

tokenizer, model, embed_model, index, meta = load_jurisgpt()
answer = ask("Explain IPC Section 302", tokenizer, model, embed_model, index, meta)
print(answer)
```

---

## 📚 Legal Corpus
| Dataset | Contents |
|---------|----------|
| Indian Constitution | All articles and amendments |
| IPC Sections | All Indian Penal Code sections |
| LexGLUE | Legal case law and reasoning |
| LegalBench | Legal reasoning tasks |
| Indian Court Judgments | Supreme Court & High Court cases |

---

## 🤗 Model on Hugging Face
➡️ https://huggingface.co/Premchan369/JurisGPT

---

## ⚠️ Disclaimer
JurisGPT provides general legal information for educational purposes only.
Always consult a qualified lawyer for serious legal matters.

---

## 📜 License
Apache 2.0
