# JurisGPT — Indian Legal AI

[![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-blue)](https://huggingface.co/Premchan369/JurisGPT)
[![Space](https://img.shields.io/badge/Space-Live_Demo-green)](https://huggingface.co/spaces/Premchan369/JurisGPT-Space)

> 2026-03-12 16:57 | Final: 70k+ corpus, all datasets verified working

**102,195 legal documents** | Qwen2-7B-Instruct + LegalBERT + FAISS RAG

## Quick Start
```bash
git clone https://github.com/Premchandyadav369/JurisGPT
cd JurisGPT
pip install -r requirements.txt
python app.py
```

## Corpus (102,195 documents)
| Dataset | Docs |
|---------|------|
| LexGLUE LEDGAR (contract provisions) | 59,983 |
| BillSum (US Congressional bills) | 18,949 |
| Multi_Legal_Pile | 0 |
| LexGLUE Case Hold | 3,000 |
| LexGLUE SCOTUS | 2,000 |
| LexGLUE ECtHR A+B | 0 |
| LexGLUE EURLex | 5,000 |
| CUAD Contracts | 0 |
| Pile-of-Law | 8,181 |
| Kaggle IPC + Constitution + Indian Legal | 5,000 |
| Hardcoded IPC + Constitution | 82 |

## Disclaimer
For informational purposes only. Always consult a qualified lawyer.
