# JurisGPT Changelog

## [2026-03-12 05:06]
Update JurisGPT — fixed datasets, improved RAG pipeline

### Changes
- Fixed dataset loading (replaced broken HF dataset IDs)
- Added hardcoded IPC sections and Constitution articles
- Improved RAG retrieval with multiple fallback sources
- Added MultiLexSum and EUR-Lex datasets
- Fixed LegalBench to use parquet format (no loading script)
- Total corpus: 5027 documents
- Added PDF analyzer, Quick Q&A, IPC Explorer tabs in Gradio UI
