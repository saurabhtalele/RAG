# 📚 PDF RAG Playground

Ask questions about your PDFs using **local AI** (Ollama), **smart PDF parsing** (Docling), and **fast retrieval** (ChromaDB).

## ✅ Requirements
- Python 3.9+
- [Ollama](https://ollama.com/) running locally (`ollama serve`)
- Models installed:  
  ```bash
  ollama pull gemma2:2b
  ollama pull nomic-embed-text