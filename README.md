# Multi-Agent Financial Research Assistant

Offline GenAI system for financial research using LangGraph, Ollama, Milvus Lite, and Streamlit.

## Features
- Multi-agent RAG pipeline
- Local LLM and embeddings
- Vector search over SEC filings
- Streamlit UI

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Start Ollama and Milvus Lite
3. Run `ingest.py` to load data
4. Launch UI: `streamlit run app.py`
