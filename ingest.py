# Ingestion script for SEC 10-K data
import os
from pymilvus import MilvusClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.embedder import OllamaEmbeddings

def ingest_documents(data_dir, vectorstore_path):
    client = MilvusClient(uri=vectorstore_path)
    embedder = OllamaEmbeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

    # Create collection if not exists
    if "financial_docs" not in client.list_collections():
        client.create_collection(
            collection_name="financial_docs",
            dimension=1024,
            metric_type="L2",
            schema={"id": "int64", "text": "str", "ticker": "str", "date": "str", "section": "str"}
        )

    doc_id = 0
    for fname in os.listdir(data_dir):
        if not fname.endswith(".txt"): continue
        with open(os.path.join(data_dir, fname), "r") as f:
            text = f.read()
        # Example metadata extraction
        ticker = fname.split("_")[0]
        date = "2025-01-01"  # Placeholder
        section = "main"      # Placeholder
        chunks = splitter.split_text(text)
        for chunk in chunks:
            embedding = embedder.embed(chunk)
            client.insert(
                collection_name="financial_docs",
                data={
                    "id": doc_id,
                    "text": chunk,
                    "ticker": ticker,
                    "date": date,
                    "section": section,
                    "embedding": embedding
                }
            )
            doc_id += 1
