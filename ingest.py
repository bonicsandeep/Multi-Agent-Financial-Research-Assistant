# Ingestion script for SEC 10-K data
import os
from pymilvus import MilvusClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.embedder import OllamaEmbeddings
from typing import List

def read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    text_parts: List[str] = []
    try:
        reader = PdfReader(path)
        for p in reader.pages:
            text_parts.append(p.extract_text() or "")
    except Exception:
        return ""
    return "\n".join(text_parts)

def ingest_documents(data_dir, milvus_host="localhost", milvus_port="19530"):
    client = MilvusClient(host=milvus_host, port=milvus_port)
    try:
        embedder = OllamaEmbeddings()
        # quick check to ensure Ollama reachable
        sample = embedder.embed("test")
        emb_dim = len(sample) if isinstance(sample, (list, tuple)) else None
    except Exception as e:
        raise RuntimeError(f"Ollama embedder init failed: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

    # ensure collection uses detected dimension
    coll_name = "financial_docs"
    if coll_name not in client.list_collections():
        if emb_dim is None:
            raise RuntimeError("Unable to detect embedding dimension from Ollama.")
        client.create_collection(
            collection_name=coll_name,
            dimension=emb_dim,
            metric_type="L2",
        )

    doc_id = 0
    processed = 0
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if not (fname.lower().endswith(".txt") or fname.lower().endswith(".pdf")):
                continue
            fpath = os.path.join(root, fname)
            try:
                if fname.lower().endswith(".pdf"):
                    text = read_pdf(fpath)
                else:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                if not text:
                    print("empty text:", fpath)
                    continue
                ticker = fname.split("_")[0]
                date = "2025-01-01"
                section = "main"
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    embedding = embedder.embed(chunk)
                    client.insert(
                        collection_name=coll_name,
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
                processed += 1
                print(f"Processed: {fpath} -> {len(chunks)} chunks")
            except Exception as ex:
                print(f"Failed {fpath}: {ex}")

    print(f"Done. Files processed: {processed}, total chunks inserted (approx): {doc_id}")
