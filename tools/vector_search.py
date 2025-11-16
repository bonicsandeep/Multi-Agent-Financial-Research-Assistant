# Vector search tool using Milvus Lite and Ollama embeddings
from pymilvus import MilvusClient
from rag.embedder import OllamaEmbeddings

def vector_search(query, top_k=5, vectorstore_path="./vectorstore/db"):
    client = MilvusClient(uri=vectorstore_path)
    embedder = OllamaEmbeddings()
    query_vec = embedder.embed(query)
    results = client.search(
        collection_name="financial_docs",
        data=[query_vec],
        limit=top_k,
        output_fields=["text", "ticker", "date", "section"]
    )
    return results[0] if results else []
