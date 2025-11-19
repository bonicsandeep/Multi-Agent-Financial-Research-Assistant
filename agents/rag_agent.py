"""
RAG Agent: Retrieves relevant information from vector database containing SEC filings and research reports.
Stub implementation for future integration with ChromaDB, Milvus, etc.
"""

from pymilvus import connections, Collection

try:
    from rag.embedder import OllamaEmbeddings
except Exception:
    OllamaEmbeddings = None


class RAGAgent:
    def __init__(self, host="127.0.0.1", port="19530"):
        try:
            connections.connect(host=host, port=port)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Milvus at {host}:{port}. Is the Docker service running? Error: {e}")
        self.collection_name = "financial_docs"

    def run(self, query: str, companies: list[str]) -> dict:
        """Run RAG retrieval: embed query+company and search Milvus collection.

        Returns a dict mapping company -> list of hit ids (or error messages).
        """
        results = {}
        col = Collection(self.collection_name)

        # prepare embedder (fallback to random if Ollama not available)
        embedder = None
        if OllamaEmbeddings is not None:
            try:
                embedder = OllamaEmbeddings()
            except Exception:
                embedder = None

        for company in companies:
            try:
                text = f"{query} {company}"
                if embedder:
                    emb = embedder.embed(text)
                else:
                    # deterministic fallback: simple hash-based vector
                    import hashlib
                    h = hashlib.sha256(text.encode()).digest()
                    emb = [float(b) / 255.0 for b in h[:64]]

                # Ensure embedding is a list of floats
                emb = [float(x) for x in emb]

                search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
                # Collection.search expects a list of queries
                hits = col.search([emb], anns_field='vector', param=search_params, limit=5)

                # hits is a list (one per query) of lists of hits
                company_hits = []
                if hits and hits[0]:
                    for h in hits[0]:
                        try:
                            # prefer an id; metadata/text fields may not exist
                            company_hits.append(getattr(h, 'id', None) or getattr(h, 'entity', {}).get('id'))
                        except Exception:
                            company_hits.append(str(h))

                results[company] = company_hits
            except Exception as e:
                results[company] = [f"Error: {str(e)}"]
        return results
