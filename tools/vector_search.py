"""In-memory vector search powered by the deterministic embedding helper."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from rag.embedder import OllamaEmbeddings


SAMPLE_DOCUMENTS: Sequence[Dict[str, str]] = (
    {
        "text": "Apple Inc. reported $100B in revenue in 2024 with strong iPhone demand.",
        "ticker": "AAPL",
        "date": "2024-12-31",
        "section": "Financial Highlights",
    },
    {
        "text": "Microsoft Azure revenue grew 24% year-over-year driven by AI workloads.",
        "ticker": "MSFT",
        "date": "2024-12-31",
        "section": "Cloud",
    },
    {
        "text": "Tesla faces margin pressure as it cuts prices to defend EV market share.",
        "ticker": "TSLA",
        "date": "2024-11-15",
        "section": "Risk Factors",
    },
    {
        "text": "Nvidia data-center sales surpassed $60B thanks to hyperscale GPU demand.",
        "ticker": "NVDA",
        "date": "2024-10-31",
        "section": "MD&A",
    },
)


def vector_search(query: str, top_k: int = 5, vectorstore_path: str = "./vectorstore/db") -> List[Dict[str, str]]:
    """Return the most relevant sample documents for the query.

    The parameters mimic the previous Milvus-based implementation so the rest
    of the codebase does not require changes. Relevance scoring relies on the
    deterministic embeddings in ``rag.embedder`` and cosine similarity.
    """
    embedder = OllamaEmbeddings()
    query_vec = embedder.embed(query)
    doc_vectors = [(doc, embedder.embed(doc["text"])) for doc in SAMPLE_DOCUMENTS]
    ranked = sorted(doc_vectors, key=lambda pair: _cosine_similarity(query_vec, pair[1]), reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))
