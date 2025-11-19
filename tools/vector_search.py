"""Vector search backed by Milvus with SQLite metadata join.

If Milvus is not available or no collection exists, the function falls back to
the in-memory `SAMPLE_DOCUMENTS` behavior for UI stability.
"""

from __future__ import annotations

from typing import Dict, List, Sequence
import sqlite3
import os

from rag.embedder import OllamaEmbeddings
from pymilvus import connections, Collection


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
    """Search Milvus for `query` and return enriched metadata results.

    Returned items are dicts with at least `text`, `ticker`, `date`, `section`.
    If Milvus or metadata are not available, returns SAMPLE_DOCUMENTS.
    """
    embedder = OllamaEmbeddings()
    qvec = embedder.embed(query)

    try:
        connections.connect(host='127.0.0.1', port='19530')
        col = Collection('financial_docs')
    except Exception:
        # Milvus unavailable — fallback to samples
        doc_vectors = [(doc, embedder.embed(doc["text"])) for doc in SAMPLE_DOCUMENTS]
        ranked = sorted(doc_vectors, key=lambda pair: _cosine_similarity(qvec, pair[1]), reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
    try:
        hits = col.search([qvec], anns_field='vector', param=search_params, limit=top_k)
    except Exception:
        return []

    # hits is list per query
    result = []
    # open SQLite metadata DB
    db = os.path.join('data', 'finnhub_meta.db')
    conn = None
    if os.path.exists(db):
        conn = sqlite3.connect(db)

    if hits and hits[0]:
        for h in hits[0]:
            try:
                mid = getattr(h, 'id', None) or getattr(h, 'entity', {}).get('id')
            except Exception:
                mid = None

            meta = None
            if conn and mid is not None:
                cur = conn.cursor()
                cur.execute('SELECT title,summary,url,ticker,published FROM articles WHERE milvus_id=?', (mid,))
                row = cur.fetchone()
                if row:
                    title, summary, url, ticker, published = row
                    meta = {
                        'text': (title or '') + '\n' + (summary or ''),
                        'ticker': ticker or '',
                        'date': str(published) if published else '',
                        'section': 'News',
                        'url': url,
                    }

            if not meta:
                # fallback: return id-only entry
                meta = {'text': f'<hit id={mid}>', 'ticker': '', 'date': '', 'section': ''}

            result.append(meta)

    if conn:
        conn.close()

    # if no results, fall back to samples
    if not result:
        doc_vectors = [(doc, embedder.embed(doc["text"])) for doc in SAMPLE_DOCUMENTS]
        ranked = sorted(doc_vectors, key=lambda pair: _cosine_similarity(qvec, pair[1]), reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    return result


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))
