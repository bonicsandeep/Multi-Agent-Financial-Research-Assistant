"""Analyst agent that synthesizes retrieved evidence without external APIs.

The original implementation depended on a locally running Ollama endpoint,
which made automated testing and offline execution brittle. The updated
version uses lightweight heuristics to summarize retrieved chunks so that the
entire project can run deterministically in constrained environments while
preserving the expected interface for LangGraph or other orchestrators.
"""

from __future__ import annotations

from typing import List, Mapping


class Analyst:
    """Summarizes retrieved context for a financial question."""

    def __init__(self, max_snippets: int = 5) -> None:
        self.max_snippets = max_snippets

    def answer_query(self, query: str, retrieved_chunks: List[Mapping[str, str]]) -> str:
        """Return a concise answer grounded in the retrieved chunks.

        Each chunk is converted into a short citation-aware highlight. The
        analyst prioritizes the most relevant snippets (first in the list) and
        echoes the metadata so downstream reviewers can trace the origin of
        every statement.
        """
        if not retrieved_chunks:
            return f"No supporting documents were retrieved to answer: '{query}'."

        highlights = []
        for chunk in retrieved_chunks[: self.max_snippets]:
            snippet = chunk.get("text", "").strip()
            if not snippet:
                continue
            metadata_bits = []
            for key in ("ticker", "section", "date"):
                if chunk.get(key):
                    metadata_bits.append(f"{key.upper()}: {chunk[key]}")
            metadata = " | ".join(metadata_bits) if metadata_bits else "UNSPECIFIED SOURCE"
            highlights.append(f"- {snippet} ({metadata})")

        body = "\n".join(highlights) if highlights else "- No textual evidence found."
        return f"Question: {query}\nKey supporting facts:\n{body}"

    def run(self, query: str, entities: list[str], news: dict, data: dict, chunks: dict) -> tuple[str, list]:
        """Compatibility wrapper used by the Orchestrator.

        The orchestrator expects `run` to return a tuple (answer, citations).
        We produce a lightweight answer by combining available chunks and
        fallback to news/data when chunks are missing.
        """
        # Aggregate retrieved chunks into a flat list of mappings
        retrieved = []
        if isinstance(chunks, dict):
            for company, c in chunks.items():
                # c may be a list of strings or error messages
                if isinstance(c, list):
                    for idx, item in enumerate(c):
                        retrieved.append({"text": str(item), "ticker": company, "section": f"chunk_{idx}"})
        # If no retrieved chunks, try to synthesize short facts from data/news
        if not retrieved:
            for company in entities:
                d = data.get(company) or data.get(company.upper()) or {}
                price = d.get("price", "N/A") if isinstance(d, dict) else "N/A"
                retrieved.append({"text": f"Price snapshot: {price}", "ticker": company, "section": "data"})
                # include one news item if available
                n = news.get(company, {})
                if isinstance(n, dict) and n.get("summary"):
                    retrieved.append({"text": n.get("summary"), "ticker": company, "section": "news"})

        answer = self.answer_query(query, retrieved)
        citations: list[str] = []
        return answer, citations
