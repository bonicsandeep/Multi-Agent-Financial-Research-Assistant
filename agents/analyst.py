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
