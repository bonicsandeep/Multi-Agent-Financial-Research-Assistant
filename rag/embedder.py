"""Deterministic embedding helper that works without external services."""

from __future__ import annotations

import hashlib
from typing import Iterable, List


class OllamaEmbeddings:
    """Lightweight embedding generator based on token hashing.

    Instead of invoking a remote Ollama server, we approximate embeddings by
    hashing tokens into a fixed-length vector. This keeps the public API
    compatible with the original implementation while enabling fast, offline
    unit tests. The resulting vectors are not semantically rich but sufficient
    for demonstrating retrieval logic in this sample project.
    """

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        vector = [0.0] * self.dimension
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(self.dimension):
                vector[i] += digest[i % len(digest)] / 255.0
        # Normalize so cosine similarity works as expected.
        norm = sum(component * component for component in vector) ** 0.5 or 1.0
        return [component / norm for component in vector]

    def _tokenize(self, text: str) -> Iterable[str]:
        return [token for token in text.lower().split() if token]
