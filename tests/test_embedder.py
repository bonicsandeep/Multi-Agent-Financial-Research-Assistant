# Unit test for OllamaEmbeddings
from rag.embedder import OllamaEmbeddings

def test_embedder():
    embedder = OllamaEmbeddings()
    vec = embedder.embed("Test sentence for embedding.")
    assert isinstance(vec, list)
    assert len(vec) > 0
