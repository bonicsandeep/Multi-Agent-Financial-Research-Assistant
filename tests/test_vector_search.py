# Unit test for vector_search
from tools.vector_search import vector_search

def test_vector_search():
    results = vector_search("What is the revenue for AAPL?")
    assert isinstance(results, list)
