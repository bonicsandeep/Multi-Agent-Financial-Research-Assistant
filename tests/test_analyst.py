# Unit test for Analyst agent
from agents.analyst import Analyst

def test_analyst():
    analyst = Analyst()
    chunks = [{"text": "Apple Inc. reported $100B in revenue.", "ticker": "AAPL", "date": "2025-01-01", "section": "main"}]
    answer = analyst.answer_query("What is the revenue for AAPL?", chunks)
    assert isinstance(answer, str)
