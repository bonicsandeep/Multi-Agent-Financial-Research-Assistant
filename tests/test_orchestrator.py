import pytest
from agents.orchestrator import Orchestrator

def test_orchestrator_basic():
    orchestrator = Orchestrator()
    query = "Compare Nvidia and AMD for AI investment, next 12 months."
    result = orchestrator.orchestrate(query)
    assert isinstance(result, str)
    assert "Nvidia" in result or "AMD" in result
    assert "QUERY UNDERSTANDING" in result
    assert "AGENT COORDINATION PLAN" in result
    assert "INVESTIGATION RESULTS" in result
    assert "INVESTMENT RECOMMENDATION" in result
    assert "SOURCE CITATIONS" in result
