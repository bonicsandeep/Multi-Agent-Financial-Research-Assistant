# RAG evaluation script
from tools.vector_search import vector_search
from agents.analyst import Analyst

def evaluate_rag(queries, expected_answers):
    correct = 0
    for query, expected in zip(queries, expected_answers):
        retrieved_chunks = vector_search(query)
        analyst = Analyst()
        answer = analyst.answer_query(query, retrieved_chunks)
        if expected.lower() in answer.lower():
            correct += 1
    precision = correct / len(queries)
    print(f"Precision: {precision:.2f}")
