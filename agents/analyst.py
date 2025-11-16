# Analyst agent logic for LangGraph
import requests

class Analyst:
    def __init__(self, ollama_model="mistral", host="http://localhost:11434"):
        self.model = ollama_model
        self.host = host

    def answer_query(self, query, retrieved_chunks):
        context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
        prompt = f"Answer the following financial research question using only the provided context.\n\nQuestion: {query}\n\nContext:\n{context}\n\nCite relevant sections."
        url = f"{self.host}/api/chat"
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
