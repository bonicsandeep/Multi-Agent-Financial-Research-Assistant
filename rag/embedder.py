# OllamaEmbeddings wrapper class
import requests

class OllamaEmbeddings:
    def __init__(self, model_name='embeddinggemma', host='http://localhost:11434'):
        self.model_name = model_name
        self.host = host

    def embed(self, text):
        url = f"{self.host}/api/embeddings"
        payload = {"model": self.model_name, "prompt": text}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["embedding"]
