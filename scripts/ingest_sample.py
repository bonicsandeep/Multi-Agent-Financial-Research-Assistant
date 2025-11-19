#!/usr/bin/env python3
"""Simple sample ingest: embed sample texts with Ollama and insert into Milvus collection
"""
from pymilvus import connections, Collection
from rag.embedder import OllamaEmbeddings

def main():
    connections.connect(host='127.0.0.1', port='19530')
    col = Collection('financial_docs')
    print('Collection:', col.name)
    print('Current entity count:', col.num_entities)

    samples = [
        'Apple reported better-than-expected earnings and raised guidance.',
        'Market volatility increased due to macroeconomic data and inflation concerns.',
        'Tesla announced a new production target and supply chain improvements.'
    ]

    emb = OllamaEmbeddings()
    vectors = []
    for s in samples:
        v = emb.embed(s)
        vectors.append(list(v))

    start_id = int(col.num_entities)
    ids = [start_id + i + 1 for i in range(len(vectors))]

    print('Inserting', len(vectors), 'vectors...')
    res = col.insert([ids, vectors])
    # flush to make sure data is persisted
    try:
        col.flush()
    except Exception:
        pass

    print('Insert result:', res)
    print('New entity count:', col.num_entities)

if __name__ == '__main__':
    main()
