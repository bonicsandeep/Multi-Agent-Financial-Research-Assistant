#!/usr/bin/env python3
"""Run Finnhub ingest for AMD and report DB/Milvus/RAG status.
This script reads `.env` via `env_loader` so set up `.env` beforehand.
"""
import os
import sqlite3
import traceback

import env_loader  # loads .env into os.environ

from agents.finnhub_agent import FinnhubNewsAgent
from pymilvus import connections, Collection


def main():
    print('FINNHUB_API_KEY present:', bool(os.environ.get('FINNHUB_API_KEY')))
    try:
        agent = FinnhubNewsAgent()
        new = agent.poll_and_upsert(['AMD'], window_minutes=60 * 24)
        print('\nIngest result:')
        print('New articles count:', len(new))
        for a in new:
            print('-', a.get('title', '(no title)')[:200])
    except Exception as e:
        print('Ingest failed:')
        traceback.print_exc()

    db = os.path.join('data', 'finnhub_meta.db')
    if os.path.exists(db):
        try:
            print('\nSQLite articles for AMD:')
            conn = sqlite3.connect(db)
            cur = conn.cursor()
            cur.execute("SELECT id,ticker,title,published FROM articles WHERE ticker=? ORDER BY published DESC LIMIT 10", ('AMD',))
            rows = cur.fetchall()
            if not rows:
                print(' - no rows')
            for r in rows:
                print('-', r[1], (r[2] or '')[:200], 'published:', r[3])
            conn.close()
        except Exception:
            print('SQLite query failed:')
            traceback.print_exc()
    else:
        print('\nNo SQLite DB at', db)

    try:
        connections.connect(host='127.0.0.1', port='19530')
        col = Collection('financial_docs')
        print('\nMilvus collection:', col.name)
        try:
            print(' - num_entities:', col.num_entities)
        except Exception:
            print(' - num_entities error')
    except Exception:
        print('\nMilvus connection/collection error:')
        traceback.print_exc()

    try:
        from agents.rag_agent import RAGAgent
        rag = RAGAgent()
        hits = rag.run('AMD stocks', ['AMD'])
        print('\nRAGAgent hits for AMD:')
        print(hits)
    except Exception:
        print('\nRAGAgent run failed:')
        traceback.print_exc()


if __name__ == '__main__':
    main()
