#!/usr/bin/env python3
"""Add `milvus_id` column to Finnhub metadata DB and backfill from hex ids.

This script is idempotent and can be run safely multiple times.
"""
import os
import sqlite3

DB = os.path.join('data', 'finnhub_meta.db')


def main():
    if not os.path.exists(DB):
        print('DB not found, nothing to migrate:', DB)
        return

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    # add column if not exists (SQLite lacks ALTER ADD IF NOT EXISTS)
    cols = [r[1] for r in cur.execute("PRAGMA table_info(articles)")]
    if 'milvus_id' not in cols:
        cur.execute('ALTER TABLE articles ADD COLUMN milvus_id INTEGER')
        conn.commit()
        print('Added milvus_id column')

    # backfill rows where milvus_id is NULL
    cur.execute('SELECT id FROM articles WHERE milvus_id IS NULL OR milvus_id = ""')
    rows = cur.fetchall()
    print('Backfilling', len(rows), 'rows')
    for (hexid,) in rows:
        try:
            mid = int(int(hexid, 16) % (2 ** 63 - 1))
            cur.execute('UPDATE articles SET milvus_id=? WHERE id=?', (mid, hexid))
        except Exception:
            continue
    conn.commit()
    conn.close()
    print('Migration complete')


if __name__ == '__main__':
    main()
