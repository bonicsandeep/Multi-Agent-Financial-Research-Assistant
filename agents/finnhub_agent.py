"""Finnhub integration: News polling and Market websocket template.

Usage:
  - Set environment variable `FINNHUB_API_KEY` or pass `api_key` to classes.
  - Run `FinnhubNewsAgent.poll_and_upsert(['AAPL','MSFT'])` to fetch recent news, embed via Ollama and upsert vectors.

This file provides a lightweight, opt-in integration that works without keys in mock mode.
"""
import os
import time
import sqlite3
import hashlib
import datetime
from typing import List, Optional

import requests

from pymilvus import connections, Collection

try:
    from rag.embedder import OllamaEmbeddings
except Exception:
    OllamaEmbeddings = None


DB_PATH = os.path.join('data', 'finnhub_meta.db')


class FinnhubNewsAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('FINNHUB_API_KEY')
        # ensure DB dir
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._ensure_db()
        # connect to Milvus
        connections.connect(host='127.0.0.1', port='19530')
        self.col = Collection('financial_docs')
        # embedder (optional)
        if OllamaEmbeddings is not None:
            try:
                self.embedder = OllamaEmbeddings()
            except Exception:
                self.embedder = None
        else:
            self.embedder = None

    def _ensure_db(self):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                ticker TEXT,
                url TEXT,
                title TEXT,
                summary TEXT,
                published TEXT,
                milvus_id INTEGER
            )"""
        )
        conn.commit()
        conn.close()

    def _article_id(self, ticker: str, url: str) -> str:
        return hashlib.sha1(f"{ticker}:{url}".encode()).hexdigest()

    def _iso_ts(self, ts) -> str:
        """Normalize Finnhub datetime (epoch seconds or ms) to ISO 8601 UTC string.

        Accepts int, float, str, or None. Returns an ISO 8601 UTC string or empty string on failure.
        """
        if ts is None or ts == "":
            return ""
        try:
            # If it's numeric string, convert
            if isinstance(ts, str):
                # some APIs return numeric strings
                if ts.isdigit():
                    ts_val = int(ts)
                else:
                    # attempt ISO parse: return as-is if looks like ISO
                    try:
                        # try to parse ISO-like strings
                        parsed = datetime.datetime.fromisoformat(ts)
                        return parsed.replace(tzinfo=datetime.timezone.utc).isoformat()
                    except Exception:
                        return ""
            elif isinstance(ts, (int, float)):
                ts_val = int(ts)
            else:
                return ""

            # Convert milliseconds to seconds if necessary
            if ts_val > 10**12:
                ts_val = ts_val // 1000

            dt = datetime.datetime.utcfromtimestamp(ts_val).replace(tzinfo=datetime.timezone.utc)
            return dt.isoformat()
        except Exception:
            return ""

    def fetch_company_news(self, ticker: str, from_date: str, to_date: str) -> List[dict]:
        """Fetch company news from Finnhub REST API.

        from_date/to_date are YYYY-MM-DD strings.
        If no API key is set, returns an empty list.
        """
        if not self.api_key:
            return []
        url = 'https://finnhub.io/api/v1/company-news'
        params = {'symbol': ticker, 'from': from_date, 'to': to_date, 'token': self.api_key}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _resolve_symbol(self, name: str) -> Optional[str]:
        """Resolve a company name or symbol-like token to a Finnhub ticker symbol.

        - If `name` already looks like an ALL-CAPS ticker (2-5 letters), returns it.
        - Otherwise calls Finnhub `/search` to attempt resolution and returns the first match's symbol.
        - Returns None when no API key is configured or when no match is found.
        """
        if not name:
            return None
        # fast-path: already looks like a ticker symbol
        if name.isupper() and name.isalpha() and 1 <= len(name) <= 5:
            return name

        if not self.api_key:
            return None

        try:
            url = 'https://finnhub.io/api/v1/search'
            params = {'q': name, 'token': self.api_key}
            r = requests.get(url, params=params, timeout=6)
            r.raise_for_status()
            data = r.json()
            results = data.get('result') or []
            if not results:
                return None
            # Prefer exact (case-insensitive) description match, else first symbol
            name_l = name.lower()
            for item in results:
                desc = (item.get('description') or '').lower()
                sym = item.get('symbol')
                if desc and name_l in desc and sym:
                    return sym
            # fallback: first symbol present
            for item in results:
                sym = item.get('symbol')
                if sym:
                    return sym
        except Exception as e:
            print('Symbol resolution error for', name, e)
        return None

    def poll_and_upsert(self, tickers: List[str], window_minutes: int = 60):
        """Poll recent company news for the tickers (window_minutes back) and upsert into Milvus.

        Stores metadata in SQLite and inserts embeddings into `financial_docs` collection.
        """
        to_dt = datetime.date.today()
        from_dt = to_dt - datetime.timedelta(minutes=window_minutes)
        from_s = from_dt.strftime('%Y-%m-%d')
        to_s = to_dt.strftime('%Y-%m-%d')

        new_articles = []
        for t in tickers:
            # Normalize/resolve the ingest target to a ticker symbol when possible
            symbol = None
            if isinstance(t, str) and t:
                # if token looks like ticker, use directly
                if t.isupper() and t.isalpha() and 1 <= len(t) <= 5:
                    symbol = t
                else:
                    symbol = self._resolve_symbol(t) or t.upper()
            else:
                symbol = None

            if not symbol:
                print('Skipping empty/unknown ingest target:', t)
                continue

            try:
                articles = self.fetch_company_news(symbol, from_s, to_s)
            except Exception as e:
                print('Finnhub fetch error for', symbol, 'from target', t, e)
                articles = []

            for a in articles:
                url = a.get('url') or a.get('summary')[:200]
                aid = self._article_id(symbol, url)
                if not self._article_exists(aid):
                    title = a.get('headline') or a.get('summary', '')[:200]
                    summary = a.get('summary') or a.get('headline', '')
                    published_raw = a.get('datetime') or a.get('published') or ''
                    published = self._iso_ts(published_raw)
                    # compute milvus int id in same way used when inserting
                    int_id = int(int(aid, 16) % (2 ** 63 - 1))
                    self._save_meta(aid, symbol, url, title, summary, published, int_id)
                    new_articles.append({'id': aid, 'ticker': symbol, 'title': title, 'summary': summary, 'url': url})

        if not new_articles:
            print('No new articles found')
            return []

        # embed all new articles in a batch (if embedder present)
        vectors = []
        ids = []
        for art in new_articles:
            text = art['title'] + '\n' + art['summary']
            if self.embedder:
                v = self.embedder.embed(text)
            else:
                # deterministic fallback: sha256 -> floats
                import hashlib
                h = hashlib.sha256(text.encode()).digest()
                v = [float(b) / 255.0 for b in h[:64]]
            vectors.append(list(map(float, v)))
            # convert id to integer keyspace using hash mod
            int_id = int(int(art['id'], 16) % (2 ** 63 - 1))
            ids.append(int_id)

        # insert into Milvus collection (matching schema [id, vector])
        print(f'Inserting {len(ids)} vectors to Milvus')
        col = self.col
        res = col.insert([ids, vectors])
        try:
            col.flush()
        except Exception:
            pass
        print('Insert result:', res)
        return new_articles

    def _article_exists(self, aid: str) -> bool:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('SELECT 1 FROM articles WHERE id=?', (aid,))
        r = cur.fetchone()
        conn.close()
        return bool(r)

    def _save_meta(self, aid: str, ticker: str, url: str, title: str, summary: str, published: str, milvus_id: int | None = None):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('INSERT OR REPLACE INTO articles (id,ticker,url,title,summary,published,milvus_id) VALUES (?,?,?,?,?,?,?)',
                    (aid, ticker, url, title, summary, published, milvus_id))
        conn.commit()
        conn.close()


class FinnhubMarketAgent:
    """Template websocket market agent. This is a lightweight template; full implementation requires an API key.

    It uses the Finnhub websocket (wss://ws.finnhub.io) and listens to trade messages.
    """
    WS_URL = 'wss://ws.finnhub.io?token={key}'

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('FINNHUB_API_KEY')

    def run_ws_template(self):
        """Run a simple websocket client that prints messages (template only)."""
        if not self.api_key:
            print('No API key provided for Finnhub websocket; aborting')
            return
        try:
            import websocket

            def on_message(ws, message):
                print('ws msg:', message)

            def on_error(ws, err):
                print('ws err:', err)

            def on_close(ws, code, reason):
                print('ws closed', code, reason)

            def on_open(ws):
                print('ws open; subscribing to AAPL')
                ws.send('{"type":"subscribe","symbol":"AAPL"}')

            ws = websocket.WebSocketApp(self.WS_URL.format(key=self.api_key),
                                        on_message=on_message, on_error=on_error, on_close=on_close)
            ws.on_open = on_open
            ws.run_forever()
        except Exception as e:
            print('websocket dependency missing or error:', e)
