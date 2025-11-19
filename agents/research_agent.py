"""
Research Agent: Conducts parallel web searches for company news and analysis.
Stub implementation for future integration with web search APIs or scraping tools.
"""

import requests

class ResearchAgent:
    def __init__(self, api_key: str = "YOUR_FINNHUB_API_KEY"):
        self.api_key = api_key

    def run(self, query: str, companies: list[str]) -> list[str]:
        results = []
        for company in companies:
            try:
                url = f"https://finnhub.io/api/v1/company-news?symbol={company}&from=2025-01-01&to=2025-12-31&token={self.api_key}"
                resp = requests.get(url)
                if resp.status_code == 200:
                    news_items = resp.json()
                    for item in news_items[:3]:
                        results.append(f"{company}: {item.get('headline', '')} ({item.get('datetime', '')})")
                else:
                    results.append(f"{company}: No news found or API error.")
            except Exception as e:
                results.append(f"{company}: Error fetching news: {e}")
        return results
