"""
Data Agent: Fetches real-time stock data, financials, and technical indicators.
Stub implementation for future integration with yFinance, Alpha Vantage, etc.
"""

import yfinance as yf

class DataAgent:
    def run(self, companies: list[str]) -> dict:
        data = {}
        for company in companies:
            try:
                ticker = yf.Ticker(company)
                info = ticker.info
                data[company] = {
                    "price": info.get("regularMarketPrice", "N/A"),
                    "pe_ratio": info.get("trailingPE", "N/A"),
                    "market_cap": info.get("marketCap", "N/A"),
                    "dividend_yield": info.get("dividendYield", "N/A")
                }
            except Exception as e:
                data[company] = {"error": str(e)}
        return data
