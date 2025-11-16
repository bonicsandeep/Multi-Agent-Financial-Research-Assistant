"""Rule-based orchestrator for coordinating the financial research agents.

The system works offline by relying on a curated knowledge base that mimics the
outputs of the Research, Data, RAG, Analyst, and Reviewer agents described in
the product requirements. The orchestrator produces a structured response that
follows the mandated format, includes explicit citations, and documents the
coordination plan for each agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from agents.analyst import Analyst

# ---------------------------------------------------------------------------
# Static knowledge base that emulates downstream agents and data sources.
# ---------------------------------------------------------------------------

SOURCE_LIBRARY: Dict[str, str] = {
    "nvda_10k_2024": "Company 10-K filed 2024-03-01 (Nvidia FY24 Form 10-K)",
    "nvda_yfinance_2024_05": "yFinance real-time data: NVDA valuation snapshot - 2024-05-30",
    "nvda_alpha_2024_05": "Alpha Vantage technicals: NVDA 50-day vs 200-day trend - 2024-05-30",
    "nvda_finnhub_2024_05": "Finnhub news: Nvidia raises FY guidance on AI demand - 2024-05-23",
    "nvda_research_2024_05": "Research report from Morgan Stanley - 2024-05-24",
    "nvda_transcript_2024_05": "Earnings call transcript: Nvidia Q1 FY2025 - 2024-05-22",
    "amd_10k_2024": "Company 10-K filed 2024-02-27 (AMD FY23 Form 10-K)",
    "amd_yfinance_2024_05": "yFinance real-time data: AMD valuation snapshot - 2024-05-30",
    "amd_alpha_2024_05": "Alpha Vantage technicals: AMD 50-day vs 200-day trend - 2024-05-30",
    "amd_finnhub_2024_05": "Finnhub news: AMD wins MI300X orders from Microsoft and Meta - 2024-05-21",
    "amd_research_2024_05": "Research report from Goldman Sachs - 2024-05-26",
    "amd_transcript_2024_05": "Earnings call transcript: AMD Q1 2024 - 2024-04-30",
}

COMPANY_PROFILES: Dict[str, Dict[str, object]] = {
    "Nvidia": {
        "ticker": "NVDA",
        "sector": "Semiconductors",
        "aliases": ["nvidia", "nvda"],
        "financials": {
            "revenue": 60.9,  # trailing twelve month revenue, billions USD
            "revenue_growth": 126,  # %
            "gross_margin": 67,
            "fcf_margin": 31,
            "pe": 54,
            "debt_to_equity": 0.42,
            "net_cash": 18.0,
            "source": "nvda_yfinance_2024_05",
        },
        "technical": {
            "signal": "Shares trade 22% above the 200-day average after consecutive breakouts.",
            "rsi": 64,
            "source": "nvda_alpha_2024_05",
        },
        "news": [
            {
                "summary": "Raised FY guidance after securing incremental hyperscale GPU commitments from AWS and Meta.",
                "source": "nvda_finnhub_2024_05",
            },
            {
                "summary": "Morgan Stanley reiterated Overweight citing CUDA software lock-in and networking pull-through.",
                "source": "nvda_research_2024_05",
            },
        ],
        "risks": [
            {
                "item": "U.S. export controls could restrict shipments of H100-class GPUs to China and Middle East customers.",
                "source": "nvda_10k_2024",
            },
            {
                "item": "Top three hyperscale clients represent more than 40% of revenue, keeping concentration risk elevated.",
                "source": "nvda_10k_2024",
            },
        ],
        "opportunities": [
            {
                "item": "Grace Hopper and DGX Cloud subscriptions expand recurring software and services revenue.",
                "source": "nvda_10k_2024",
            }
        ],
        "management": "Management guided data-center revenue to grow >150% YoY and highlighted accelerated networking backlog.",
        "management_source": "nvda_transcript_2024_05",
        "strategy": "Maintains training leadership with CUDA, Spectrum-X networking, and NVLink ecosystem advantages.",
        "strategy_source": "nvda_research_2024_05",
    },
    "AMD": {
        "ticker": "AMD",
        "sector": "Semiconductors",
        "aliases": ["amd", "advanced micro devices"],
        "financials": {
            "revenue": 23.0,
            "revenue_growth": 17,
            "gross_margin": 51,
            "fcf_margin": 12,
            "pe": 38,
            "debt_to_equity": 0.05,
            "net_cash": 3.2,
            "source": "amd_yfinance_2024_05",
        },
        "technical": {
            "signal": "Trading 8% above the 200-day average but consolidating below the March highs.",
            "rsi": 58,
            "source": "amd_alpha_2024_05",
        },
        "news": [
            {
                "summary": "MI300X design wins with Microsoft and Meta add multi-billion dollar datacenter pipeline.",
                "source": "amd_finnhub_2024_05",
            },
            {
                "summary": "Goldman Sachs highlighted upside from MI400 roadmap yet noted slower PC recovery.",
                "source": "amd_research_2024_05",
            },
        ],
        "risks": [
            {
                "item": "Advanced packaging capacity at TSMC remains tight, limiting MI300 supply during 2024.",
                "source": "amd_transcript_2024_05",
            },
            {
                "item": "Gross margin leverage depends on mix shift away from lower-margin client CPUs.",
                "source": "amd_10k_2024",
            },
        ],
        "opportunities": [
            {
                "item": "MI300X and MI400 product cycles expand data-center footprint with ROCm software adoption.",
                "source": "amd_research_2024_05",
            }
        ],
        "management": "Management expects MI300 revenue to exceed $4B in 2024 and is investing in MI400 tape-outs.",
        "management_source": "amd_transcript_2024_05",
        "strategy": "Targets inference share gains with open ROCm ecosystem and partnerships across hyperscalers.",
        "strategy_source": "amd_research_2024_05",
    },
}

METRIC_KEYWORDS: Dict[str, Sequence[str]] = {
    "growth": ("growth", "revenue", "sales", "top line"),
    "profitability": ("margin", "profitability", "gross margin"),
    "valuation": ("valuation", "pe", "p/e", "multiple"),
    "cash flow": ("cash", "cash flow", "fcf", "free cash"),
    "debt": ("debt", "leverage", "balance sheet"),
    "technical": ("technical", "chart", "momentum", "trend"),
    "sentiment": ("sentiment", "news", "analyst"),
}

TIMEFRAME_KEYWORDS: Dict[str, Sequence[str]] = {
    "Short-term (0-3 months)": ("short-term", "swing", "weeks", "near term"),
    "Medium-term (3-12 months)": ("next year", "12 months", "medium-term"),
    "Long-term (3+ years)": ("long-term", "multi-year", "5 year", "decade"),
}

RISK_KEYWORDS: Dict[str, Sequence[str]] = {
    "Conservative": ("capital preservation", "low risk", "conservative"),
    "Balanced": ("moderate risk", "balanced", "medium risk"),
    "Aggressive": ("high risk", "aggressive", "speculative"),
}


@dataclass
class CitationRegistry:
    """Tracks unique citations and returns numbered references."""

    library: Dict[str, str]

    def __post_init__(self) -> None:
        self._order: List[str] = []
        self._index: Dict[str, int] = {}

    def cite(self, source_id: str) -> str:
        if source_id not in self.library:
            raise KeyError(f"Unknown source id: {source_id}")
        if source_id not in self._index:
            self._order.append(source_id)
            self._index[source_id] = len(self._order)
        return f"[{self._index[source_id]}]"

    def render(self) -> List[str]:
        return [f"[{self._index[src]}] {self.library[src]}" for src in self._order]


class Orchestrator:
    """Coordinates the financial research workflow end-to-end."""

    def __init__(self, analyst: Analyst | None = None) -> None:
        self.analyst = analyst or Analyst()
        self.registry = CitationRegistry(SOURCE_LIBRARY)

    # ------------------------------------------------------------------ #
    # Query understanding helpers
    # ------------------------------------------------------------------ #
    def clarify_query(self, query: str) -> Dict[str, object]:
        normalized = query.lower()
        entities = self._extract_entities(normalized)
        if not entities:
            entities = ["Nvidia"]
        timeframe = self._extract_timeframe(query)
        depth = self._extract_depth(normalized)
        risk = self._extract_risk(normalized)
        metrics = self._extract_metrics(normalized)
        return {
            "core_question": query.strip(),
            "entities": entities,
            "timeframe": timeframe,
            "depth": depth,
            "risk_tolerance": risk,
            "metrics": metrics,
        }

    def _extract_entities(self, normalized_query: str) -> List[str]:
        alias_map = {}
        for name, profile in COMPANY_PROFILES.items():
            for alias in profile.get("aliases", []):
                alias_map[alias.lower()] = name
            alias_map[name.lower()] = name
            alias_map[profile["ticker"].lower()] = name  # type: ignore[index]
        entities: List[str] = []
        for alias, canonical in alias_map.items():
            if alias in normalized_query and canonical not in entities:
                entities.append(canonical)
        return entities[:2]  # Limit to pairwise comparisons for clarity

    def _extract_timeframe(self, query: str) -> str:
        normalized = query.lower()
        for label, keywords in TIMEFRAME_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                return label
        match = re.search(r"(\d+)\s*(year|yr|month|mo|week|wk)", normalized)
        if match:
            value = match.group(1)
            unit = match.group(2)
            unit_label = "year" if unit.startswith("y") else "month" if unit.startswith("m") else "week"
            return f"{value}-{unit_label} horizon"
        return "Standard (12-24 month) horizon"

    def _extract_depth(self, normalized_query: str) -> str:
        if "deep dive" in normalized_query or "scenario" in normalized_query:
            return "Deep Dive"
        if "quick" in normalized_query or "summary" in normalized_query:
            return "Quick"
        return "Standard"

    def _extract_risk(self, normalized_query: str) -> str:
        for label, keywords in RISK_KEYWORDS.items():
            if any(keyword in normalized_query for keyword in keywords):
                return label
        return "Balanced"

    def _extract_metrics(self, normalized_query: str) -> List[str]:
        metrics = []
        for label, keywords in METRIC_KEYWORDS.items():
            if any(keyword in normalized_query for keyword in keywords):
                metrics.append(label)
        if not metrics:
            metrics = ["growth", "profitability", "valuation", "sentiment"]
        return metrics

    # ------------------------------------------------------------------ #
    # Agent coordination and investigation
    # ------------------------------------------------------------------ #
    def agent_coordination_plan(self, clarification: Dict[str, object]) -> Dict[str, object]:
        entities = ", ".join(clarification["entities"])  # type: ignore[index]
        timeframe = clarification["timeframe"]
        risk = clarification["risk_tolerance"]
        metrics = ", ".join(clarification["metrics"])  # type: ignore[index]
        assignments = [
            {
                "agent": "Data Agent",
                "task": f"Retrieve {metrics} metrics for {entities} aligned to the {timeframe} view.",
            },
            {
                "agent": "Research Agent",
                "task": f"Scan Finnhub, news, and analyst notes for sentiment shifts impacting {entities}.",
            },
            {
                "agent": "RAG Agent",
                "task": f"Query SEC filings and MD&A passages to extract risk factors and opportunities for {entities}.",
            },
            {
                "agent": "Analyst Agent",
                "task": f"Synthesize comparative thesis tailored to a {risk} mandate.",
            },
            {
                "agent": "Reviewer Agent",
                "task": "Validate completeness, citations, and contradiction checks before release.",
            },
        ]
        execution = [
            "Phase 1 (parallel): Data Agent, Research Agent, and RAG Agent operate simultaneously because they only depend on clarified entities.",
            "Phase 2 (sequential): Analyst Agent waits for upstream results to build the thesis and risk assessment.",
            "Phase 3 (quality): Reviewer Agent confirms citations, ensures risk disclosure, and flags conflicting datapoints.",
        ]
        quality_gates = [
            "Every quantitative claim cites an authoritative source (yFinance, Finnhub, Alpha Vantage, SEC).",
            "Risk factors from the latest 10-K or MD&A must be summarized explicitly.",
            "Conflicting metrics are called out with guidance on which source is newer or more reliable.",
        ]
        expected_sources = {
            "Data Agent": "yFinance fundamentals, Alpha Vantage technical indicators.",
            "Research Agent": "Finnhub news feed, sell-side research digests.",
            "RAG Agent": "SEC EDGAR filings, vectorized MD&A/risk factor database.",
            "Analyst Agent": "Internal reasoning models referencing upstream payloads.",
            "Reviewer Agent": "Rule-based checklist for citations and coverage.",
        }
        return {
            "assignments": assignments,
            "execution": execution,
            "quality_gates": quality_gates,
            "sources": expected_sources,
        }

    def investigate(self, clarification: Dict[str, object], plan: Dict[str, object]) -> Dict[str, object]:
        entities: List[str] = clarification["entities"]  # type: ignore[assignment]
        financial_data = self._collect_financial_data(entities)
        news_data = self._collect_news_data(entities)
        doc_data = self._collect_document_data(entities)

        financial_snapshot = self._format_financial_snapshot(
            financial_data, clarification["metrics"]  # type: ignore[arg-type]
        )
        market_context = self._format_market_context(news_data)
        document_insights = self._format_document_insights(doc_data)
        competitive_positioning = self._format_competitive_positioning(financial_data, doc_data)

        self._review_sections(
            [
                ("Financial Snapshot", financial_snapshot),
                ("Market Context", market_context),
                ("Document Insights", document_insights),
                ("Competitive Positioning", competitive_positioning),
            ]
        )

        return {
            "financial_snapshot": financial_snapshot,
            "market_context": market_context,
            "document_insights": document_insights,
            "competitive_positioning": competitive_positioning,
            "raw": {
                "financial_data": financial_data,
                "news_data": news_data,
                "doc_data": doc_data,
            },
        }

    # ------------------------------------------------------------------ #
    # Formatting helpers
    # ------------------------------------------------------------------ #
    def _collect_financial_data(self, entities: List[str]) -> List[Dict[str, object]]:
        data = []
        for entity in entities:
            profile = COMPANY_PROFILES.get(entity)
            if profile:
                data.append(
                    {
                        "name": entity,
                        "ticker": profile["ticker"],
                        "financials": profile["financials"],
                        "technical": profile["technical"],
                    }
                )
        return data

    def _collect_news_data(self, entities: List[str]) -> List[Dict[str, str]]:
        news = []
        for entity in entities:
            profile = COMPANY_PROFILES.get(entity)
            if profile:
                for item in profile["news"]:
                    news.append({"company": entity, **item})
        return news

    def _collect_document_data(self, entities: List[str]) -> List[Dict[str, object]]:
        docs = []
        for entity in entities:
            profile = COMPANY_PROFILES.get(entity)
            if profile:
                docs.append(
                    {
                        "name": entity,
                        "risks": profile["risks"],
                        "opportunities": profile["opportunities"],
                        "management": profile["management"],
                        "management_source": profile["management_source"],
                        "strategy": profile["strategy"],
                        "strategy_source": profile["strategy_source"],
                    }
                )
        return docs

    def _format_financial_snapshot(self, data: List[Dict[str, object]], metrics: List[str]) -> str:
        focus = ", ".join(metrics)
        lines = [f"Focus metrics: {focus}."]
        for entry in data:
            financials = entry["financials"]  # type: ignore[assignment]
            tech = entry["technical"]  # type: ignore[assignment]
            fin_cite = self.registry.cite(financials["source"])  # type: ignore[index]
            tech_cite = self.registry.cite(tech["source"])  # type: ignore[index]
            lines.append(
                (
                    f"{entry['name']} ({entry['ticker']}) prints ${financials['revenue']:.1f}B TTM revenue with "
                    f"{financials['revenue_growth']}% YoY growth, {financials['gross_margin']}% gross margin, "
                    f"{financials['fcf_margin']}% FCF margin, and trades at {financials['pe']}x forward earnings "
                    f"with debt-to-equity of {financials['debt_to_equity']:.2f} and net cash of "
                    f"${financials['net_cash']:.1f}B {fin_cite}. "
                    f"Technical posture: {tech['signal']} (RSI {tech['rsi']}) {tech_cite}."
                )
            )
        return " ".join(lines)

    def _format_market_context(self, news_data: List[Dict[str, str]]) -> str:
        if not news_data:
            return "Research Agent found no recent news requiring action."
        snippets = []
        for item in news_data:
            cite = self.registry.cite(item["source"])
            snippets.append(f"{item['company']}: {item['summary']} {cite}.")
        return " ".join(snippets)

    def _format_document_insights(self, doc_data: List[Dict[str, object]]) -> str:
        lines = []
        for entry in doc_data:
            risk_bits = [
                f"{risk['item']} {self.registry.cite(risk['source'])}"
                for risk in entry["risks"]  # type: ignore[index]
            ]
            opportunity_bits = [
                f"{op['item']} {self.registry.cite(op['source'])}"
                for op in entry["opportunities"]  # type: ignore[index]
            ]
            management_cite = self.registry.cite(entry["management_source"])  # type: ignore[index]
            lines.append(
                (
                    f"{entry['name']} risks: {', '.join(risk_bits)}. "
                    f"Opportunities: {', '.join(opportunity_bits)}. "
                    f"Management commentary: {entry['management']} {management_cite}."
                )
            )
        return " ".join(lines)

    def _format_competitive_positioning(
        self, financial_data: List[Dict[str, object]], doc_data: List[Dict[str, object]]
    ) -> str:
        if len(financial_data) < 2:
            entry = financial_data[0]
            cite = self.registry.cite(entry["financials"]["source"])  # type: ignore[index]
            return f"Single-company view: {entry['name']} profile summarized above {cite}."
        first, second = financial_data[:2]
        first_fin = first["financials"]  # type: ignore[assignment]
        second_fin = second["financials"]  # type: ignore[assignment]
        first_doc = doc_data[0]
        second_doc = doc_data[1]
        first_strategy_cite = self.registry.cite(first_doc["strategy_source"])  # type: ignore[index]
        second_strategy_cite = self.registry.cite(second_doc["strategy_source"])  # type: ignore[index]
        first_fin_cite = self.registry.cite(first_fin["source"])  # type: ignore[index]
        second_fin_cite = self.registry.cite(second_fin["source"])  # type: ignore[index]
        comparison = (
            f"{first['name']} retains scale and {first_fin['gross_margin']}% gross margins versus "
            f"{second['name']}'s {second_fin['gross_margin']}%, supporting a higher {first_fin['pe']}x vs "
            f"{second_fin['pe']}x multiple {first_fin_cite}{second_fin_cite}. "
            f"{first['name']} benefits from CUDA/networking moat {first_strategy_cite}, while "
            f"{second['name']} closes the gap with MI300/MI400 roadmaps and open ROCm partnerships {second_strategy_cite}."
        )
        return comparison

    def _review_sections(self, sections: List[tuple[str, str]]) -> None:
        """Simple reviewer that ensures every section has a citation and risk coverage."""
        for name, text in sections:
            if "[" not in text:
                raise ValueError(f"Reviewer Agent flagged missing citation in {name}")
        if "risk" not in sections[2][1].lower():
            raise ValueError("Reviewer Agent requires explicit risk disclosure.")

    # ------------------------------------------------------------------ #
    # Recommendation synthesis
    # ------------------------------------------------------------------ #
    def synthesize_recommendation(self, results: Dict[str, object], clarification: Dict[str, object]) -> Dict[str, str]:
        financial_data = results["raw"]["financial_data"]  # type: ignore[index]
        doc_data = results["raw"]["doc_data"]  # type: ignore[index]
        news_data = results["raw"]["news_data"]  # type: ignore[index]

        nvda = financial_data[0]
        amd = financial_data[1] if len(financial_data) > 1 else None
        nvda_fin_cite = self.registry.cite(nvda["financials"]["source"])  # type: ignore[index]
        nvda_strategy_cite = self.registry.cite(doc_data[0]["strategy_source"])  # type: ignore[index]
        nvda_risk_cite = self.registry.cite(doc_data[0]["risks"][0]["source"])  # type: ignore[index]

        if amd:
            amd_fin_cite = self.registry.cite(amd["financials"]["source"])  # type: ignore[index]
            amd_opportunity_cite = self.registry.cite(doc_data[1]["opportunities"][0]["source"])  # type: ignore[index]
            amd_risk_cite = self.registry.cite(doc_data[1]["risks"][0]["source"])  # type: ignore[index]
        else:
            amd_fin_cite = amd_opportunity_cite = amd_risk_cite = ""

        catalysts = ", ".join(
            [
                f"{item['company']} – {item['summary']} {self.registry.cite(item['source'])}"
                for item in news_data
            ]
        )

        thesis_parts = [
            f"{nvda['name']} maintains AI training leadership with superior margins and CUDA lock-in {nvda_fin_cite}{nvda_strategy_cite}",
        ]
        if amd:
            thesis_parts.append(
                f"{amd['name']} offers diversification through MI300 acceleration and trades at a lower multiple {amd_fin_cite}{amd_opportunity_cite}"
            )
        thesis = "; ".join(thesis_parts) + "."

        risk_parts = [
            f"Upside is high because hyperscale capex remains supportive, but export controls and supply bottlenecks could derail execution {nvda_risk_cite}"
        ]
        if amd:
            risk_parts.append(f"AMD still depends on TSMC advanced packaging availability, adding operational risk {amd_risk_cite}")
        risk_reward = " ".join(risk_parts)

        recommendation = {
            "thesis": thesis,
            "risk_reward": risk_reward,
            "catalysts": catalysts,
            "disclaimer": (
                "This is for educational analysis only and not investment advice. Consult a financial advisor before investing. "
                "Past performance does not guarantee future results."
            ),
        }
        return recommendation

    # ------------------------------------------------------------------ #
    # Response composition
    # ------------------------------------------------------------------ #
    def orchestrate(self, query: str) -> str:
        self.registry = CitationRegistry(SOURCE_LIBRARY)
        clarification = self.clarify_query(query)
        plan = self.agent_coordination_plan(clarification)
        results = self.investigate(clarification, plan)
        recommendation = self.synthesize_recommendation(results, clarification)
        sources = self.registry.render()
        response = self._format_response(clarification, plan, results, recommendation, sources)
        return response.strip()

    def _format_response(
        self,
        clarification: Dict[str, object],
        plan: Dict[str, object],
        results: Dict[str, object],
        recommendation: Dict[str, str],
        sources: List[str],
    ) -> str:
        assignment_lines = "\n".join(
            f"  - {item['agent']} – {item['task']}" for item in plan["assignments"]  # type: ignore[index]
        )
        execution_lines = "\n".join(f"  - {line}" for line in plan["execution"])  # type: ignore[index]
        quality_lines = "\n".join(f"  - {line}" for line in plan["quality_gates"])  # type: ignore[index]
        source_lines = "\n".join(
            f"  - {agent}: {desc}" for agent, desc in plan["sources"].items()  # type: ignore[index]
        )
        citation_lines = "\n".join(f"- {item}" for item in sources)
        metrics = ", ".join(clarification["metrics"])  # type: ignore[index]
        entities = ", ".join(clarification["entities"])  # type: ignore[index]
        response = f"""
### QUERY UNDERSTANDING
- User's core question: {clarification['core_question']}
- Key entities identified: {entities}
- Time horizon: {clarification['timeframe']}
- Risk tolerance: {clarification['risk_tolerance']}
- Metrics of interest: {metrics}
- Analysis depth requested: {clarification['depth']}

### AGENT COORDINATION PLAN
- Agent assignments:
{assignment_lines}
- Execution sequence:
{execution_lines}
- Quality gates:
{quality_lines}
- Expected data sources:
{source_lines}

### INVESTIGATION RESULTS
- Financial Snapshot: {results['financial_snapshot']}
- Market Context: {results['market_context']}
- Document Insights: {results['document_insights']}
- Competitive Positioning: {results['competitive_positioning']}

### INVESTMENT RECOMMENDATION
- Final thesis: {recommendation['thesis']}
- Risk/Reward assessment: {recommendation['risk_reward']}
- Key catalysts to monitor: {recommendation['catalysts']}
- Disclaimer: {recommendation['disclaimer']}

### SOURCE CITATIONS
{citation_lines}
"""
        return response
