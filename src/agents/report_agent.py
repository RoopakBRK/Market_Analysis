from langchain_core.prompts import ChatPromptTemplate
import datetime
import json
import re

from src.llm.gateway import get_llm
from src.llm.structured_output import build_json_instruction, unwrap_schema_echo
from src.models.report import (
    CompanyIntelligence,
    CompanyNarrative,
    DailyMarketReport,
    ReportNarrative,
)
from src.prompts.report import SYSTEM_PROMPT
from src.tools.common.normalization import utc_now_iso

# Headlines per company shown to the LLM and listed in the report.
_MAX_HEADLINES = 5


def _format_market(market) -> str | None:
    """One-line market snapshot, or None if no market data was retrieved."""
    if market is None or market.current_price is None:
        return None

    price = f"Price Rs {market.current_price:,.2f}"
    if market.day_change_percent is not None:
        price += f" ({market.day_change_percent:+.2f}% on the day)"
    parts = [price]

    if market.rsi is not None:
        parts.append(f"RSI(14) {market.rsi:.1f}")
    if market.macd is not None and market.macd_signal is not None:
        parts.append(f"MACD {market.macd:.2f} vs signal {market.macd_signal:.2f}")
    if market.vwap is not None:
        parts.append(f"20-session VWAP Rs {market.vwap:,.2f}")
    if market.sector_index and market.sector_performance is not None:
        parts.append(f"{market.sector_index} {market.sector_performance:+.2f}%")
    return "; ".join(parts)


def _format_financials(fin) -> str | None:
    """
    One-line financial summary, or None if the provider returned no figures.
    (CompanyFinancials always carries a ticker, source and timestamp, so it
    is never "empty" as an object — only its figures can be missing.)
    """
    if fin is None:
        return None

    parts = []
    if fin.market_cap is not None:
        parts.append(f"Market cap {fin.market_cap:,.0f}")
    if fin.pe_ratio is not None:
        parts.append(f"P/E {fin.pe_ratio:.1f}")
    if fin.eps is not None:
        parts.append(f"EPS {fin.eps:.2f}")
    if fin.revenue is not None:
        parts.append(f"Revenue {fin.revenue:,.0f}")
    if fin.debt_to_equity is not None:
        parts.append(f"Debt/equity {fin.debt_to_equity:.2f}")
    if fin.upcoming_events:
        parts.append("Upcoming: " + ", ".join(fin.upcoming_events))
    return "; ".join(parts) or None


def _format_reddit(reddit) -> str | None:
    """Reddit signal summary, or None if no posts were retrieved."""
    if reddit is None or not reddit.posts:
        return None
    return f"{reddit.overall_sentiment} ({reddit.post_count} posts)"


def _headlines(news) -> list[str]:
    if news is None:
        return []
    return [f"{a.title} ({a.source})" for a in news.articles[:_MAX_HEADLINES]]


def _find_narrative(
    narratives: list[CompanyNarrative], ticker: str, company_name: str
) -> CompanyNarrative | None:
    """
    Find the model's commentary for a company. It does not always echo the
    ticker verbatim — "TMPV.NS", "TMPV (Tata Motors Passenger Vehicles)" or
    just the company name all occur — so fall back to looser matches.
    """
    ticker_l, name_l = ticker.lower(), company_name.lower()
    labels = [(n.ticker.strip().lower(), n) for n in narratives]

    for label, narrative in labels:
        if label == ticker_l:
            return narrative
    for label, narrative in labels:
        if ticker_l in re.findall(r"[a-z0-9&-]+", label) or (name_l and name_l in label):
            return narrative
    return None


def build_report_facts(
    macro_summary, company_news, market_data, financial_data, reddit_signals, sentiments,
    historical_context=None, market_changes=None, run_changes=None, previous_run_date=None,
) -> str:
    """
    Everything the commentary may draw on, as text. The ReportAgent writes
    from it and the FactCheckAgent checks the result against it, so both
    must see exactly the same facts.
    """
    historical_context = historical_context or {}

    # Exclude massive raw data to save tokens
    macro_data_clean = macro_summary.model_dump(exclude={"market_data"})

    facts = f"Macro Data:\n{json.dumps(macro_data_clean, indent=2, ensure_ascii=False)}\n"

    if market_changes:
        facts += "\nMoves since the previous close:\n"
        facts += "".join(f"- {r.item}: {r.previous} -> {r.current} ({r.change})\n" for r in market_changes)
    if run_changes:
        facts += f"\nChanges since the previous run ({previous_run_date}):\n"
        facts += "".join(f"- {r.item}: {r.previous} -> {r.current} ({r.change})\n" for r in run_changes)

    for ticker, sentiment in sentiments.items():
        # Only what the commentary needs; the per-source breakdown and
        # bookkeeping fields would just invite the model to echo them.
        sentiment_view = sentiment.model_dump(
            include={"sentiment", "confidence", "impact", "summary", "positive_drivers", "negative_drivers"}
        )
        headlines = _headlines(company_news.get(ticker))

        facts += f"\n=== {ticker} ({sentiment.company_name}) ===\n"
        facts += f"Sentiment analysis:\n{json.dumps(sentiment_view, indent=2, ensure_ascii=False)}\n"
        facts += "Recent headlines:\n"
        facts += "".join(f"- {h}\n" for h in headlines) if headlines else "- none retrieved\n"
        facts += f"Market data: {_format_market(market_data.get(ticker)) or 'unavailable'}\n"
        facts += f"Financial data: {_format_financials(financial_data.get(ticker)) or 'unavailable'}\n"
        facts += f"Reddit/community: {_format_reddit(reddit_signals.get(ticker)) or 'unavailable'}\n"
        passages = historical_context.get(ticker) or []
        if passages:
            facts += "Historical context (from 20 years of price history; prices adjusted for splits and dividends):\n"
            facts += "".join(f"- {passage}\n" for passage in passages)

    # Upstream LLM text uses typographic spaces ("$1.4\u202fbillion"); plain
    # spaces read the same and keep figures easy to match.
    return facts.replace("\u202f", " ").replace("\u2009", " ").replace("\u00a0", " ")


class ReportAgent:
    def __init__(self):
        self.llm = get_llm(agent_name="ReportAgent")

        json_instruction = build_json_instruction(ReportNarrative)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT + "\n\n" + json_instruction),
                ("human", "{input}")
            ]
        )

    def run(
        self,
        macro_summary,
        company_news,
        market_data,
        financial_data,
        reddit_signals,
        sentiments,
        historical_context=None,
        market_changes=None,
        run_changes=None,
        previous_run_date=None,
    ) -> DailyMarketReport:

        historical_context = historical_context or {}
        market_changes = market_changes or []
        run_changes = run_changes or []

        input_text = self._build_input(
            macro_summary, company_news, market_data, financial_data, reddit_signals, sentiments,
            historical_context, market_changes, run_changes, previous_run_date,
        )

        # The commentary is the only LLM-dependent part. If it fails, the
        # report is still assembled from the collected facts.
        try:
            narrative = self._write_narrative(input_text)
        except Exception as e:
            print(f"[ReportAgent] Commentary unavailable, building a facts-only report: {e}")
            narrative = ReportNarrative()
        else:
            self._fill_missing_narratives(narrative, input_text, sentiments)

        company_intelligence = []
        for ticker, sentiment in sentiments.items():
            note = _find_narrative(
                narrative.company_narratives, ticker, sentiment.company_name
            ) or CompanyNarrative(ticker=ticker)
            company_intelligence.append(
                CompanyIntelligence(
                    ticker=ticker,
                    company_name=sentiment.company_name,
                    sentiment=sentiment.sentiment,
                    confidence=sentiment.confidence,
                    key_positive_signals=list(sentiment.positive_drivers),
                    key_negative_signals=list(sentiment.negative_drivers),
                    important_news=_headlines(company_news.get(ticker)),
                    market_snapshot=_format_market(market_data.get(ticker)),
                    financial_context=_format_financials(financial_data.get(ticker)),
                    reddit_community_signal=_format_reddit(reddit_signals.get(ticker)),
                    macro_relevance=note.macro_relevance or None,
                    historical_context=list(historical_context.get(ticker) or []),
                    overall_interpretation=note.overall_interpretation or sentiment.summary,
                )
            )

        by_confidence = sorted(sentiments.values(), key=lambda s: s.confidence, reverse=True)

        return DailyMarketReport(
            date=datetime.date.today().isoformat(),
            overall_market_sentiment=macro_summary.overall_sentiment,
            overall_confidence=macro_summary.confidence,
            macro_summary=macro_summary,
            macro_overview=narrative.macro_overview or macro_summary.summary,
            key_macro_drivers=list(macro_summary.key_drivers),
            company_intelligence=company_intelligence,
            top_positive_stocks=[s for s in by_confidence if s.sentiment == "Bullish"],
            top_negative_stocks=[s for s in by_confidence if s.sentiment == "Bearish"],
            market_events=list(macro_summary.market_events),
            market_changes=market_changes,
            run_changes=run_changes,
            previous_run_date=previous_run_date,
            final_market_view=narrative.final_market_view,
            major_catalysts=narrative.major_catalysts,
            major_risks=narrative.major_risks,
            generated_at=utc_now_iso(),
        )

    def _build_input(
        self, macro_summary, company_news, market_data, financial_data, reddit_signals, sentiments,
        historical_context, market_changes, run_changes, previous_run_date,
    ) -> str:
        input_text = build_report_facts(
            macro_summary, company_news, market_data, financial_data, reddit_signals, sentiments,
            historical_context, market_changes, run_changes, previous_run_date,
        )

        # The output example shows a single list entry, and the model will
        # sometimes mirror that literally unless told how many are wanted.
        input_text += (
            f"\ncompany_narratives must contain exactly {len(sentiments)} entries, "
            f"one for each of these tickers: {', '.join(sentiments)}.\n"
        )
        return input_text

    def _fill_missing_narratives(self, narrative: ReportNarrative, input_text: str, sentiments) -> None:
        """
        If the model left companies out of its commentary, ask once more and
        take the missing entries from the second answer. Entries already
        present are kept: _find_narrative returns the first match.
        """
        missing = [
            ticker for ticker, sentiment in sentiments.items()
            if _find_narrative(narrative.company_narratives, ticker, sentiment.company_name) is None
        ]
        if not missing:
            return

        print(f"[ReportAgent] Commentary missing for {', '.join(missing)}; asking again.")
        reminder = f"\nYour previous answer left out: {', '.join(missing)}. Include every ticker this time.\n"
        try:
            retry = self._write_narrative(input_text + reminder)
        except Exception as e:
            print(f"[ReportAgent] Retry failed, keeping the partial commentary: {e}")
            return
        narrative.company_narratives.extend(retry.company_narratives)

    def _write_narrative(self, input_text: str) -> ReportNarrative:
        messages = self.prompt.invoke({"input": input_text})
        response = self.llm.invoke(messages)

        content = str(response.content).strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # raw_decode from the first '{' instead of find('{')/rfind('}'):
        # rfind grabs the LAST closing brace in the response, so any
        # trailing text after a valid object corrupts the slice.
        start = content.find('{')
        if start == -1:
            raise ValueError(f"No JSON object found in LLM response: {content[:200]!r}")
        try:
            data, _end = json.JSONDecoder().raw_decode(content, start)
            return ReportNarrative.model_validate(unwrap_schema_echo(data))
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from LLM: {e}\nResponse content: {response.content}")
