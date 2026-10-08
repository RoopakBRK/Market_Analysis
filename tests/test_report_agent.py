"""Offline tests for ReportAgent: the LLM is mocked."""

import datetime
import json

from langchain_core.messages import AIMessage

from src.agents.report_agent import ReportAgent
from src.models.company import CompanyNews, NewsArticle
from src.models.financial_data import CompanyFinancials
from src.models.macro import MacroSummary
from src.models.market import MarketData
from src.models.reddit import RedditSignal
from src.models.sentiment import SentimentResult

NOW = "2026-10-08T11:00:00+00:00"


class MockLLM:
    """Replies with `content`; given a list, replies with each item in turn."""

    def __init__(self, content=None, error=None):
        self.replies = content if isinstance(content, list) else None
        self.content = content
        self.error = error
        self.calls = 0

    def invoke(self, messages, *args, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        if self.replies is not None:
            return AIMessage(content=self.replies[min(self.calls, len(self.replies)) - 1])
        return AIMessage(content=self.content)


def _inputs():
    macro = MacroSummary(
        overall_sentiment="Bearish",
        confidence=70,
        summary="Macro summary.",
        key_drivers=["Net FII outflows of INR 6,121.37 crore"],
        market_events=["RBI policy meeting"],
        last_updated=NOW,
        market_data={"crude": {"price": 105.21}},
    )
    sentiment = SentimentResult(
        ticker="TMPV",
        company_name="Tata Motors Passenger Vehicles",
        sentiment="Bearish",
        confidence=70,
        impact="Medium",
        summary="Sentiment summary.",
        positive_drivers=["Launch of Jaguar's £130,000 Type 01 luxury EV"],
        negative_drivers=["Share price decline of 3.53% on the day"],
        articles_analyzed=1,
    )
    news = CompanyNews(
        ticker="TMPV",
        company_name="Tata Motors Passenger Vehicles",
        articles=[NewsArticle(title="Jaguar unveils electric car", summary="", source="Reuters", url="u", published_at="")],
        total_articles=1,
    )
    return {
        "macro_summary": macro,
        "company_news": {"TMPV": news},
        "market_data": {"TMPV": MarketData(ticker="TMPV", current_price=273.0, day_change_percent=-3.53, rsi=29.48)},
        # What the stub financial API and an unconfigured Reddit client return.
        "financial_data": {"TMPV": CompanyFinancials(ticker="TMPV", data_source="FinancialAgentAPI", retrieved_at=NOW)},
        "reddit_signals": {"TMPV": RedditSignal(ticker="TMPV", company_name="Tata Motors Passenger Vehicles", retrieved_at=NOW)},
        "sentiments": {"TMPV": sentiment},
    }


def _agent(llm) -> ReportAgent:
    agent = ReportAgent()
    agent.llm = llm
    return agent


def test_facts_come_from_source_data_not_the_llm():
    # A model that tries to supply its own facts alongside the commentary.
    llm_output = {
        "macro_overview": "Overview.",
        "company_narratives": [{
            "ticker": "TMPV",
            "macro_relevance": "Relevance.",
            "overall_interpretation": "Interpretation.",
            "key_positive_signals": ["Launch of Jaguar's Rs 13,00,000 Type 01 luxury EV"],
            "reddit_community_signal": "Neutral",
        }],
        "final_market_view": "View.",
        "major_catalysts": ["Catalyst"],
        "major_risks": ["Risk"],
        "generated_at": "2026-10-08T12:30:00Z",
        "overall_market_sentiment": "Bullish",
        "macro_summary": {"overall_sentiment": "Bullish"},
    }
    inputs = _inputs()
    report = _agent(MockLLM(json.dumps(llm_output))).run(**inputs)
    company = report.company_intelligence[0]

    # Drivers are copied verbatim from the sentiment result.
    assert company.key_positive_signals == ["Launch of Jaguar's £130,000 Type 01 luxury EV"]
    assert company.key_negative_signals == ["Share price decline of 3.53% on the day"]
    assert company.important_news == ["Jaguar unveils electric car (Reuters)"]

    # Sources that supplied nothing are reported as absent, not guessed.
    assert company.reddit_community_signal is None
    assert company.financial_context is None

    # Market data reaches the report.
    assert "273.00" in company.market_snapshot and "-3.53%" in company.market_snapshot

    # Commentary is the model's.
    assert company.overall_interpretation == "Interpretation."
    assert report.macro_overview == "Overview."
    assert report.major_risks == ["Risk"]

    # Top-level facts are the real ones.
    assert report.overall_market_sentiment == "Bearish"
    assert report.macro_summary.market_data == {"crude": {"price": 105.21}}
    assert report.key_macro_drivers == inputs["macro_summary"].key_drivers
    assert report.top_negative_stocks == [inputs["sentiments"]["TMPV"]]
    assert report.top_positive_stocks == []

    generated = datetime.datetime.fromisoformat(report.generated_at)
    assert abs((datetime.datetime.now(datetime.timezone.utc) - generated).total_seconds()) < 60


def test_report_is_still_built_when_the_llm_fails():
    report = _agent(MockLLM(error=RuntimeError("model unavailable"))).run(**_inputs())
    company = report.company_intelligence[0]

    assert company.sentiment == "Bearish"
    assert company.key_positive_signals == ["Launch of Jaguar's £130,000 Type 01 luxury EV"]
    # With no commentary, fall back to text the upstream agents produced.
    assert company.overall_interpretation == "Sentiment summary."
    assert report.macro_overview == "Macro summary."
    assert report.final_market_view == ""


def test_real_financial_and_reddit_data_are_summarised():
    from src.models.reddit import RedditPost

    inputs = _inputs()
    inputs["financial_data"]["TMPV"] = CompanyFinancials(
        ticker="TMPV", pe_ratio=12.5, upcoming_events=["Q2 results"], retrieved_at=NOW
    )
    inputs["reddit_signals"]["TMPV"] = RedditSignal(
        ticker="TMPV",
        company_name="Tata Motors Passenger Vehicles",
        posts=[RedditPost(title="t", subreddit="s", url="u", published_at=NOW)],
        overall_sentiment="Bullish",
        post_count=1,
        retrieved_at=NOW,
    )
    report = _agent(MockLLM("{}")).run(**inputs)
    company = report.company_intelligence[0]

    assert company.financial_context == "P/E 12.5; Upcoming: Q2 results"
    assert company.reddit_community_signal == "Bullish (1 posts)"


def test_commentary_is_matched_when_the_ticker_is_not_echoed_exactly():
    for label in ["tmpv", "TMPV.NS", "TMPV (Tata Motors Passenger Vehicles)", "Tata Motors Passenger Vehicles"]:
        llm_output = {"company_narratives": [
            {"ticker": "ADANIPORTS", "overall_interpretation": "Other company."},
            {"ticker": label, "overall_interpretation": "Interpretation."},
        ]}
        report = _agent(MockLLM(json.dumps(llm_output))).run(**_inputs())
        assert report.company_intelligence[0].overall_interpretation == "Interpretation.", label


def _two_company_inputs():
    inputs = _inputs()
    inputs["sentiments"]["ADANIPORTS"] = SentimentResult(
        ticker="ADANIPORTS", company_name="Adani Ports", sentiment="Bullish", confidence=90,
        impact="High", summary="Ports summary.", articles_analyzed=0,
    )
    return inputs


def _narratives(*tickers):
    return json.dumps({"company_narratives": [
        {"ticker": t, "overall_interpretation": f"{t} interpretation."} for t in tickers
    ]})


def test_missing_company_commentary_is_requested_again():
    llm = MockLLM([_narratives("ADANIPORTS"), _narratives("TMPV", "ADANIPORTS")])
    report = _agent(llm).run(**_two_company_inputs())

    assert llm.calls == 2
    assert [c.overall_interpretation for c in report.company_intelligence] == [
        "TMPV interpretation.", "ADANIPORTS interpretation.",
    ]


def test_complete_commentary_is_not_requested_again():
    llm = MockLLM(_narratives("TMPV", "ADANIPORTS"))
    _agent(llm).run(**_two_company_inputs())

    assert llm.calls == 1


def test_commentary_still_missing_after_retry_falls_back_to_the_summary():
    llm = MockLLM(_narratives("ADANIPORTS"))
    report = _agent(llm).run(**_two_company_inputs())

    assert llm.calls == 2
    assert report.company_intelligence[0].overall_interpretation == "Sentiment summary."
    assert report.company_intelligence[1].overall_interpretation == "ADANIPORTS interpretation."


# ── Day-over-day changes and historical context ──────────────────────────────

def _extras():
    from src.models.report import ChangeRow

    return {
        "historical_context": {"TMPV": ["Tata Motors Passenger Vehicles (TMPV) on NSE, 06 Jan 2015 (Tuesday). The stock fell 4.3%."]},
        "market_changes": [ChangeRow(item="Nifty 50", previous="22,603.05", current="22,231.80", change="-371.25 (-1.64%)")],
        "run_changes": [ChangeRow(item="TMPV sentiment", previous="Bearish (70%)", current="Bearish (70%)", change="unchanged")],
        "previous_run_date": "2026-10-07",
    }


def test_changes_and_historical_context_reach_the_report_and_the_prompt():
    seen = {}

    class RecordingLLM(MockLLM):
        def invoke(self, messages, *args, **kwargs):
            seen["prompt"] = messages.to_messages()[1].content
            return super().invoke(messages, *args, **kwargs)

    extras = _extras()
    report = _agent(RecordingLLM("{}")).run(**_inputs(), **extras)

    assert report.market_changes == extras["market_changes"]
    assert report.run_changes == extras["run_changes"]
    assert report.previous_run_date == "2026-10-07"
    assert report.company_intelligence[0].historical_context == extras["historical_context"]["TMPV"]

    assert "- Nifty 50: 22,603.05 -> 22,231.80 (-371.25 (-1.64%))" in seen["prompt"]
    assert "Changes since the previous run (2026-10-07):" in seen["prompt"]
    assert "06 Jan 2015 (Tuesday). The stock fell 4.3%." in seen["prompt"]


def test_report_without_changes_or_history_is_unchanged():
    report = _agent(MockLLM("{}")).run(**_inputs())

    assert report.market_changes == [] and report.run_changes == []
    assert report.previous_run_date is None
    assert report.company_intelligence[0].historical_context == []


def test_pdf_renders_with_and_without_the_new_sections(tmp_path):
    from src.services.report_service import build_report_pdf

    for name, extras in (("full.pdf", _extras()), ("first_run.pdf", {**_extras(), "run_changes": [], "previous_run_date": None}), ("plain.pdf", {})):
        report = _agent(MockLLM("{}")).run(**_inputs(), **extras)
        path = build_report_pdf(report, str(tmp_path / name))

        with open(path, "rb") as f:
            assert f.read(5) == b"%PDF-", name
