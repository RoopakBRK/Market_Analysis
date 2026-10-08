"""Offline tests for FactCheckAgent: the checker LLM is mocked."""

import json

from langchain_core.messages import AIMessage

from src.agents.fact_check_agent import FactCheckAgent, split_sentences
from src.models.macro import MacroSummary
from src.models.report import CompanyIntelligence, DailyMarketReport

NOW = "2026-10-08T11:00:00+00:00"
FACTS = "Nomura retains 'Buy' on Adani Ports. TMPV market data: 20-session VWAP Rs 293.01."


class MockLLM:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.prompts = []

    def invoke(self, messages, *args, **kwargs):
        self.prompts.append(messages.to_messages()[1].content)
        if self.error:
            raise self.error
        return AIMessage(content=self.content)


def _flag(*claims) -> str:
    return json.dumps({"unsupported": [{"id": i, "reason": reason} for i, reason in claims]})


def _agent(llm) -> FactCheckAgent:
    agent = FactCheckAgent()
    agent.llm = llm
    return agent


def _report() -> DailyMarketReport:
    macro = MacroSummary(
        overall_sentiment="Bearish", confidence=80, summary="Macro summary.", key_drivers=[], last_updated=NOW
    )
    return DailyMarketReport(
        date="2026-10-08",
        overall_market_sentiment="Bearish",
        overall_confidence=80,
        macro_summary=macro,
        # Units 1-2
        macro_overview="The macro backdrop is bearish. Crude rose 4.7% to USD 104.90.",
        company_intelligence=[
            CompanyIntelligence(
                ticker="TMPV", company_name="Tata Motors Passenger Vehicles", sentiment="Bullish", confidence=90,
                # Unit 3
                macro_relevance="Higher rates may weigh on auto demand.",
                # Units 4-5
                overall_interpretation="Sales momentum is strong. The stock trades below key moving averages.",
            ),
            CompanyIntelligence(
                ticker="ADANIPORTS", company_name="Adani Ports", sentiment="Bullish", confidence=90,
                # Unit 6 (no macro relevance for this one)
                overall_interpretation="Cargo growth supports the bullish view.",
            ),
        ],
        # Unit 7
        final_market_view="Markets remain under pressure.",
        # Units 8-9
        major_catalysts=["Nomura retains Buy rating on both companies", "Adani Ports terminal expansion"],
        # Unit 10
        major_risks=["Weakening INR"],
        generated_at=NOW,
    )


# ── Sentence splitting ───────────────────────────────────────────────────────

def test_split_keeps_numbers_and_abbreviations_whole():
    text = "The stock fell 2.41% to Rs 1,708.00. U.S. markets were weak. RSI is 44.0, a neutral reading!"

    assert split_sentences(text) == [
        "The stock fell 2.41% to Rs 1,708.00.",
        "U.S. markets were weak.",
        "RSI is 44.0, a neutral reading!",
    ]
    assert split_sentences("Adani Ports Ltd. reported growth. Volumes rose.") == [
        "Adani Ports Ltd. reported growth.", "Volumes rose.",
    ]
    assert split_sentences("") == [] and split_sentences(None) == []


# ── Removal ──────────────────────────────────────────────────────────────────

def test_flagged_sentences_are_removed_and_the_rest_is_untouched():
    llm = MockLLM(_flag((5, "No moving averages in the facts, only a VWAP"), (8, "Rating is for Adani Ports only")))
    report = _agent(llm).run(_report(), FACTS)
    tmpv, adani = report.company_intelligence

    assert tmpv.overall_interpretation == "Sales momentum is strong."
    assert report.major_catalysts == ["Adani Ports terminal expansion"]

    # Everything else is exactly as written.
    assert report.macro_overview == "The macro backdrop is bearish. Crude rose 4.7% to USD 104.90."
    assert tmpv.macro_relevance == "Higher rates may weigh on auto demand."
    assert adani.overall_interpretation == "Cargo growth supports the bullish view."
    assert report.final_market_view == "Markets remain under pressure."
    assert report.major_risks == ["Weakening INR"]

    # Each removal is recorded with its reason.
    assert report.removed_claims == [
        "(TMPV: interpretation) The stock trades below key moving averages. -- No moving averages in the facts, only a VWAP",
        "(major catalyst) Nomura retains Buy rating on both companies -- Rating is for Adani Ports only",
    ]


def test_checker_sees_the_facts_and_numbered_labelled_sentences():
    llm = MockLLM(_flag())
    _agent(llm).run(_report(), FACTS)
    prompt = llm.prompts[0]

    assert prompt.startswith(f"FACTS:\n{FACTS}")
    assert "[1] (macro overview) The macro backdrop is bearish." in prompt
    assert "[5] (TMPV: interpretation) The stock trades below key moving averages." in prompt
    assert "[6] (ADANIPORTS: interpretation) Cargo growth supports the bullish view." in prompt
    assert "[8] (major catalyst) Nomura retains Buy rating on both companies" in prompt
    assert "[10] (major risk) Weakening INR" in prompt


def test_nothing_flagged_leaves_the_report_unchanged():
    original = _report()
    report = _agent(MockLLM(_flag())).run(original, FACTS)

    assert report == original
    assert report.removed_claims == []


def test_ids_outside_the_range_are_ignored_and_loose_id_formats_accepted():
    llm = MockLLM(json.dumps({"unsupported": [
        {"id": 99, "reason": "no such sentence"},
        {"id": 0, "reason": "no such sentence"},
        {"id": "[10]", "reason": "written as a string"},
    ]}))
    report = _agent(llm).run(_report(), FACTS)

    assert report.major_risks == []
    assert len(report.removed_claims) == 1


def test_emptied_fields_fall_back_to_upstream_text():
    llm = MockLLM(_flag((1, "x"), (2, "x"), (3, "x"), (4, "x"), (5, "x"), (7, "x")))
    report = _agent(llm).run(_report(), FACTS, fallback_interpretations={"TMPV": "Sentiment summary."})
    tmpv = report.company_intelligence[0]

    assert report.macro_overview == "Macro summary."
    assert tmpv.overall_interpretation == "Sentiment summary."
    assert tmpv.macro_relevance is None
    assert report.final_market_view == ""


# ── Failure handling ─────────────────────────────────────────────────────────

def test_report_is_returned_unchanged_when_the_check_fails():
    original = _report()

    assert _agent(MockLLM(error=RuntimeError("all models failed"))).run(original, FACTS) == original
    assert _agent(MockLLM("I could not find any JSON to return")).run(original, FACTS) == original
    assert _agent(MockLLM('{"unsupported": "none"}')).run(original, FACTS) == original


def test_no_llm_call_when_there_is_no_commentary():
    macro = MacroSummary(overall_sentiment="Neutral", confidence=0, summary="", key_drivers=[], last_updated=NOW)
    empty = DailyMarketReport(
        date="2026-10-08", overall_market_sentiment="Unknown", overall_confidence=0,
        macro_summary=macro, generated_at=NOW,
    )
    llm = MockLLM(_flag())

    assert _agent(llm).run(empty, FACTS) == empty
    assert llm.prompts == []


# ── Verifying "not in the facts" verdicts in code ────────────────────────────

MSC_FACTS = (
    "Macro Data: crude rose to $104.25 per barrel\n"
    '    "MSC\'s $1.4 billion purchase of a 49% stake in Vizhinjam, adding a strategic partner",\n'
    "- TMPV sales rose 15% in September\n"
    "Market data: Price Rs 1,708.00 (-2.41% on the day)\n"
)


def _flag_kinds(*claims) -> str:
    return json.dumps({"unsupported": [{"id": i, "kind": kind, "reason": "r"} for i, kind in claims]})


def _report_with_catalysts(*catalysts) -> DailyMarketReport:
    return _report().model_copy(update={
        "macro_overview": "", "company_intelligence": [], "final_market_view": "",
        "major_catalysts": list(catalysts), "major_risks": [],
    })


def test_figures_are_read_with_their_units():
    from src.agents.fact_check_agent import figures

    assert figures("a 49% stake for $1.4 billion at Rs 1,708.00") == {(49.0, True), (1.4, False), (1708.0, False)}
    assert figures("5.50% and 5.5 %") == {(5.5, True)}
    assert figures("no numbers here") == set()


def test_missing_figure_verdict_is_overturned_when_the_figures_are_in_the_facts():
    claim = "MSC's $1.4 billion acquisition of a 49% stake in Vizhinjam"
    report = _agent(MockLLM(_flag_kinds((1, "not_in_facts")))).run(_report_with_catalysts(claim), MSC_FACTS)

    assert report.major_catalysts == [claim]
    assert report.removed_claims == []


def test_other_verdicts_stand_even_when_the_figures_are_in_the_facts():
    # Right figures, wrong relationship: only the checker can judge that.
    claim = "Adani bought a 49% stake in MSC for $1.4 billion"
    for kind in ("contradicts", "wrong_company", ""):
        report = _agent(MockLLM(_flag_kinds((1, kind)))).run(_report_with_catalysts(claim), MSC_FACTS)
        assert report.major_catalysts == [], kind


def test_missing_figure_verdict_stands_when_the_match_is_not_convincing():
    cases = [
        "RBI policy meeting scheduled for 15 October",            # one short figure; "15%" is not "15"
        "Brent crude at $118 per barrel raising the import bill",  # figure absent from the facts
        "A 49% stake sale as crude rose to $104.25",               # both present, but on different lines
        "The stock is in a strong uptrend",                        # no figures to verify
    ]
    for claim in cases:
        report = _agent(MockLLM(_flag_kinds((1, "not_in_facts")))).run(_report_with_catalysts(claim), MSC_FACTS)
        assert report.major_catalysts == [], claim


def test_one_distinctive_figure_is_enough_to_overturn():
    claim = "Crude above $104.25 per barrel"
    report = _agent(MockLLM(_flag_kinds((1, "not_in_facts")))).run(_report_with_catalysts(claim), MSC_FACTS)

    assert report.major_catalysts == [claim]
