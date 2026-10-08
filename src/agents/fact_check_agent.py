import json
import re
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate

from src.llm.gateway import get_checker_llm
from src.llm.structured_output import build_json_instruction, unwrap_schema_echo
from src.models.fact_check import FactCheckResult
from src.models.report import DailyMarketReport
from src.prompts.fact_check import SYSTEM_PROMPT

# Abbreviations whose full stop does not end a sentence.
_ABBREVIATIONS = ("U.S.", "U.K.", "Ltd.", "Inc.", "Rs.", "vs.", "No.", "e.g.", "i.e.", "approx.")


def split_sentences(text: str) -> list[str]:
    """
    Split a paragraph into sentences. A full stop inside a number ("Rs 1,708.00")
    or after a known abbreviation does not end one. When unsure this keeps text
    together: a merged sentence is checked as one unit, whereas a wrong split
    could leave half a sentence behind after a removal.
    """
    sentences: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", (text or "").strip()):
        if sentences and sentences[-1].endswith(_ABBREVIATIONS):
            sentences[-1] += " " + part
        elif part:
            sentences.append(part)
    return sentences


def figures(text: str) -> set[tuple[float, bool]]:
    """The numbers in a text, as (value, is_percentage): "49% of $1.4 billion" -> {(49, True), (1.4, False)}."""
    found = set()
    for number, percent in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s?(%?)", text):
        try:
            found.add((float(number.replace(",", "")), bool(percent)))
        except ValueError:
            continue
    return found


def figures_appear_together(sentence: str, facts: str) -> bool:
    """
    True if the sentence's figures all appear on a single line of the facts.

    Used to double-check a "not in the facts" verdict: if the figures the
    checker called missing are in fact there, side by side, its premise is
    wrong. To make a chance match implausible this needs either two or more
    figures, or one with at least three digits ("104.25", not "15").
    """
    wanted = figures(sentence)
    digits = sum(len(re.sub(r"\D", "", f"{value:g}")) for value, _ in wanted)
    if not wanted or (len(wanted) < 2 and digits < 3):
        return False
    return any(wanted <= figures(line) for line in facts.splitlines())


class FactCheckAgent:
    """
    Checks the report's LLM-written commentary against the facts the writer
    was given, and removes the sentences those facts do not support.

    The commentary is the only free text in the report; everything else is
    copied in code. This is its safety net: a second model reads the same
    facts and flags statements such as a rating attributed to the wrong
    company or an indicator that was never supplied.

    Flagged sentences are deleted in code, never rewritten, so the check
    cannot introduce new claims. If the check itself fails, the report is
    returned unchanged.

    The checker is itself an LLM and can overlook a fact that is there. So a
    verdict that a sentence's figures are "not in the facts" is verified in
    code first, and the sentence is kept if they demonstrably are.
    """

    def __init__(self):
        # A different model from the one that wrote the commentary, so it is
        # less likely to repeat the same misreading.
        self.llm = get_checker_llm(agent_name="FactCheckAgent")

        json_instruction = build_json_instruction(
            FactCheckResult,
            # The example below shows one entry; without this the model tends
            # to return exactly one, flagging something even when all is well.
            extra_rules='The list may be empty or hold several entries. If every sentence is supported, return {{"unsupported": []}}.',
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT + "\n\n" + json_instruction),
                ("human", "FACTS:\n{facts}\n\nSENTENCES:\n{sentences}"),
            ]
        )

    @staticmethod
    def _units(report: DailyMarketReport) -> list[tuple[tuple, str, str]]:
        """
        Break the commentary into checkable units: (slot, label, text).
        Paragraphs are split into sentences; list items are one unit each.
        """
        units = [(("macro_overview",), "macro overview", s) for s in split_sentences(report.macro_overview)]

        for i, company in enumerate(report.company_intelligence):
            units += [
                (("company", i, "macro_relevance"), f"{company.ticker}: macro relevance", s)
                for s in split_sentences(company.macro_relevance)
            ]
            units += [
                (("company", i, "overall_interpretation"), f"{company.ticker}: interpretation", s)
                for s in split_sentences(company.overall_interpretation)
            ]

        units += [(("final_market_view",), "final market view", s) for s in split_sentences(report.final_market_view)]
        units += [(("major_catalysts",), "major catalyst", item) for item in report.major_catalysts if item.strip()]
        units += [(("major_risks",), "major risk", item) for item in report.major_risks if item.strip()]
        return units

    def _check(self, facts: str, units: list[tuple[tuple, str, str]]) -> dict[int, str]:
        """
        Ask the checker which numbered units are unsupported: {number: reason}.
        Verdicts the facts themselves disprove are dropped.
        """
        sentences = "\n".join(f"[{n}] ({label}) {text}" for n, (_slot, label, text) in enumerate(units, 1))
        response = self.llm.invoke(self.prompt.invoke({"facts": facts, "sentences": sentences}))

        content = str(response.content).strip()
        start = content.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in fact-check response: {content[:200]!r}")
        data, _end = json.JSONDecoder().raw_decode(content, start)
        result = FactCheckResult.model_validate(unwrap_schema_echo(data))

        flagged = {}
        for claim in result.unsupported:
            if not 1 <= claim.id <= len(units):
                continue
            _slot, label, text = units[claim.id - 1]
            if "not_in" in claim.kind.lower() and figures_appear_together(text, facts):
                print(f"[FactCheckAgent] Kept ({label}) {text!r}: called unsupported, but its figures are in the facts.")
                continue
            flagged[claim.id] = claim.reason
        return flagged

    def run(
        self,
        report: DailyMarketReport,
        facts: str,
        fallback_interpretations: dict[str, str] | None = None,
    ) -> DailyMarketReport:
        """
        Return the report with unsupported commentary removed.

        fallback_interpretations: {ticker: text} used when every sentence of a
        company's interpretation is removed (the sentiment summary).
        """
        units = self._units(report)
        if not units:
            return report

        try:
            flagged = self._check(facts, units)
        except Exception as e:
            print(f"[FactCheckAgent] Check unavailable, commentary left as written: {e}")
            return report

        if not flagged:
            return report

        kept = defaultdict(list)
        removed = []
        for number, (slot, label, text) in enumerate(units, 1):
            if number in flagged:
                removed.append(f"({label}) {text} -- {flagged[number] or 'not supported by the facts'}")
            else:
                kept[slot].append(text)

        print(f"[FactCheckAgent] Removed {len(removed)} unsupported statement(s):")
        for line in removed:
            print(f"  - {line}")

        fallbacks = fallback_interpretations or {}
        companies = [
            company.model_copy(update={
                "macro_relevance": " ".join(kept[("company", i, "macro_relevance")]) or None,
                "overall_interpretation": (
                    " ".join(kept[("company", i, "overall_interpretation")]) or fallbacks.get(company.ticker, "")
                ),
            })
            for i, company in enumerate(report.company_intelligence)
        ]

        return report.model_copy(update={
            "macro_overview": " ".join(kept[("macro_overview",)]) or report.macro_summary.summary,
            "company_intelligence": companies,
            "final_market_view": " ".join(kept[("final_market_view",)]),
            "major_catalysts": kept[("major_catalysts",)],
            "major_risks": kept[("major_risks",)],
            "removed_claims": removed,
        })
