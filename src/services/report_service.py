"""
Renders a DailyMarketReport into a formatted PDF file.

Uses reportlab's platypus flowables (paragraphs/tables/spacers laid out
automatically across pages) rather than manual canvas positioning, since the
number of companies/drivers/news items varies run to run.
"""

from __future__ import annotations

import datetime
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.models.report import CompanyIntelligence, DailyMarketReport

# ── Text sanitization ────────────────────────────────────────────────────────
# Helvetica (reportlab's default base-14 font, used here so no font file needs
# to be embedded) only covers WinAnsi/Latin-1. LLM output routinely contains
# typographic Unicode punctuation (curly quotes, en/em dashes, the non-
# breaking hyphen U+2011) and currency symbols (₹) that fall outside that
# range — reportlab silently renders those as a black "missing glyph" square
# (■) instead of raising, so this has to be fixed proactively, not detected
# via an exception. Map each to its closest ASCII equivalent before layout.
_UNICODE_REPLACEMENTS = {
    "‘": "'", "’": "'",  # curly single quotes
    "“": '"', "”": '"',  # curly double quotes
    "–": "-", "—": "-", "‑": "-",  # en/em dash, non-breaking hyphen
    "…": "...",  # ellipsis
    "₹": "Rs ",  # rupee sign
    " ": " ",  # non-breaking space
    "•": "-",  # bullet (we render our own bullets via ListFlowable)
}


def _sanitize(text: object) -> str:
    """Coerce to str and replace Unicode punctuation Helvetica can't render."""
    if text is None:
        return ""
    s = str(text)
    for char, repl in _UNICODE_REPLACEMENTS.items():
        s = s.replace(char, repl)
    # Escape XML special characters for reportlab's mini-markup in Paragraph,
    # then anything still outside Latin-1 (Helvetica's range) is dropped
    # rather than left to render as a missing-glyph square.
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s.encode("latin-1", errors="ignore").decode("latin-1")


def _format_display_date(date_str: str) -> str:
    """
    DailyMarketReport.date is kept as ISO (YYYY-MM-DD) for sorting/filenames,
    but that reads awkwardly on the page for an Indian-market report.
    Render it as "29 Sep 2026" (day-month-year, as used in India) instead.
    Falls back to the raw string if it isn't a plain ISO date.
    """
    try:
        return datetime.date.fromisoformat(date_str).strftime("%d %b %Y")
    except (ValueError, TypeError):
        return date_str or ""


# ── Sentiment -> color mapping, used for badges and table rows ──────────────
_SENTIMENT_COLORS = {
    "Bullish": colors.HexColor("#1a7f37"),
    "Bearish": colors.HexColor("#c62828"),
    "Neutral": colors.HexColor("#8a6d00"),
    "Unknown": colors.HexColor("#666666"),
}

_BRAND_COLOR = colors.HexColor("#0f2f5f")
_LIGHT_GREY = colors.HexColor("#f2f4f7")
_BORDER_GREY = colors.HexColor("#d0d5dd")


def _sentiment_color(label: str) -> colors.Color:
    return _SENTIMENT_COLORS.get((label or "").strip(), _SENTIMENT_COLORS["Unknown"])


def _build_styles() -> dict:
    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontSize=22, textColor=_BRAND_COLOR,
            spaceAfter=4,
        ),
        "SubTitle": ParagraphStyle(
            "ReportSubTitle", parent=base["Normal"], fontSize=11, textColor=colors.grey,
            alignment=TA_CENTER, spaceAfter=14,
        ),
        "H1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontSize=15, textColor=_BRAND_COLOR,
            spaceBefore=18, spaceAfter=8, borderPadding=0,
        ),
        "H2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontSize=12.5, textColor=colors.HexColor("#111111"),
            spaceBefore=12, spaceAfter=4,
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["Normal"], fontSize=9.5, leading=13.5,
        ),
        "BodySmall": ParagraphStyle(
            "BodySmall", parent=base["Normal"], fontSize=8.5, leading=12, textColor=colors.HexColor("#444444"),
        ),
        "Bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"], fontSize=9.5, leading=13,
        ),
        "Badge": ParagraphStyle(
            "Badge", parent=base["Normal"], fontSize=10, leading=12,
            textColor=colors.white, alignment=TA_CENTER,
        ),
    }
    return styles


def _badge_table(label: str, confidence: int | None, styles: dict) -> Table:
    """A small colored pill showing sentiment (+ confidence, if given)."""
    label = _sanitize(label)
    text = label if confidence is None else f"{label} - {confidence}% confidence"
    t = Table([[Paragraph(text, styles["Badge"])]], colWidths=[None])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _sentiment_color(label)),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    return t


def _bullet_list(items: list[str], styles: dict, empty_text: str = "None reported.") -> Paragraph | ListFlowable:
    if not items:
        return Paragraph(f"<i>{empty_text}</i>", styles["BodySmall"])
    return ListFlowable(
        [ListItem(Paragraph(_sanitize(item), styles["Bullet"]), leftIndent=6) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=14,
    )


def _kv_table(rows: list[tuple[str, str]], styles: dict) -> Table:
    data = [
        [Paragraph(f"<b>{_sanitize(k)}</b>", styles["BodySmall"]), Paragraph(_sanitize(v), styles["BodySmall"])]
        for k, v in rows
    ]
    t = Table(data, colWidths=[4.2 * cm, None])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, _BORDER_GREY),
    ]))
    return t


def _company_section(company: CompanyIntelligence, styles: dict) -> list:
    flow = []
    header = Table(
        [[Paragraph(f"<b>{_sanitize(company.ticker)}</b> - {_sanitize(company.company_name)}", styles["H2"]),
          _badge_table(company.sentiment, company.confidence, styles)]],
        colWidths=[None, 6.5 * cm],
    )
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    flow.append(header)
    flow.append(Spacer(1, 4))

    if company.overall_interpretation:
        flow.append(Paragraph(_sanitize(company.overall_interpretation), styles["Body"]))
        flow.append(Spacer(1, 6))

    signal_data = [
        [Paragraph("<b>Positive Signals</b>", styles["BodySmall"]),
         Paragraph("<b>Negative Signals</b>", styles["BodySmall"])],
        [_bullet_list(company.key_positive_signals, styles),
         _bullet_list(company.key_negative_signals, styles)],
    ]
    signal_table = Table(signal_data, colWidths=[8.6 * cm, 8.6 * cm])
    signal_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), _LIGHT_GREY),
        ("BOX", (0, 0), (-1, -1), 0.4, _BORDER_GREY),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, _BORDER_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    flow.append(signal_table)
    flow.append(Spacer(1, 6))

    if company.important_news:
        flow.append(Paragraph("<b>Relevant News</b>", styles["BodySmall"]))
        flow.append(_bullet_list(company.important_news, styles))
        flow.append(Spacer(1, 6))

    extra_rows = []
    if company.macro_relevance:
        extra_rows.append(("Macro Relevance", company.macro_relevance))
    if company.reddit_community_signal:
        extra_rows.append(("Reddit / Community", company.reddit_community_signal))
    if company.financial_context:
        extra_rows.append(("Financial Context", company.financial_context))
    if extra_rows:
        flow.append(_kv_table(extra_rows, styles))

    flow.append(Spacer(1, 4))
    flow.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER_GREY))
    return flow


def build_report_pdf(report: DailyMarketReport, output_path: str) -> str:
    """
    Render a DailyMarketReport as a PDF at output_path.
    Returns the absolute path written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    styles = _build_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        title=f"Daily Market Intelligence Report — {report.date}",
    )

    story: list = []

    # ── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("Daily Market Intelligence Report", styles["Title"]))
    story.append(Paragraph(
        f"NIFTY 50 Sentiment &amp; Macro Briefing - {_format_display_date(report.date)}",
        styles["SubTitle"],
    ))

    overall_badge = _badge_table(report.overall_market_sentiment, report.overall_confidence, styles)
    overall_row = Table([[overall_badge]], colWidths=[None])
    overall_row.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(overall_row)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=_BRAND_COLOR))

    # ── Macro overview ───────────────────────────────────────────────────
    story.append(Paragraph("Macro Environment", styles["H1"]))
    if report.macro_overview:
        story.append(Paragraph(_sanitize(report.macro_overview), styles["Body"]))
        story.append(Spacer(1, 6))

    macro = report.macro_summary
    macro_table_data = [[
        Paragraph("<b>Drivers</b>", styles["BodySmall"]),
        Paragraph("<b>Events to Watch</b>", styles["BodySmall"]),
    ], [
        _bullet_list(report.key_macro_drivers or macro.key_drivers, styles),
        _bullet_list(report.market_events or macro.market_events, styles),
    ]]
    macro_table = Table(macro_table_data, colWidths=[8.6 * cm, 8.6 * cm])
    macro_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), _LIGHT_GREY),
        ("BOX", (0, 0), (-1, -1), 0.4, _BORDER_GREY),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, _BORDER_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(macro_table)

    # Raw macro data snapshot (commodity/FX/FII figures), if present.
    if macro.market_data:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Market Data Snapshot", styles["H2"]))
        rows = []
        for key, val in macro.market_data.items():
            if not isinstance(val, dict) or "error" in val:
                continue
            label = key.replace("_", " ").title()
            parts = [f"{k}: {v}" for k, v in val.items() if k != "source" and not isinstance(v, dict)]
            rows.append((label, ", ".join(parts) if parts else str(val)))
        if rows:
            story.append(_kv_table(rows, styles))

    # ── Company intelligence ─────────────────────────────────────────────
    if report.company_intelligence:
        story.append(Paragraph("Company Intelligence", styles["H1"]))
        for i, company in enumerate(report.company_intelligence):
            story.extend(_company_section(company, styles))
            if i < len(report.company_intelligence) - 1:
                story.append(Spacer(1, 10))

    # ── Final market view ────────────────────────────────────────────────
    story.append(Paragraph("Final Market View", styles["H1"]))
    if report.final_market_view:
        story.append(Paragraph(_sanitize(report.final_market_view), styles["Body"]))
        story.append(Spacer(1, 8))

    risk_data = [[
        Paragraph("<b>Major Catalysts</b>", styles["BodySmall"]),
        Paragraph("<b>Major Risks</b>", styles["BodySmall"]),
    ], [
        _bullet_list(report.major_catalysts, styles),
        _bullet_list(report.major_risks, styles),
    ]]
    risk_table = Table(risk_data, colWidths=[8.6 * cm, 8.6 * cm])
    risk_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), _LIGHT_GREY),
        ("BOX", (0, 0), (-1, -1), 0.4, _BORDER_GREY),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, _BORDER_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(risk_table)

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.8 * cm, 1.0 * cm, f"Generated {report.generated_at} - Market Analysis")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return os.path.abspath(output_path)


def default_report_path(report: DailyMarketReport, out_dir: str = "reports") -> str:
    """Conventional output path: reports/market_report_<date>.pdf"""
    safe_date = (report.date or datetime.date.today().isoformat()).replace("/", "-")
    return os.path.join(out_dir, f"market_report_{safe_date}.pdf")
