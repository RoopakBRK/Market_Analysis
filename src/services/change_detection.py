from src.models.report import ChangeRow


def _num(value: float) -> str:
    return f"{value:,.2f}"


def _move(previous: float, current: float) -> str:
    """Absolute and percentage move, e.g. "-371.25 (-1.64%)"."""
    change = current - previous
    if previous == 0:
        return f"{change:+,.2f}"
    return f"{change:+,.2f} ({change / previous * 100:+.2f}%)"


class ChangeDetectionService:
    """
    Builds the day-over-day comparison shown in the report.

    Everything here is arithmetic on collected figures; no LLM is involved.
    """

    def detect_change(
        self,
        previous_sentiment,
        current_sentiment,
    ):

        if previous_sentiment is None:
            return None

        if previous_sentiment == current_sentiment:
            return None

        return {
            "previous": previous_sentiment,
            "current": current_sentiment,
            "changed": True,
        }

    # ── Markets: today versus the previous session's close ──────────────────

    def market_changes(self, macro_summary, market_data: dict) -> list[ChangeRow]:
        """
        Price moves since the previous close for the benchmark indices,
        commodities, the rupee, US indices and each watchlist stock.
        Items whose data was unavailable are left out.
        """
        macro = macro_summary.market_data if macro_summary is not None else {}
        india = macro.get("india_market") or {}
        us = macro.get("us_market") or {}

        # (label, quote dict, key holding the current value)
        quotes = [
            ("Nifty 50", india.get("nifty50"), "current"),
            ("Sensex", india.get("sensex"), "current"),
            ("Gold (USD/oz)", macro.get("gold"), "price"),
            ("Brent crude (USD/barrel)", macro.get("crude"), "price"),
            ("USD/INR", macro.get("usd_inr"), "exchange_rate"),
            ("S&P 500", us.get("sp500"), "current"),
            ("Nasdaq", us.get("nasdaq"), "current"),
            ("Dow Jones", us.get("dow_jones"), "current"),
        ]

        rows = []
        for label, quote, current_key in quotes:
            if not quote or quote.get(current_key) is None or quote.get("previous_close") is None:
                continue
            rows.append(self._price_row(label, quote["previous_close"], quote[current_key]))

        for ticker, market in market_data.items():
            if market is None or market.current_price is None or market.previous_close is None:
                continue
            rows.append(self._price_row(f"{ticker} (Rs)", market.previous_close, market.current_price))

        return rows

    @staticmethod
    def _price_row(label: str, previous: float, current: float) -> ChangeRow:
        return ChangeRow(item=label, previous=_num(previous), current=_num(current), change=_move(previous, current))

    # ── Pipeline signals: this run versus the previous run ──────────────────

    def run_changes(self, macro_summary, sentiments: dict, previous: dict | None) -> list[ChangeRow]:
        """
        Changes since the previous run's snapshot (see historical_service):
        sentiment and confidence, FII/DII flows and the RBI repo rate.
        Returns [] when there is no previous run.
        """
        if not previous:
            return []

        rows = []
        prev_macro = previous.get("macro") or {}
        macro = macro_summary.market_data if macro_summary is not None else {}

        if macro_summary is not None:
            rows.append(self._sentiment_row(
                "Macro sentiment",
                prev_macro.get("sentiment"), prev_macro.get("confidence"),
                macro_summary.overall_sentiment, macro_summary.confidence,
            ))

        flows = macro.get("fii_dii_flows") or {}
        for label, key in (("FII net flow (INR crore)", "fii_net"), ("DII net flow (INR crore)", "dii_net")):
            before, now = prev_macro.get(key), flows.get(key)
            if before is None or now is None:
                continue
            if flows.get("date") == prev_macro.get("flows_date"):
                # NSE publishes one figure per session; an unchanged date
                # means no new figure, not a flow that happened to repeat.
                change = f"no new figure since {flows.get('date')}"
            else:
                change = f"{now - before:+,.2f}"
            rows.append(ChangeRow(item=label, previous=_num(before), current=_num(now), change=change))

        before, now = prev_macro.get("repo_rate"), (macro.get("rbi") or {}).get("policy_repo_rate")
        if before is not None and now is not None:
            change = "unchanged" if now == before else f"{now - before:+.2f} pp"
            rows.append(ChangeRow(item="RBI repo rate", previous=f"{before:.2f}%", current=f"{now:.2f}%", change=change))

        prev_companies = previous.get("companies") or {}
        for ticker, sentiment in sentiments.items():
            before = prev_companies.get(ticker) or {}
            rows.append(self._sentiment_row(
                f"{ticker} sentiment",
                before.get("sentiment"), before.get("confidence"),
                sentiment.sentiment, sentiment.confidence,
            ))

        return rows

    def _sentiment_row(self, label, prev_label, prev_confidence, label_now, confidence_now) -> ChangeRow:
        current = f"{label_now} ({confidence_now}%)"
        if prev_label is None:
            return ChangeRow(item=label, previous="not covered", current=current, change="new")

        previous = f"{prev_label} ({prev_confidence}%)" if prev_confidence is not None else prev_label

        if self.detect_change(prev_label, label_now):
            change = f"{prev_label} to {label_now}"
        elif prev_confidence is not None and confidence_now != prev_confidence:
            change = f"confidence {confidence_now - prev_confidence:+d}"
        else:
            change = "unchanged"
        return ChangeRow(item=label, previous=previous, current=current, change=change)
