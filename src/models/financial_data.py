from typing import Optional
from pydantic import BaseModel, Field


class CompanyFinancials(BaseModel):
    """
    Structured financial profile for a company.

    All numeric fields are Optional — None means the provider did not supply
    the value or the API was unavailable. Do NOT substitute 0 / 0.0 for None.

    NOTE: The Financial Agent API provider has not been confirmed yet.
    This model is implemented with a stub adapter that will be replaced
    once the real API contract is documented.
    """

    ticker: str

    # Provider may return None if ticker is unrecognized.
    company_name: Optional[str] = None

    sector: Optional[str] = None

    # In INR crore for Indian stocks; provider-dependent for globals.
    market_cap: Optional[float] = Field(
        default=None,
        description="Market capitalisation in provider units (INR crore for NSE)."
    )

    pe_ratio: Optional[float] = Field(
        default=None,
        description="Price-to-Earnings ratio."
    )

    eps: Optional[float] = Field(
        default=None,
        description="Earnings Per Share (trailing twelve months)."
    )

    revenue: Optional[float] = Field(
        default=None,
        description="Annual revenue in provider units."
    )

    debt_to_equity: Optional[float] = Field(
        default=None,
        description="Total Debt / Total Equity."
    )

    # Upcoming earnings, AGM, dividend, or regulatory events.
    upcoming_events: list[str] = Field(
        default_factory=list,
        description="Upcoming corporate events (earnings, AGM, dividends, etc.)."
    )

    # Identity of the API/provider that supplied the data.
    data_source: str = "unknown"

    retrieved_at: str = Field(
        description="UTC ISO-8601 timestamp of retrieval."
    )
