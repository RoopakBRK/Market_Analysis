from pydantic import BaseModel, Field


class MarketData(BaseModel):
    """
    Structured output from the Market Data Agent.
    """

    ticker: str | None = None

    current_price: float | None = None

    previous_close: float | None = None

    day_change_percent: float | None = None

    volume: int | None = None

    fifty_two_week_high: float | None = None

    fifty_two_week_low: float | None = None

    rsi: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Relative Strength Index",
    )

    macd: float | None = None

    vwap: float | None = None

    sector: str | None = None

    sector_performance: float | None = None