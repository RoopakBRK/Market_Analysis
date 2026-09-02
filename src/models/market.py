from pydantic import BaseModel, Field


class MarketData(BaseModel):
    """
    Structured output from the Market Data Agent.
    """

    ticker: str

    current_price: float

    previous_close: float

    day_change_percent: float

    volume: int

    fifty_two_week_high: float

    fifty_two_week_low: float

    rsi: float = Field(
        ge=0,
        le=100,
        description="Relative Strength Index",
    )

    macd: float

    vwap: float

    sector: str

    sector_performance: float