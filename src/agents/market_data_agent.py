from langchain_core.prompts import ChatPromptTemplate

from src.llm.gateway import get_llm
from src.models.market import MarketData

from src.tools.market.price import get_stock_price
from src.tools.market.indicators import get_technical_indicators
from src.tools.market.sector import get_sector_performance


TOOLS = [
    get_stock_price,
    get_technical_indicators,
    get_sector_performance,
]


SYSTEM_PROMPT = """
You are the Market Data Agent.

Collect today's market data.

Use tools whenever necessary.

Return structured output only.
"""


class MarketDataAgent:

    def __init__(self):

        self.llm = (
            get_llm()
            .bind_tools(TOOLS)
            .with_structured_output(MarketData) # type: ignore
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{ticker}")
            ]
        )

    def run(self, ticker: str) -> MarketData:

        messages = self.prompt.invoke(
            {
                "ticker": ticker
            }
        )

        return self.llm.invoke(messages)