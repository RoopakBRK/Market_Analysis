from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.llm.gateway import get_llm
from src.models.macro import MacroSummary

from src.tools.macro.crude import get_crude_price
from src.tools.macro.fii_dii import get_fii_dii_flows
from src.tools.macro.gold import get_gold_price
from src.tools.macro.inflation import get_inflation_data
from src.tools.macro.rbi import get_rbi_updates
from src.tools.macro.us_market import get_us_market_summary
from src.tools.macro.usd_inr import get_usd_inr_rate


TOOLS = [
    get_fii_dii_flows,
    get_crude_price,
    get_us_market_summary,
    get_usd_inr_rate,
    get_inflation_data,
    get_gold_price,
    get_rbi_updates,
]


SYSTEM_PROMPT = """
You are the Macro Market Intelligence Agent.

Your job is to determine today's overall macro sentiment for the Indian stock market.

Use the available tools whenever necessary.

Base your reasoning only on the tool outputs.

Return the final answer as the required structured output.
"""


class MacroAgent:

    def __init__(self):
        self.llm = get_llm().bind_tools(TOOLS).with_structured_output(MacroSummary)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}")
            ]
        )

    def run(self) -> MacroSummary:

        messages = self.prompt.invoke(
            {
                "input": "Generate today's macro market summary."
            }
        )

        return self.llm.invoke(messages)