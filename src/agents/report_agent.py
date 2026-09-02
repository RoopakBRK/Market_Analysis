from langchain_core.prompts import ChatPromptTemplate

from src.llm.gateway import get_llm
from src.models.report import DailyMarketReport


SYSTEM_PROMPT = """
Generate a concise daily market report.

Include

- Macro Summary

- Top Positive Stocks

- Top Negative Stocks

- Important Events

Return structured output only.
"""


class ReportAgent:

    def __init__(self):

        self.llm = (
            get_llm()
            .with_structured_output(DailyMarketReport)
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}")
            ]
        )

    def run(
        self,
        macro_summary,
        sentiments,
    ):

        messages = self.prompt.invoke(
            {
                "input": f"""
Macro

{macro_summary}

Sentiments

{sentiments}
"""
            }
        )

        return self.llm.invoke(messages)