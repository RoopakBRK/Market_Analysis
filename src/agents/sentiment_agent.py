from langchain_core.prompts import ChatPromptTemplate

from src.llm.gateway import get_llm
from src.models.sentiment import SentimentResult


SYSTEM_PROMPT = """
You are an expert equity research analyst.

Analyze

1. Macro environment
2. Company news
3. Technical indicators

Determine

- Overall sentiment
- Impact
- Expected duration

Never hallucinate.

Base every conclusion on evidence.

Return structured output only.
"""


class SentimentAgent:

    def __init__(self):

        self.llm = (
            get_llm()
            .with_structured_output(SentimentResult)
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
        company_news,
        market_data,
    ):

        messages = self.prompt.invoke(
            {
                "input": f"""
Macro:
{macro_summary}

News:
{company_news}

Market:
{market_data}
"""
            }
        )

        return self.llm.invoke(messages)