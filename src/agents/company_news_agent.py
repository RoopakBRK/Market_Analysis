from langchain_core.prompts import ChatPromptTemplate

from src.llm.gateway import get_llm
from src.models.company import CompanyNews

from src.tools.company.economic_times import search_economic_times_news
from src.tools.company.investor_relations import get_investor_relations
from src.tools.company.mint import search_mint_news
from src.tools.company.moneycontrol import search_moneycontrol_news
from src.tools.company.nse import get_nse_announcements
from src.tools.company.reuters import search_reuters_news


TOOLS = [
    search_reuters_news,
    search_moneycontrol_news,
    search_mint_news,
    search_economic_times_news,
    get_nse_announcements,
    get_investor_relations,
]


SYSTEM_PROMPT = """
You are the Company News Agent.

Your responsibility is to collect the latest news for the requested company.

Use the available tools.

Prioritize official announcements.

Remove duplicate news.

Return only structured output.
"""


class CompanyNewsAgent:

    def __init__(self):

        self.llm = (
            get_llm()
            .bind_tools(TOOLS)
            .with_structured_output(CompanyNews) # type: ignore
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{company}")
            ]
        )

    def run(self, company: str) -> CompanyNews:

        messages = self.prompt.invoke(
            {
                "company": company
            }
        )

        return self.llm.invoke(messages)