import json
import ast
from typing import TypedDict, Annotated, Any
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.llm.gateway import get_llm
from src.models.company import CompanyNews, NewsArticle
from src.prompts.company import SYSTEM_PROMPT

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


class CompanyNewsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    final_result: CompanyNews | None
    company_input: str


class CompanyNewsAgent:
    def __init__(self):
        self.llm = get_llm().bind_tools(TOOLS)
        
        # Tool Node
        self.tool_node = ToolNode(TOOLS)
        
        # Graph Builder
        builder = StateGraph(CompanyNewsState)
        
        builder.add_node("llm_node", self.call_llm)
        builder.add_node("tools_node", self.tool_node)
        builder.add_node("merge_node", self.merge_output)
        
        builder.add_edge(START, "llm_node")
        
        builder.add_conditional_edges(
            "llm_node",
            tools_condition,
            {
                "tools": "tools_node",
                "__end__": "merge_node",
            },
        )
        
        builder.add_edge("tools_node", "llm_node")
        builder.add_edge("merge_node", END)
        
        self.graph = builder.compile()

    def call_llm(self, state: CompanyNewsState):
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def deduplicate_articles(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        seen_urls = set()
        seen_titles = set()
        deduped = []
        for a in articles:
            # Normalize title for fallback dedup
            norm_title = a.title.strip().lower() if a.title else ""
            
            if a.url:
                if a.url not in seen_urls:
                    seen_urls.add(a.url)
                    if norm_title:
                        seen_titles.add(norm_title)
                    deduped.append(a)
            else:
                if norm_title and norm_title not in seen_titles:
                    seen_titles.add(norm_title)
                    deduped.append(a)
                    
        return deduped

    def merge_output(self, state: CompanyNewsState):
        company = state.get("company_input", "")
        company_name = company
        
        all_articles_data = []
        
        for msg in state["messages"]:
            if msg.type == "tool":
                # Safely parse the tool string content
                if isinstance(msg.content, dict):
                    data = msg.content
                else:
                    content = str(msg.content).strip()
                    data = None
                    try:
                        data = json.loads(content)
                    except Exception:
                        try:
                            data = ast.literal_eval(content)
                        except Exception:
                            pass
                
                if not isinstance(data, dict):
                    continue
                
                # Check if tool provided a longer, more descriptive company name
                tool_company = data.get("company")
                if tool_company and isinstance(tool_company, str) and len(tool_company) > len(company_name):
                    company_name = tool_company
                    
                articles = data.get("articles", [])
                if isinstance(articles, list):
                    for a in articles:
                        if isinstance(a, dict):
                            all_articles_data.append(a)
        
        # Convert dictionary to Pydantic NewsArticle
        news_articles = []
        for a_data in all_articles_data:
            try:
                # model_validate is safer as it converts and validates types correctly
                news_articles.append(NewsArticle.model_validate(a_data))
            except Exception:
                # Skip malformed articles
                continue
                
        # Deduplicate
        deduped_articles = self.deduplicate_articles(news_articles)
        
        result = CompanyNews(
            ticker=company,
            company_name=company_name,
            articles=deduped_articles,
            total_articles=len(deduped_articles)
        )
        
        return {"final_result": result}


    def run(self, company: str) -> CompanyNews:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Fetch the latest news and announcements for {company}")
        ]
        
        initial_state: CompanyNewsState = {
            "messages": messages,
            "final_result": None,
            "company_input": company
        }
        state = self.graph.invoke(initial_state)
        return state["final_result"]