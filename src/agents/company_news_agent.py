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
        self.llm = get_llm(agent_name="CompanyNewsAgent").bind_tools(TOOLS)
        
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

    def normalize_url(self, url: str) -> str:
        import urllib.parse
        if not url: return ""
        parsed = urllib.parse.urlparse(url)
        # Remove tracking parameters
        query = urllib.parse.parse_qs(parsed.query)
        query = {k: v for k, v in query.items() if not k.startswith("utm_")}
        new_query = urllib.parse.urlencode(query, doseq=True)
        # Reconstruct URL without fragment
        clean_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), parsed.params, new_query, ''))
        return clean_url

    def normalize_title(self, title: str) -> str:
        import re
        if not title: return ""
        # Lowercase and remove all non-alphanumeric characters
        return re.sub(r'[^a-z0-9]', '', title.lower())

    def score_article(self, article: NewsArticle) -> int:
        score = 0
        text = f"{article.title or ''} {article.summary or ''}".lower()
        
        # Relevance Scoring
        if any(w in text for w in ["earnings", "results", "profit", "revenue", "q1", "q2", "q3", "q4"]):
            score += 10
        elif any(w in text for w in ["acquisition", "buyout", "merger"]):
            score += 10
        elif any(w in text for w in ["investment", "stake", "fund"]):
            score += 9
        elif any(w in text for w in ["dividend", "bonus", "buyback"]):
            score += 8
        elif any(w in text for w in ["contract", "deal", "partnership", "regulatory"]):
            score += 8
        elif any(w in text for w in ["market", "nifty", "sensex", "stocks in news"]):
            # Generic market articles get penalized unless they specifically mention company earnings
            score -= 5
        else:
            score += 1 # generic company mention
            
        # Source Scoring
        source = article.source or ""
        if source in ["NSE", "Investor Relations"]:
            score += 10
        elif source == "Reuters":
            score += 9
        elif source == "Economic Times":
            score += 8
        elif source in ["Moneycontrol", "Mint"]:
            score += 7
            
        return score

    def rank_and_filter_articles(self, articles: list[NewsArticle], top_n: int = 10, max_per_source: int = 5) -> list[NewsArticle]:
        seen_urls = set()
        seen_titles = set()
        deduped = []
        
        # Deduplication
        for a in articles:
            norm_url = self.normalize_url(a.url)
            norm_title = self.normalize_title(a.title)
            
            if norm_url and norm_url in seen_urls:
                continue
            if norm_title and norm_title in seen_titles:
                continue
                
            if norm_url: seen_urls.add(norm_url)
            if norm_title: seen_titles.add(norm_title)
            
            # Store original article with its score dynamically attached for sorting
            a._score = self.score_article(a)
            deduped.append(a)
            
        # Sort by score descending
        deduped.sort(key=lambda x: getattr(x, '_score', 0), reverse=True)
        
        # Filter and enforce diversity
        final_articles = []
        source_counts = {}
        
        for a in deduped:
            if getattr(a, '_score', 0) < 0:
                continue # Skip terrible articles
                
            source = a.source or "Unknown"
            count = source_counts.get(source, 0)
            if count >= max_per_source:
                continue
                
            source_counts[source] = count + 1
            # Clean up the dynamic attribute before Pydantic validation later
            final_articles.append(a)
            
            if len(final_articles) >= top_n:
                break
                
        return final_articles

    def merge_output(self, state: CompanyNewsState):
        company = state.get("company_input", "")
        company_name = company
        
        all_articles_data = []
        
        for msg in state["messages"]:
            if msg.type == "tool":
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
                news_articles.append(NewsArticle.model_validate(a_data))
            except Exception:
                continue
                
        # Rank, filter and deduplicate
        final_articles = self.rank_and_filter_articles(news_articles, top_n=5, max_per_source=5)
        
        # Clean up dynamic _score attribute before returning to avoid Pydantic issues
        for a in final_articles:
            if hasattr(a, '_score'):
                delattr(a, '_score')
        
        result = CompanyNews(
            ticker=company,
            company_name=company_name,
            articles=final_articles,
            total_articles=len(final_articles)
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