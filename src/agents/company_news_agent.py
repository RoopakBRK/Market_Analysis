import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END

from src.models.company import CompanyNews, NewsArticle
from src.graph.messages_state import AgentMessagesState

from src.tools.company.economic_times import search_economic_times_news
from src.tools.company.investor_relations import get_investor_relations
from src.tools.company.mint import search_mint_news
from src.tools.company.moneycontrol import search_moneycontrol_news
from src.tools.company.nse import get_nse_announcements
from src.tools.company.reuters import search_reuters_news
from src.tools.tavily.company_news import search_company_news_tavily
from src.tools.common.normalization import deduplicate_article_dicts
from src.tools.common.source_classifier import classify_source

TOOLS = [
    search_reuters_news,
    search_moneycontrol_news,
    search_mint_news,
    search_economic_times_news,
    get_nse_announcements,
    get_investor_relations,
    search_company_news_tavily,
]

# Tools that look a company up by its NSE symbol rather than by name.
_SYMBOL_TOOLS = {get_nse_announcements.name, get_investor_relations.name}


class CompanyNewsAgent:
    def __init__(self):
        builder = StateGraph(AgentMessagesState)

        builder.add_node("collect_data_node", self.collect_data)
        builder.add_node("merge_node", self.merge_output)

        builder.add_edge(START, "collect_data_node")
        builder.add_edge("collect_data_node", "merge_node")
        builder.add_edge("merge_node", END)

        self.graph = builder.compile()

    def collect_data(self, state: AgentMessagesState):
        """Deterministically collect data from all news tools without an LLM."""
        company = state.get("company_input", "")
        ticker = state.get("ticker_input") or company
        tool_results = []
        
        def run_tool(tool):
            try:
                # Call tool directly using its underlying function
                if "company" in tool.args:
                    query = ticker if tool.name in _SYMBOL_TOOLS else company
                    result = tool.invoke({"company": query})
                else:
                    result = tool.invoke({})
                return tool.name, result
            except Exception as e:
                return tool.name, {"error": str(e), "company": company, "articles": []}
                
        with ThreadPoolExecutor(max_workers=len(TOOLS)) as executor:
            futures = [executor.submit(run_tool, t) for t in TOOLS]
            for future in as_completed(futures):
                name, result = future.result()
                tool_results.append(
                    ToolMessage(
                        content=str(result),
                        name=name,
                        tool_call_id=f"deterministic_{name}"
                    )
                )
                
        return {"messages": tool_results}

    def score_article(self, article: NewsArticle, company: str = "", ticker: str = "") -> int:
        score = article.relevance_score
        text = f"{article.title or ''} {article.summary or ''}".lower()
        # Match whole words only: as substrings, "fund" also hits "refund"
        # and "deal" hits "dealer".
        words = set(re.findall(r"[a-z0-9]+", text))

        def has(*keywords: str) -> bool:
            return any(k in text if " " in k else k in words for k in keywords)
        
        # Relevance Scoring
        if has("earnings", "results", "profit", "revenue", "q1", "q2", "q3", "q4"):
            score += 10
        elif has("acquisition", "buyout", "merger"):
            score += 10
        elif has("investment", "stake", "fund"):
            score += 9
        elif has("dividend", "bonus", "buyback"):
            score += 8
        elif has("contract", "deal", "partnership", "regulatory"):
            score += 8
        elif has("market", "nifty", "sensex", "stocks in news"):
            score -= 5
        else:
            score += 1 

        # Topic and tag pages also list loosely related stories, so favour
        # articles that actually name the company.
        if self._mentions_company(text, words, company, ticker):
            score += 5
            
        # Source Scoring based on tier
        if article.source_type == "official":
            score += 10
        elif article.source_type == "tier1":
            score += 9
        elif article.source_type == "tier2":
            score += 7
            
        return score

    @staticmethod
    def _mentions_company(text: str, words: set[str], company: str, ticker: str) -> bool:
        if ticker and ticker.lower() in words:
            return True
        # The leading two words identify the company ("tata motors", "adani
        # ports"); headlines rarely spell out the full registered name.
        short_name = " ".join(company.lower().split()[:2])
        return bool(short_name) and short_name in text

    def rank_and_filter_articles(
        self,
        articles: list[NewsArticle],
        top_n: int = 10,
        max_per_source: int = 5,
        company: str = "",
        ticker: str = "",
    ) -> list[NewsArticle]:
        # Deduplication happens at dict level in merge_output, here we just rank and limit
        # Store original article with its score dynamically attached for sorting
        for a in articles:
            a.relevance_score = self.score_article(a, company, ticker)
            
        # Sort by score descending
        articles.sort(key=lambda x: x.relevance_score, reverse=True)
        
        final_articles = []
        source_counts = {}
        
        for a in articles:
            if a.relevance_score < 0:
                continue 
                
            source = a.source or "Unknown"
            count = source_counts.get(source, 0)
            if count >= max_per_source:
                continue
                
            source_counts[source] = count + 1
            final_articles.append(a)
            
            if len(final_articles) >= top_n:
                break
                
        return final_articles

    def merge_output(self, state: AgentMessagesState):
        import json
        import ast
        company = state.get("company_input", "")
        ticker = state.get("ticker_input") or company
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
                # Symbol tools echo the ticker back; that is not a display name.
                if tool_company and isinstance(tool_company, str) and tool_company != ticker and len(tool_company) > len(company_name):
                    company_name = tool_company
                    
                articles = data.get("articles", [])
                if isinstance(articles, list):
                    for a in articles:
                        if isinstance(a, dict):
                            # Make sure source_type is populated properly
                            if "source_type" not in a or not a["source_type"]:
                                a["source_type"] = classify_source(a.get("source", ""))
                            if a["source_type"] == "tavily" and "underlying_source_type" in a:
                                a["source_type"] = a["underlying_source_type"]
                            a["is_official"] = (a["source_type"] == "official")
                            all_articles_data.append(a)
        
        # Deduplicate deterministically by URL / Title+Source
        deduped_dicts = deduplicate_article_dicts(all_articles_data)

        # Convert dictionary to Pydantic NewsArticle
        news_articles = []
        for a_data in deduped_dicts:
            try:
                news_articles.append(NewsArticle.model_validate(a_data))
            except Exception:
                continue
                
        # Rank and filter
        final_articles = self.rank_and_filter_articles(
            news_articles, top_n=8, max_per_source=3, company=company, ticker=ticker
        )
        
        result = CompanyNews(
            ticker=ticker,
            company_name=company_name,
            articles=final_articles,
            total_articles=len(final_articles)
        )
        
        return {"final_result": result}

    def run(self, company: str, ticker: str | None = None) -> CompanyNews:
        initial_state: AgentMessagesState = {
            "messages": [],
            "final_result": None,
            "company_input": company,
            "ticker_input": ticker or company,
        }
        state = self.graph.invoke(initial_state)
        return state["final_result"]