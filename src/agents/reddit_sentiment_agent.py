from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END

from src.models.reddit import RedditSignal, RedditPost
from src.graph.messages_state import AgentMessagesState
from src.tools.reddit.company_posts import search_company_reddit
from src.tools.common.normalization import utc_now_iso


class RedditSentimentAgent:
    """
    Deterministically computes a community sentiment signal from Reddit posts.
    No LLM is used. Relies on keyword heuristics and post scores.
    """

    def __init__(self):
        builder = StateGraph(AgentMessagesState)

        builder.add_node("collect_data_node", self.collect_data)
        builder.add_node("merge_node", self.merge_output)

        builder.add_edge(START, "collect_data_node")
        builder.add_edge("collect_data_node", "merge_node")
        builder.add_edge("merge_node", END)

        self.graph = builder.compile()

    def collect_data(self, state: AgentMessagesState):
        company = state.get("company_input", "")
        # Since we only have one tool for company posts here, we just call it.
        try:
            result = search_company_reddit.invoke({"ticker": company, "company": company})
        except Exception as e:
            result = {"error": str(e), "posts": []}
            
        msg = ToolMessage(
            content=str(result),
            name="search_company_reddit",
            tool_call_id="deterministic_reddit"
        )
                
        return {"messages": [msg]}

    def _compute_sentiment(self, posts: list[RedditPost]) -> tuple[str, float]:
        if not posts:
            return "Unknown", 0.0
            
        bullish_words = {"buy", "bull", "calls", "moon", "undervalued", "breakout", "long", "hold", "profit"}
        bearish_words = {"sell", "bear", "puts", "overvalued", "crash", "short", "dump", "loss"}
        
        total_score = 0
        total_weight = 0
        
        for post in posts:
            text = f"{post.title} {post.body}".lower()
            weight = post.score if post.score > 0 else 1
            
            bull_count = sum(1 for w in bullish_words if w in text)
            bear_count = sum(1 for w in bearish_words if w in text)
            
            if bull_count > bear_count:
                total_score += weight
            elif bear_count > bull_count:
                total_score -= weight
                
            total_weight += weight
            
        if total_weight == 0:
            return "Neutral", 0.0
            
        avg_score = total_score / total_weight
        
        if avg_score > 0.2:
            return "Bullish", avg_score
        elif avg_score < -0.2:
            return "Bearish", avg_score
        return "Neutral", avg_score

    def merge_output(self, state: AgentMessagesState):
        import json
        import ast
        company = state.get("company_input", "")
        
        all_posts_data = []
        
        for msg in state["messages"]:
            if msg.type == "tool":
                data = None
                if isinstance(msg.content, dict):
                    data = msg.content
                else:
                    content = str(msg.content).strip()
                    try:
                        data = json.loads(content)
                    except Exception:
                        try:
                            data = ast.literal_eval(content)
                        except Exception:
                            pass
                            
                if not isinstance(data, dict):
                    continue
                    
                posts = data.get("posts", [])
                if isinstance(posts, list):
                    for p in posts:
                        if isinstance(p, dict):
                            all_posts_data.append(p)
                            
        # Convert dictionary to Pydantic RedditPost
        reddit_posts = []
        for p_data in all_posts_data:
            try:
                reddit_posts.append(RedditPost.model_validate(p_data))
            except Exception:
                continue
                
        sentiment_label, avg_score = self._compute_sentiment(reddit_posts)
        
        result = RedditSignal(
            ticker=company,
            company_name=company,
            posts=reddit_posts,
            overall_sentiment=sentiment_label,
            post_count=len(reddit_posts),
            avg_score=avg_score,
            retrieved_at=utc_now_iso()
        )
        
        return {"final_result": result}

    def run(self, company: str) -> RedditSignal:
        initial_state: AgentMessagesState = {
            "messages": [],
            "final_result": None,
            "company_input": company
        }
        state = self.graph.invoke(initial_state)
        return state["final_result"]
