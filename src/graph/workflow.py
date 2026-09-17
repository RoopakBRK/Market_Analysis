from langgraph.graph import START, END, StateGraph

from src.graph.state import GraphState

from src.graph.nodes import (
    macro_node,
    company_news_node,
    market_data_node,
    financial_data_node,
    reddit_sentiment_node,
    sentiment_node,
    report_node,
)


builder = StateGraph(GraphState)

# -----------------------------
# Nodes
# -----------------------------
builder.add_node("macro_agent", macro_node)
builder.add_node("company_news_agent", company_news_node)
builder.add_node("market_data_agent", market_data_node)
builder.add_node("financial_data_agent", financial_data_node)
builder.add_node("reddit_sentiment_agent", reddit_sentiment_node)
builder.add_node("sentiment_agent", sentiment_node)
builder.add_node("report_agent", report_node)


# -----------------------------
# Edges
# -----------------------------
# Parallel Execution from START
builder.add_edge(START, "macro_agent")
builder.add_edge(START, "company_news_agent")
builder.add_edge(START, "market_data_agent")
builder.add_edge(START, "financial_data_agent")
builder.add_edge(START, "reddit_sentiment_agent")

# Wait for ALL collection branches to complete before sentiment
builder.add_edge("macro_agent", "sentiment_agent")
builder.add_edge("company_news_agent", "sentiment_agent")
builder.add_edge("market_data_agent", "sentiment_agent")
builder.add_edge("financial_data_agent", "sentiment_agent")
builder.add_edge("reddit_sentiment_agent", "sentiment_agent")

# Generate report
builder.add_edge("sentiment_agent", "report_agent")
builder.add_edge("report_agent", END)


graph = builder.compile()