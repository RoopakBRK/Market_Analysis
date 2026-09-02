from langgraph.graph import START, END, StateGraph

from src.graph.state import GraphState

from src.graph.nodes import (
    macro_node,
    company_news_node,
    market_data_node,
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

builder.add_node("sentiment_agent", sentiment_node)

builder.add_node("report_agent", report_node)


# -----------------------------
# Edges
# -----------------------------
builder.add_edge(START, "macro_agent")

# Parallel Execution
builder.add_edge("macro_agent", "company_news_agent")
builder.add_edge("macro_agent", "market_data_agent")

# Wait for BOTH to complete
builder.add_edge("company_news_agent", "sentiment_agent")
builder.add_edge("market_data_agent", "sentiment_agent")

# Generate report
builder.add_edge("sentiment_agent", "report_agent")

builder.add_edge("report_agent", END)


graph = builder.compile()