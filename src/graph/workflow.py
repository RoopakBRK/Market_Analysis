from langgraph.graph import END, START, StateGraph

from src.graph.nodes import macro_node
from src.graph.state import GraphState



builder = StateGraph(GraphState)

builder.add_node("macro_agent", macro_node)

builder.add_edge(START, "macro_agent")
builder.add_edge("macro_agent", END)
builder.add_node("company_news_agent", company_news_node)

builder.add_edge(START, "macro_agent")
builder.add_edge("macro_agent", "company_news_agent")
builder.add_edge("company_news_agent", END)

builder.add_node(
    "market_data_agent",
    market_data_node,
)

builder.add_edge(
    "company_news_agent",
    "market_data_agent",
)

builder.add_edge(
    "market_data_agent",
    END,
)

builder.add_node(
    "sentiment_agent",
    sentiment_node,
)

builder.add_edge(
    "market_data_agent",
    "sentiment_agent",
)

builder.add_edge(
    "sentiment_agent",
    END,
)

graph = builder.compile()