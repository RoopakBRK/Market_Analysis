from typing import Any
from langgraph.graph import MessagesState


class AgentMessagesState(MessagesState):
    final_result: Any | None
    # Per-run input (company name / ticker) threaded through sub-agent graphs.
    # MUST be declared here — LangGraph builds its state channels strictly from
    # this TypedDict's annotated keys, so any key missing from this schema is
    # silently dropped by graph.invoke(), even though a plain dict literal has it.
    company_input: str
