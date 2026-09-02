from typing import Any
from langgraph.graph import MessagesState


class AgentMessagesState(MessagesState):
    final_result: Any | None
