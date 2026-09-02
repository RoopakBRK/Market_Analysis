from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
import json
from pydantic import ValidationError

from src.llm.gateway import get_llm
from src.models.company import CompanyNews
from src.graph.messages_state import AgentMessagesState
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


class CompanyNewsAgent:
    def __init__(self):
        self.llm = get_llm().bind_tools(TOOLS)
        
        self.formatter_llm = get_llm()
        
        # Tool Node
        self.tool_node = ToolNode(TOOLS)
        
        # Graph Builder
        builder = StateGraph(AgentMessagesState)
        
        builder.add_node("llm_node", self.call_llm)
        builder.add_node("tools_node", self.tool_node)
        builder.add_node("formatter_node", self.format_output)
        
        builder.add_edge(START, "llm_node")
        
        builder.add_conditional_edges(
            "llm_node",
            tools_condition,
            {
                "tools": "tools_node",
                "__end__": "formatter_node",
            },
        )
        
        builder.add_edge("tools_node", "llm_node")
        builder.add_edge("formatter_node", END)
        
        self.graph = builder.compile()

    def call_llm(self, state: AgentMessagesState):
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def format_output(self, state: AgentMessagesState):
        messages = state["messages"]

        tool_results = [
            message
            for message in messages
            if message.type == "tool"
        ]

        schema = CompanyNews.model_json_schema()

        instruction = SystemMessage(
            content=f"""You are a financial news formatter.

Using ONLY the collected tool results below, produce the final
company news output.

Do NOT call tools.
Do NOT invent information.
Do NOT add information that is not present in the tool results.

Return ONLY valid JSON.
Do NOT wrap JSON in markdown.
Do NOT explain anything.

The JSON must strictly match this schema:

    {json.dumps(schema, indent=2)}

    Collected tool results:
    """
            + "\n".join(str(message.content) for message in tool_results)
        )

        response = self.formatter_llm.invoke(
            [
                instruction,
                HumanMessage(
                    content="Analyze the collected company news and produce the required structured JSON output."
                ),
            ]
        )

        try:
            content = str(response.content).strip()

            if content.startswith("```"):
                content = content.removeprefix("```json").removeprefix("```").strip()
                content = content.removesuffix("```").strip()

            data = json.loads(content)
            result = CompanyNews.model_validate(data)
            return {"final_result": result}

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse formatter JSON: {e}\n"
                f"Response: {response.content}"
            ) from e

        except ValidationError as e:
            raise ValueError(
                f"CompanyNews validation failed: {e}\n"
                f"Response: {response.content}"
            ) from e


    def run(self, company: str) -> CompanyNews:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=company)
        ]
        
        initial_state: AgentMessagesState = {
            "messages": messages,
            "final_result": None
        }
        state = self.graph.invoke(initial_state)
        return state["final_result"]