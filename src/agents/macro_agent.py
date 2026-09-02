from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
import json
from pydantic import ValidationError

from src.llm.gateway import get_llm
from src.models.macro import MacroSummary
from src.graph.messages_state import AgentMessagesState

from src.tools.macro.crude import get_crude_price
from src.tools.macro.fii_dii import get_fii_dii_flows
from src.tools.macro.gold import get_gold_price
from src.tools.macro.inflation import get_inflation_data
from src.tools.macro.rbi import get_rbi_updates
from src.tools.macro.us_market import get_us_market_summary
from src.tools.macro.usd_inr import get_usd_inr_rate


TOOLS = [
    get_fii_dii_flows,
    get_crude_price,
    get_us_market_summary,
    get_usd_inr_rate,
    get_inflation_data,
    get_gold_price,
    get_rbi_updates,
]


class MacroAgent:
    def __init__(self):
        # Plain LLM for final JSON formatting
        self.formatter_llm = get_llm()

        # Graph builder
        builder = StateGraph(AgentMessagesState)

        builder.add_node("collect_data_node", self.collect_data)
        builder.add_node("formatter_node", self.format_output)

        builder.add_edge(START, "collect_data_node")
        builder.add_edge("collect_data_node", "formatter_node")
        builder.add_edge("formatter_node", END)

        self.graph = builder.compile()

    def collect_data(self, state: AgentMessagesState):
        """Deterministically collect data from all macro tools without an LLM."""
        
        tool_results = []
        
        def run_tool(tool):
            try:
                # Call tool directly using its underlying function
                # The LangChain @tool decorator wraps it, so we can use .invoke
                result = tool.invoke({})
                return tool.name, result
            except Exception as e:
                return tool.name, {"error": str(e)}
                
        with ThreadPoolExecutor(max_workers=len(TOOLS)) as executor:
            futures = [executor.submit(run_tool, t) for t in TOOLS]
            for future in as_completed(futures):
                name, result = future.result()
                # Create a pseudo-ToolMessage so the formatter can process it identically
                tool_results.append(
                    ToolMessage(
                        content=str(result),
                        name=name,
                        tool_call_id=f"deterministic_{name}"
                    )
                )
                
        return {"messages": tool_results}


    def format_output(self, state: AgentMessagesState):
        """Convert collected tool results into a validated MacroSummary."""

        messages = state["messages"]

        # Extract only the outputs returned by tools.
        tool_results = [
            message
            for message in messages
            if message.type == "tool"
        ]

        # Generate the Pydantic JSON schema.
        schema = MacroSummary.model_json_schema()

        # Build a clean formatting prompt.
        tool_data = "\n\n".join(
            f"Tool: {message.name}\nData: {str(message.content)}"
            for message in tool_results
        )

        instruction = SystemMessage(
            content=f"""You are a financial market intelligence formatter.

Your task is to convert the collected macroeconomic tool results
into a structured macro market summary.

Use ONLY the information contained in the tool results.

Rules:
- Do NOT call any tools.
- Do NOT invent information.
- Do NOT infer unsupported facts.
- Do NOT add information that is not present in the tool results.
- If information is unavailable, represent it according to the schema.
- Return ONLY valid JSON.
- Do NOT wrap JSON inside markdown.
- Do NOT explain your answer.
- Do NOT add any text before or after the JSON.

The JSON must strictly match this Pydantic schema:

{json.dumps(schema, indent=2)}

Collected macroeconomic tool results:

{tool_data}
"""
        )

        # IMPORTANT:
        # formatter_llm is NOT bound to any tools.
        response = self.formatter_llm.invoke(
    [
        instruction,
        HumanMessage(
            content="Analyze the collected macroeconomic data and produce the required MacroSummary JSON."
        ),
    ]
)


        content = str(response.content)

        try:
            data = json.loads(content)

        except json.JSONDecodeError as e:
            raise ValueError(
                "Failed to parse JSON from macro formatter LLM.\n"
                f"Error: {e}\n"
                f"Response content: {content}"
            ) from e

        try:
            result = MacroSummary.model_validate(data)

        except ValidationError as e:
            raise ValueError(
                "MacroSummary validation failed.\n"
                f"Validation error: {e}\n"
                f"Parsed data: {data}"
            ) from e

        return {
            "final_result": result
        }

    def run(self) -> MacroSummary:
        """Run the macro intelligence agent."""
        initial_state: AgentMessagesState = {
            "messages": [],
            "final_result": None,
        }

        state = self.graph.invoke(initial_state)

        return state["final_result"]
