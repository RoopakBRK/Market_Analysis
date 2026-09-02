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
        self.formatter_llm = get_llm(agent_name="MacroAgent")

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
        import re
        from src.models.macro import MacroSummary
        
        messages = state["messages"]

        # 1. Deterministically build the raw market data dictionary
        raw_market_data = {}
        for msg in messages:
            if msg.type == "tool":
                tool_name = msg.name.replace("get_", "").replace("_price", "").replace("_data", "").replace("_updates", "").replace("_summary", "").replace("_rate", "")
                
                # safely parse the raw string back to dict since it was converted to string in collect_data
                content = str(msg.content).strip()
                try:
                    # In python strings from dicts use single quotes, ast.literal_eval is safest
                    import ast
                    data = ast.literal_eval(content)
                    raw_market_data[tool_name] = data
                except BaseException:
                    raw_market_data[tool_name] = {"error": "Failed to parse tool output"}

        # 2. Prepare the prompt for the reasoning LLM
        schema = MacroSummary.model_json_schema()
        # Remove the market_data field from the schema prompt since the LLM shouldn't generate it
        if "properties" in schema and "market_data" in schema["properties"]:
            del schema["properties"]["market_data"]

        tool_data_str = json.dumps(raw_market_data, indent=2)

        instruction = SystemMessage(
            content=f"""You are a financial market intelligence expert.

Your task is to analyze the provided macroeconomic data and generate a structured sentiment summary.

Rules:
- Do NOT call any tools.
- Do NOT invent information.
- Return ONLY valid JSON representing the object itself (NOT a JSON schema).
- Do NOT wrap JSON inside markdown blocks (e.g. ```json).
- Do NOT add any text before or after the JSON.

The JSON MUST contain exactly these keys and types:

{json.dumps(schema.get("properties", schema), indent=2)}

Macroeconomic Data:

{tool_data_str}
"""
        )

        # 3. Call the reasoning LLM (we only make ONE LLM call here)
        # Using self.formatter_llm as the reasoning LLM.
        response = self.formatter_llm.invoke([
            instruction,
            HumanMessage(content="Analyze the collected macroeconomic data and produce the required JSON sentiment summary.")
        ])

        content = str(response.content).strip()

        # 4. Diagnostic Error Handling
        if not content:
            model_info = getattr(self.formatter_llm, 'model', 'unknown model')
            prompt_len = len(instruction.content)
            raise ValueError(
                f"MacroAgent LLM Failure: The model '{model_info}' returned an entirely empty response.\n"
                f"This usually indicates a context length issue (prompt length: {prompt_len} chars) "
                f"or a model generation filter. Please try a different model or reduce data size."
            )

        # Defensive JSON parsing
        json_content = content
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            json_content = match.group(0)

        try:
            data = json.loads(json_content)
            # If the model accidentally wrapped it in a schema definition
            if "properties" in data and "overall_sentiment" in data["properties"]:
                data = data["properties"]
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from MacroAgent LLM.\n"
                f"Error: {e}\n"
                f"Response content:\n{content}"
            ) from e

        # 5. Deterministically attach the raw market data
        data["market_data"] = raw_market_data

        # 6. Validate
        try:
            result = MacroSummary.model_validate(data)
        except ValidationError as e:
            raise ValueError(
                f"MacroSummary validation failed.\n"
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
