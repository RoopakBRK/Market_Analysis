from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END

from src.models.financial_data import CompanyFinancials
from src.graph.messages_state import AgentMessagesState

from src.tools.financial_api.company_profile import get_company_profile
from src.tools.financial_api.financial_metrics import get_financial_metrics
from src.tools.financial_api.company_events import get_company_events
from src.tools.common.normalization import utc_now_iso

TOOLS = [
    get_company_profile,
    get_financial_metrics,
    get_company_events,
]


class FinancialDataAgent:
    """
    Deterministically retrieves financial data for a company.
    Executes profile, metrics, and events API calls in parallel.
    No LLM is used.
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
        tool_results = []
        
        def run_tool(tool):
            try:
                result = tool.invoke({"ticker": company})
                return tool.name, result
            except Exception as e:
                return tool.name, {"error": str(e), "ticker": company}
                
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

    def merge_output(self, state: AgentMessagesState):
        import json
        import ast
        ticker = state.get("company_input", "")
        
        # We start with empty dicts for each endpoint's expected data
        profile_data = {}
        metrics_data = {}
        events_data = []
        
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
                    
                if msg.name == "get_company_profile" and data.get("available"):
                    profile_data = data
                elif msg.name == "get_financial_metrics" and data.get("available"):
                    metrics_data = data
                elif msg.name == "get_company_events" and data.get("available"):
                    events_data = data.get("events", [])
        
        # Merge into the CompanyFinancials model
        result = CompanyFinancials(
            ticker=ticker,
            company_name=profile_data.get("company_name"),
            sector=profile_data.get("sector"),
            market_cap=metrics_data.get("market_cap"),
            pe_ratio=metrics_data.get("pe_ratio"),
            eps=metrics_data.get("eps"),
            revenue=metrics_data.get("revenue"),
            debt_to_equity=metrics_data.get("debt_to_equity"),
            upcoming_events=events_data,
            data_source="FinancialAgentAPI",
            retrieved_at=utc_now_iso()
        )
        
        return {"final_result": result}

    def run(self, ticker: str) -> CompanyFinancials:
        initial_state: AgentMessagesState = {
            "messages": [],
            "final_result": None,
            "company_input": ticker
        }
        state = self.graph.invoke(initial_state)
        return state["final_result"]
