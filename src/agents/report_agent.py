from langchain_core.prompts import ChatPromptTemplate
import json
from pydantic import ValidationError

from src.llm.gateway import get_llm
from src.llm.structured_output import build_json_instruction, unwrap_schema_echo
from src.models.report import DailyMarketReport
from src.prompts.report import SYSTEM_PROMPT


class ReportAgent:
    def __init__(self):
        self.llm = get_llm(agent_name="ReportAgent")

        json_instruction = build_json_instruction(DailyMarketReport)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT + "\n\n" + json_instruction),
                ("human", "{input}")
            ]
        )

    def run(
        self,
        macro_summary,
        company_news,
        market_data,
        financial_data,
        reddit_signals,
        sentiments,
    ) -> DailyMarketReport:
        
        # Exclude massive raw data to save tokens
        if hasattr(macro_summary, "model_dump"):
            macro_data_clean = macro_summary.model_dump(exclude={"market_data"})
        else:
            macro_data_clean = macro_summary

        def safe_dump(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return obj

        input_text = f"""
Macro Data:
{json.dumps(safe_dump(macro_data_clean), indent=2)}

Company News Summaries:
"""
        for ticker, news in company_news.items():
            input_text += f"\n- {ticker}: {getattr(news, 'total_articles', 0)} articles"
            
        input_text += "\n\nFinancial Context:\n"
        for ticker, fin in financial_data.items():
            fin_dump = safe_dump(fin)
            # Remove empty fields to save tokens
            clean_fin = {k: v for k, v in fin_dump.items() if v}
            input_text += f"- {ticker}: {json.dumps(clean_fin)}\n"
            
        input_text += "\nReddit/Community Signals:\n"
        for ticker, red in reddit_signals.items():
            input_text += f"- {ticker}: Sentiment {getattr(red, 'overall_sentiment', 'Unknown')}, Count: {getattr(red, 'post_count', 0)}\n"

        input_text += "\nCompany Sentiments:\n"
        for ticker, sent in sentiments.items():
            input_text += f"- {ticker}:\n{json.dumps(safe_dump(sent), indent=2)}\n"

        messages = self.prompt.invoke({"input": input_text})

        response = self.llm.invoke(messages)
        
        try:
            content = str(response.content).strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # raw_decode from the first '{' instead of find('{')/rfind('}'):
            # rfind grabs the LAST closing brace in the response, so any
            # trailing text after a valid object corrupts the slice.
            start = content.find('{')
            if start == -1:
                raise ValueError("No JSON object found in response.")
            data, _end = json.JSONDecoder().raw_decode(content, start)
            data = unwrap_schema_echo(data)

            # Inject generated_at and date if missing
            from src.tools.common.normalization import utc_now_iso
            import datetime
            if "generated_at" not in data:
                data["generated_at"] = utc_now_iso()
            if "date" not in data:
                data["date"] = datetime.date.today().isoformat()
            if "macro_summary" not in data or not data["macro_summary"]:
                data["macro_summary"] = safe_dump(macro_summary)
                
            return DailyMarketReport.model_validate(data)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from LLM: {e}\nResponse content: {response.content}")