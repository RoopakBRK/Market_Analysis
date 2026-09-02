from langchain_core.prompts import ChatPromptTemplate
import json
from pydantic import ValidationError

from src.llm.gateway import get_llm
from src.models.report import DailyMarketReport
from src.prompts.report import SYSTEM_PROMPT


class ReportAgent:
    def __init__(self):
        self.llm = get_llm()

        schema = DailyMarketReport.model_json_schema()
        schema_str = json.dumps(schema, separators=(",", ":")).replace("{", "{{").replace("}", "}}")
        json_instruction = f"""
Return ONLY valid JSON.
Do NOT wrap JSON inside markdown.
Do NOT explain.
Do NOT add extra text.
The JSON must strictly match this schema:
{schema_str}"""

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT + "\n\n" + json_instruction),
                ("human", "{input}")
            ]
        )

    def run(
        self,
        macro_summary,
        sentiments,
    ) -> DailyMarketReport:
        messages = self.prompt.invoke(
            {
                "input": f"""
Macro Data:
{macro_summary}

Company Sentiments:
{sentiments}
"""
            }
        )

        response = self.llm.invoke(messages)
        
        try:
            content = str(response.content).strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            else:
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1:
                    content = content[start:end+1]
                    
            data = json.loads(content)
            return DailyMarketReport.model_validate(data)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from LLM: {e}\nResponse content: {response.content}")