from langchain_core.prompts import ChatPromptTemplate
import json
from pydantic import ValidationError

from src.llm.gateway import get_llm
from src.models.sentiment import SentimentResult
from src.prompts.sentiment import SYSTEM_PROMPT


class SentimentAgent:
    def __init__(self):
        self.llm = get_llm()

        schema = SentimentResult.model_json_schema()
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
        company_news,
        market_data,
    ) -> SentimentResult:
        messages = self.prompt.invoke(
            {
                "input": f"""
Macro:
{macro_summary}

News:
{company_news}

Market:
{market_data}
"""
            }
        )

        response = self.llm.invoke(messages)
        
        try:
            content = str(response.content).strip()
            
            # Defensive parsing for OSS models that wrap in markdown or hallucinate schemas
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            else:
                # Find the first { and last }
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1:
                    content = content[start:end+1]
                    
            data = json.loads(content)
            return SentimentResult.model_validate(data)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from LLM: {e}\nResponse content: {response.content}")