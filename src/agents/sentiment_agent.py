from langchain_core.prompts import ChatPromptTemplate
import json
from pydantic import ValidationError

from src.llm.gateway import get_llm
from src.models.sentiment import SentimentResult
from src.prompts.sentiment import SYSTEM_PROMPT


class SentimentAgent:
    def __init__(self):
        self.llm = get_llm().bind(response_format={"type": "json_object"})

        schema = SentimentResult.model_json_schema()
        schema_str = json.dumps(schema, indent=2).replace("{", "{{").replace("}", "}}")
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
            content = str(response.content)
            data = json.loads(content)
            return SentimentResult.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM: {e}\nResponse content: {response.content}")
        except ValidationError as e:
            raise e