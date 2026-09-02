from langchain_core.prompts import ChatPromptTemplate
import json
from pydantic import ValidationError

from src.llm.gateway import get_llm
from src.models.sentiment import SentimentResult
from src.prompts.sentiment import SYSTEM_PROMPT


class SentimentAgent:
    def __init__(self):
        self.llm = get_llm(agent_name="SentimentAgent")
        self.schema = SentimentResult.model_json_schema()

    def run(
        self,
        company_news
    ) -> SentimentResult:
        
        # 1. Check for zero articles short-circuit
        if not company_news or not company_news.articles:
            return SentimentResult(
                ticker=company_news.ticker if company_news else "Unknown",
                company_name=company_news.company_name if company_news else "Unknown",
                sentiment="Unknown",
                impact="Unknown",
                confidence=0,
                summary="No valid news articles found for this company.",
                positive_drivers=[],
                negative_drivers=[],
                articles_analyzed=0
            )

        # 2. Format a highly compact representation of the articles
        articles_text = f"Company: {company_news.company_name}\nTicker: {company_news.ticker}\n\nArticles:\n"
        
        for i, article in enumerate(company_news.articles, 1):
            articles_text += f"\n{i}. Title: {article.title}\n"
            articles_text += f"   Source: {article.source}\n"
            if article.published_at:
                articles_text += f"   Published: {article.published_at}\n"
            if article.summary:
                articles_text += f"   Summary: {article.summary}\n"
            articles_text += f"   URL: {article.url}\n"

        schema_str = json.dumps(self.schema.get("properties", self.schema), separators=(",", ":")).replace("{", "{{").replace("}", "}}")
        
        json_instruction = f"""
Return ONLY valid JSON representing the object itself (NOT a JSON schema).
Do NOT wrap JSON inside markdown blocks.
Do NOT explain or add extra text.
The JSON must strictly contain these keys and types:
{schema_str}"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT + "\n\n" + json_instruction),
                ("human", "{input}")
            ]
        )

        messages = prompt.invoke({"input": articles_text})
        response = self.llm.invoke(messages)
        content = str(response.content).strip()

        # 3. Defensive Empty Response Check
        if not content:
            model_info = getattr(self.llm, 'model', 'unknown model')
            msgs_list = messages.to_messages()
            raise ValueError(
                f"SentimentAgent LLM Failure: The model '{model_info}' returned an entirely empty response.\n"
                f"Prompt length: {len(msgs_list[0].content) + len(msgs_list[1].content)} chars."
            )

        # 4. Defensive JSON parsing
        json_content = content
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            json_content = match.group(0)

        try:
            data = json.loads(json_content)
            if "properties" in data and "sentiment" in data["properties"]:
                data = data["properties"]
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from SentimentAgent LLM.\n"
                f"Error: {e}\n"
                f"Response content:\n{content}"
            ) from e

        # Ensure ticker and company name matches the context
        data["ticker"] = company_news.ticker
        data["company_name"] = company_news.company_name
        data["articles_analyzed"] = len(company_news.articles)

        try:
            return SentimentResult.model_validate(data)
        except ValidationError as e:
            raise ValueError(
                f"SentimentResult validation failed.\n"
                f"Validation error: {e}\n"
                f"Parsed data: {data}"
            ) from e