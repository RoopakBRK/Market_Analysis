from langchain_core.prompts import ChatPromptTemplate
import json
from pydantic import ValidationError

from src.llm.gateway import get_llm
from src.models.sentiment import SentimentResult
from src.models.reddit import RedditSignal
from src.models.financial_data import CompanyFinancials
from src.prompts.sentiment import SYSTEM_PROMPT


class SentimentAgent:
    def __init__(self):
        self.llm = get_llm(agent_name="SentimentAgent")
        self.schema = SentimentResult.model_json_schema()

    def run(
        self,
        company_news,
        reddit_signal: RedditSignal | None = None,
        financial_data: CompanyFinancials | None = None,
    ) -> SentimentResult:
        
        # 1. Check for zero inputs short-circuit
        has_news = company_news and company_news.articles
        has_reddit = reddit_signal and reddit_signal.posts
        has_financials = financial_data and (financial_data.market_cap or financial_data.upcoming_events)
        
        ticker = company_news.ticker if company_news else (financial_data.ticker if financial_data else "Unknown")
        company_name = company_news.company_name if company_news else (financial_data.company_name if financial_data else "Unknown")
        
        if not has_news and not has_reddit and not has_financials:
            return SentimentResult(
                ticker=ticker,
                company_name=company_name,
                sentiment="Unknown",
                impact="Unknown",
                confidence=0,
                summary="No valid news, financial data, or reddit signals found for this company.",
                positive_drivers=[],
                negative_drivers=[],
                articles_analyzed=0,
                verified_news_sentiment=None,
                financial_data_signal=None,
                reddit_sentiment=None,
                source_breakdown={}
            )

        # 2. Format a highly compact representation of the inputs
        input_text = f"Company: {company_name}\nTicker: {ticker}\n\n"
        
        input_text += "--- 1. VERIFIED NEWS ---\n"
        if has_news:
            for i, article in enumerate(company_news.articles, 1):
                tier = article.source_type.upper() if hasattr(article, 'source_type') else "UNKNOWN TIER"
                input_text += f"\n{i}. [{tier}] Title: {article.title}\n"
                input_text += f"   Source: {article.source}\n"
                if article.published_at:
                    input_text += f"   Published: {article.published_at}\n"
                if article.summary:
                    input_text += f"   Summary: {article.summary}\n"
        else:
            input_text += "No verified news articles available.\n"
            
        input_text += "\n--- 2. FINANCIAL DATA ---\n"
        if has_financials:
            input_text += f"Market Cap: {financial_data.market_cap}\n"
            input_text += f"PE Ratio: {financial_data.pe_ratio}\n"
            input_text += f"EPS: {financial_data.eps}\n"
            input_text += f"Revenue: {financial_data.revenue}\n"
            input_text += f"Debt to Equity: {financial_data.debt_to_equity}\n"
            input_text += f"Upcoming Events: {', '.join(financial_data.upcoming_events)}\n"
        else:
            input_text += "No financial data available.\n"
            
        input_text += "\n--- 3. REDDIT / COMMUNITY SIGNAL ---\n"
        if has_reddit:
            input_text += f"Aggregate Sentiment Label: {reddit_signal.overall_sentiment}\n"
            input_text += f"Post Count: {reddit_signal.post_count}\n"
            for i, post in enumerate(reddit_signal.posts[:5], 1):
                input_text += f"\n{i}. Title: {post.title}\n"
                input_text += f"   Subreddit: {post.subreddit} | Score: {post.score}\n"
        else:
            input_text += "No Reddit community signals available.\n"

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

        messages = prompt.invoke({"input": input_text})
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
        data["ticker"] = ticker
        data["company_name"] = company_name
        data["articles_analyzed"] = len(company_news.articles) if has_news else 0

        try:
            return SentimentResult.model_validate(data)
        except ValidationError as e:
            raise ValueError(
                f"SentimentResult validation failed.\n"
                f"Validation error: {e}\n"
                f"Parsed data: {data}"
            ) from e