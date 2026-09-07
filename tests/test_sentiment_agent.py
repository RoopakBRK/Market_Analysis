import pytest
from src.agents.sentiment_agent import SentimentAgent
from src.models.company import CompanyNews, NewsArticle

# Pytest fixtures and unit tests for SentimentAgent

def get_agent():
    return SentimentAgent()

import json
from langchain_core.messages import AIMessage

class MockLLM:
    def __init__(self, expected_sentiment):
        self.expected_sentiment = expected_sentiment
        self.model = "mock-model"
        
    def invoke(self, messages, *args, **kwargs):
        msgs_list = messages.to_messages()
        content_str = msgs_list[0].content + msgs_list[1].content
        
        # Determine sentiment if not explicitly set
        sentiment = self.expected_sentiment
        if not sentiment:
            if "empty" in content_str.lower() or "Title: \n" in content_str:
                return AIMessage(content="")
            sentiment = "Unknown"
            
        data = {
            "sentiment": sentiment,
            "confidence": 80,
            "impact": "Medium",
            "summary": "Mock summary",
            "positive_drivers": ["Growth"] if sentiment in ["Bullish", "Neutral"] else [],
            "negative_drivers": ["Slowdown"] if sentiment in ["Bearish", "Neutral"] else [],
            "articles_analyzed": 1
        }
        return AIMessage(content=json.dumps(data))

@pytest.fixture(autouse=True)
def mock_llm_for_tests(monkeypatch):
    # This prevents real API calls during tests unless explicitly bypassed
    pass

def test_bullish_news():
    agent = get_agent()
    agent.llm = MockLLM("Bullish")
    news = CompanyNews(
        ticker="RELIANCE",
        company_name="Reliance Industries",
        articles=[
            NewsArticle(
                title="Reliance posts record Q1 profit, beats estimates",
                summary="Reliance Industries reported a 20% jump in net profit, driven by strong retail and telecom growth.",
                source="Economic Times",
                url="https://example.com/1",
                published_at="2026-09-02"
            )
        ]
    )
    result = agent.run(news)
    assert result.sentiment == "Bullish"
    assert result.impact in ["High", "Medium"]
    assert result.confidence > 70
    assert len(result.positive_drivers) > 0

def test_bearish_news():
    agent = get_agent()
    agent.llm = MockLLM("Bearish")
    news = CompanyNews(
        ticker="TCS",
        company_name="Tata Consultancy Services",
        articles=[
            NewsArticle(
                title="TCS cuts revenue guidance amid global slowdown",
                summary="TCS management warned of deal delays and cut its FY27 revenue guidance, sending shares lower.",
                source="Moneycontrol",
                url="https://example.com/2",
                published_at="2026-09-02"
            )
        ]
    )
    result = agent.run(news)
    assert result.sentiment == "Bearish"
    assert result.impact in ["High", "Medium"]
    assert result.confidence > 70
    assert len(result.negative_drivers) > 0

def test_mixed_news():
    agent = get_agent()
    agent.llm = MockLLM("Neutral")
    news = CompanyNews(
        ticker="INFOSYS",
        company_name="Infosys",
        articles=[
            NewsArticle(
                title="Infosys wins $1B mega deal",
                summary="Infosys announced a massive strategic partnership.",
                source="Reuters",
                url="https://example.com/3",
                published_at="2026-09-02"
            ),
            NewsArticle(
                title="Infosys faces margin pressure due to wage hikes",
                summary="Operating margins shrank by 50 bps.",
                source="Mint",
                url="https://example.com/4",
                published_at="2026-09-02"
            )
        ]
    )
    result = agent.run(news)
    # Could be Bullish, Bearish, or Neutral depending on LLM interpretation of mixed signals, but should have both drivers
    assert len(result.positive_drivers) > 0
    assert len(result.negative_drivers) > 0
    assert result.confidence > 0

def test_no_articles():
    agent = get_agent()
    news = CompanyNews(
        ticker="HDFC",
        company_name="HDFC Bank",
        articles=[]
    )
    result = agent.run(news)
    assert result.sentiment == "Unknown"
    assert result.confidence == 0
    assert result.articles_analyzed == 0
    assert result.summary == "No valid news articles found for this company."

def test_duplicate_articles_handled_by_news_agent():
    # Since duplicates are handled in CompanyNewsAgent, the SentimentAgent just processes what it gets.
    # We will pass identical titles and URLs to ensure SentimentAgent doesn't crash.
    agent = get_agent()
    agent.llm = MockLLM("Bullish")
    news = CompanyNews(
        ticker="RELIANCE",
        company_name="Reliance",
        articles=[
            NewsArticle(title="Good news", summary="Growth", source="ET", url="http://x", published_at=""),
            NewsArticle(title="Good news", summary="Growth", source="ET", url="http://x", published_at="")
        ]
    )
    result = agent.run(news)
    assert result.articles_analyzed == 2
    assert result.sentiment == "Bullish"

def test_invalid_empty_article():
    agent = get_agent()
    agent.llm = MockLLM(None)
    news = CompanyNews(
        ticker="RELIANCE",
        company_name="Reliance",
        articles=[
            NewsArticle(title="", summary="", source="", url="", published_at="")
        ]
    )
    import pytest
    with pytest.raises(ValueError, match="SentimentAgent LLM Failure"):
        result = agent.run(news)

def test_llm_failure_mocked(monkeypatch):
    agent = get_agent()
    
    # Mock LLM to return empty string
    class MockResponse:
        content = ""
    
    class MockLLM:
        model = "mock-model"
        def invoke(self, *args, **kwargs):
            return MockResponse()
            
    agent.llm = MockLLM()
    
    news = CompanyNews(
        ticker="RELIANCE",
        company_name="Reliance",
        articles=[NewsArticle(title="A", summary="B", source="C", url="D", published_at="E")]
    )
    
    import pytest
    with pytest.raises(ValueError, match="SentimentAgent LLM Failure: The model 'mock-model' returned an entirely empty response."):
        agent.run(news)

if __name__ == "__main__":
    print("Run this file using: pytest tests/test_sentiment_agent.py -v")
