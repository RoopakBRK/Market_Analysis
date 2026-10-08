"""Offline tests for the keyword heuristics in the news and Reddit agents."""

from src.agents.company_news_agent import CompanyNewsAgent
from src.agents.reddit_sentiment_agent import RedditSentimentAgent
from src.models.company import NewsArticle
from src.models.reddit import RedditPost


def _article(title: str) -> NewsArticle:
    return NewsArticle(title=title, summary="", source="Economic Times", url="https://example.com", published_at="")


def _post(title: str, score: int = 10) -> RedditPost:
    return RedditPost(title=title, subreddit="IndianStockMarket", url="https://example.com", score=score, published_at="")


# ── News scoring ─────────────────────────────────────────────────────────────

def test_keywords_match_whole_words_only():
    agent = CompanyNewsAgent()
    plain = agent.score_article(_article("Council to consider a seventeen day timeline"))

    # "refund" contains "fund" and "dealers" contains "deal"; neither is a hit.
    assert agent.score_article(_article("Council to consider a seventeen day refund timeline")) == plain
    assert agent.score_article(_article("Car dealers see a seventeen day waiting timeline")) == plain

    # The real words still score.
    assert agent.score_article(_article("Sovereign fund buys into the company")) > plain
    assert agent.score_article(_article("Company signs a supply deal")) > plain


def test_articles_naming_the_company_rank_higher():
    agent = CompanyNewsAgent()
    named = _article("Adani Ports cargo volumes rise in September")
    unnamed = _article("Cargo volumes rise in September across the coast")

    args = {"company": "Adani Ports", "ticker": "ADANIPORTS"}
    assert agent.score_article(named, **args) > agent.score_article(unnamed, **args)

    ranked = agent.rank_and_filter_articles([unnamed, named], **args)
    assert ranked[0] is named


def test_ticker_counts_as_naming_the_company():
    agent = CompanyNewsAgent()
    args = {"company": "Tata Motors Passenger Vehicles", "ticker": "TMPV"}

    by_ticker = agent.score_article(_article("TMPV unveils a new brand identity"), **args)
    by_short_name = agent.score_article(_article("Tata Motors PV unveils a new brand identity"), **args)
    neither = agent.score_article(_article("Carmaker unveils a new brand identity"), **args)

    assert by_ticker == by_short_name > neither


# ── Reddit heuristic ─────────────────────────────────────────────────────────

def test_reddit_keywords_match_whole_words_only():
    agent = RedditSentimentAgent()

    # "recalls" contains "calls", "shareholders" contains "hold",
    # "shortage" contains "short": none of them is a sentiment word.
    label, _ = agent._compute_sentiment([_post("Tata Motors recalls cars; shareholders discuss the chip shortage")])
    assert label == "Neutral"


def test_reddit_real_keywords_still_count():
    agent = RedditSentimentAgent()

    assert agent._compute_sentiment([_post("Time to buy, this looks undervalued")])[0] == "Bullish"
    assert agent._compute_sentiment([_post("I would sell before the crash")])[0] == "Bearish"
    assert agent._compute_sentiment([])[0] == "Unknown"
