"""Offline tests for the Firecrawl integration: no network, no API key."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from firecrawl import RateLimitError
from firecrawl.v2.types import Document, DocumentMetadata, SearchData, SearchResultNews
from langchain_core.messages import AIMessage, ToolMessage

import src.tools.firecrawl.client as firecrawl_module
from src.models.company import NewsArticle
from src.tools.common.date_utils import is_within_days, parse_published_date
from src.tools.common.normalization import MAX_BODY_CHARS
from src.tools.common.source_classifier import classify_source, publisher_from_url
from src.tools.firecrawl.article_content import enrich_articles


class FakeFirecrawl:
    """Stands in for the SDK client: canned responses and a record of the calls."""

    def __init__(self):
        self.news = []
        self.pages = {}
        self.error = None
        self.searches = []
        self.scrapes = []
        self.waits = {}

    def search(self, query, **kwargs):
        self.searches.append({"query": query, **kwargs})
        if self.error:
            raise self.error
        return SearchData(news=self.news)

    def scrape(self, url, **kwargs):
        self.scrapes.append(url)
        self.waits[url] = kwargs.get("wait_for")
        if self.error:
            raise self.error
        return self.pages[url]


@pytest.fixture
def firecrawl(monkeypatch):
    """A configured Firecrawl whose client is a FakeFirecrawl."""
    fake = FakeFirecrawl()
    monkeypatch.setattr(firecrawl_module.settings, "FIRECRAWL_API_KEY", "fc-test")
    monkeypatch.setattr(firecrawl_module, "_build_client", lambda: fake)
    return fake


def _news(title, url, snippet="", date=None):
    return SearchResultNews(title=title, url=url, snippet=snippet, date=date)


_ARTICLE_MARKDOWN = """# Adani Ports Q2 profit rises 27%

![chart](https://img.example.com/chart.png)

By A Reporter

File Photo: Containers are stacked at Mundra port in the western state of Gujarat, India. REUTERS/A Photographer

Adani Ports Q2 profit rises 27% as cargo volumes climb at Mundra and other ports (Photo: Company handout)

Adani Ports and Special Economic Zone on Thursday reported a 27% rise in second-quarter profit, helped by [higher cargo volumes](https://example.com/cargo) at its Mundra port.

Also Read | Another story about something else entirely that runs long enough to look like a paragraph

- Revenue from operations rose 21% to 91.67 billion rupees in the quarter ended September 30, the company said.
"""

_ARTICLE_TEXT = (
    "Adani Ports and Special Economic Zone on Thursday reported a 27% rise in second-quarter profit, "
    "helped by higher cargo volumes at its Mundra port. "
    "Revenue from operations rose 21% to 91.67 billion rupees in the quarter ended September 30, the company said."
)


def _page(markdown=_ARTICLE_MARKDOWN, status=200, published="2026-10-08T09:20:33+05:30", url=None):
    return Document(markdown=markdown, metadata=DocumentMetadata(status_code=status, published_time=published, url=url))


def _article(title, url, summary="", published_at="", source="Economic Times"):
    return NewsArticle(title=title, summary=summary, source=source, url=url, published_at=published_at)


# ── Publication dates ────────────────────────────────────────────────────────

_NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def test_relative_dates_are_measured_back_from_now():
    assert parse_published_date("3 hours ago", now=_NOW) == _NOW - timedelta(hours=3)
    assert parse_published_date("1 day ago", now=_NOW) == _NOW - timedelta(days=1)
    assert parse_published_date("an hour ago", now=_NOW) == _NOW - timedelta(hours=1)
    assert parse_published_date("5 mins ago", now=_NOW) == _NOW - timedelta(minutes=5)
    assert parse_published_date("2 weeks ago", now=_NOW) == _NOW - timedelta(weeks=2)
    assert parse_published_date("Yesterday", now=_NOW) == _NOW - timedelta(days=1)


def test_calendar_and_timestamp_dates_parse():
    assert parse_published_date("Oct 7, 2026").date().isoformat() == "2026-10-07"
    assert parse_published_date("7 October 2026").date().isoformat() == "2026-10-07"
    assert parse_published_date("Wed, 07 Oct 2026 09:20:33 GMT").date().isoformat() == "2026-10-07"
    assert parse_published_date("2026-10-07T09:20:33+05:30").utcoffset() == timedelta(hours=5, minutes=30)


def test_unreadable_dates_are_none_and_count_as_recent():
    assert parse_published_date("") is None
    assert parse_published_date("sometime last quarter") is None
    assert is_within_days(None, days=3)
    assert is_within_days(_NOW - timedelta(days=3), days=3, now=_NOW)
    assert not is_within_days(_NOW - timedelta(days=5), days=3, now=_NOW)


# ── Source names ─────────────────────────────────────────────────────────────

def test_known_publishers_are_named_from_their_domain():
    assert publisher_from_url("https://www.reuters.com/world/india/a-story") == "Reuters"
    assert publisher_from_url("https://economictimes.indiatimes.com/markets/a-story") == "Economic Times"
    assert publisher_from_url("https://www.livemint.com/companies/a-story") == "Mint"
    assert publisher_from_url("https://blog.example.org/a-story") == "blog.example.org"
    assert publisher_from_url("") == ""


def test_ap_is_not_matched_inside_other_names():
    assert classify_source("apnews.com") == "tier1"
    assert classify_source("Associated Press") == "tier1"
    assert classify_source("capitalmarket.com") == "tier2"


# ── Search ───────────────────────────────────────────────────────────────────

def test_search_normalises_results_and_enforces_the_date_window(firecrawl):
    firecrawl.news = [
        _news("Fresh", "https://www.reuters.com/a?utm_source=feed", snippet="What happened.", date="2 hours ago"),
        _news("Stale", "https://www.reuters.com/b", date="3 months ago"),
        _news("Undated", "https://blog.example.org/c"),
        _news("", "https://blog.example.org/untitled", date="1 hour ago"),
    ]

    fresh, undated = firecrawl_module.search("Adani Ports", days_back=3)

    assert fresh == {
        "title": "Fresh",
        "url": "https://www.reuters.com/a",
        "source": "Reuters",
        "published_at": (datetime.now(timezone.utc) - timedelta(hours=2)).date().isoformat(),
        "summary": "What happened.",
        "source_type": "tier1",
        "is_official": False,
    }
    assert undated["title"] == "Undated"
    assert undated["source"] == "blog.example.org"
    assert undated["published_at"] == ""


def test_search_asks_for_indian_news_across_all_sites(firecrawl):
    firecrawl_module.search("Adani Ports", max_results=3)
    [call] = firecrawl.searches

    assert call["sources"] == ["news"]
    assert call["country"] == "IN"
    # With a domain filter the news search returns a site's section pages
    # whatever the query, so one must never be sent.
    assert "include_domains" not in call
    # Ten results cost the same as three; the surplus covers stale ones.
    assert call["limit"] == 10


def test_search_returns_no_more_than_was_asked_for(firecrawl):
    firecrawl.news = [_news(f"Story {n}", f"https://blog.example.org/{n}", date="1 day ago") for n in range(8)]
    assert [a["title"] for a in firecrawl_module.search("anything", max_results=3)] == ["Story 0", "Story 1", "Story 2"]


def test_search_and_scrape_do_nothing_without_a_key():
    assert firecrawl_module.search("anything") == []
    assert firecrawl_module.scrape("https://blog.example.org/a") is None


def test_search_and_scrape_survive_an_api_error(firecrawl):
    firecrawl.error = RateLimitError("Rate Limit Exceeded", 429)
    assert firecrawl_module.search("anything") == []
    assert firecrawl_module.scrape("https://blog.example.org/a") is None


# ── Scrape ───────────────────────────────────────────────────────────────────

def test_scrape_keeps_the_prose_and_drops_the_page_furniture(firecrawl):
    firecrawl.pages["https://blog.example.org/a"] = _page()

    page = firecrawl_module.scrape("https://blog.example.org/a")

    assert page == {"text": _ARTICLE_TEXT, "published_at": "2026-10-08T09:20:33+05:30", "url": ""}


def test_scrape_reports_where_a_redirect_link_led(firecrawl):
    link = "https://news.google.com/rss/articles/CBMi"
    firecrawl.pages[link] = _page(url="https://www.livemint.com/companies/a-story?utm_source=google")

    assert firecrawl_module.scrape(link)["url"] == "https://www.livemint.com/companies/a-story"


def test_scrape_finds_the_date_under_the_other_names_publishers_use(firecrawl):
    # Moneycontrol states it only as og:article:published_time.
    firecrawl.pages["https://blog.example.org/a"] = Document(
        markdown=_ARTICLE_MARKDOWN,
        metadata=DocumentMetadata.model_validate({"status_code": 200, "og:article:published_time": "2026-10-06T10:35:33+05:30"}),
    )

    assert firecrawl_module.scrape("https://blog.example.org/a")["published_at"] == "2026-10-06T10:35:33+05:30"


def test_scrape_rejects_block_pages_and_stubs(firecrawl):
    firecrawl.pages["https://blog.example.org/blocked"] = _page(status=403)
    firecrawl.pages["https://blog.example.org/stub"] = _page(markdown="Subscribe to continue reading this story.")

    assert firecrawl_module.scrape("https://blog.example.org/blocked") is None
    assert firecrawl_module.scrape("https://blog.example.org/stub") is None


def test_client_calls_match_the_installed_sdk(monkeypatch):
    """
    Runs the real SDK over a stubbed HTTP layer, so an SDK upgrade that
    renames an argument or a response field fails here rather than in a run.
    """
    import firecrawl.v2.utils.http_client as sdk_http

    sent = {}

    class Response:
        status_code = 200
        ok = True
        headers = {}

        def __init__(self, body):
            self.body = body
            self.text = json.dumps(body)

        def json(self):
            return self.body

    def fake_post(url, headers=None, json=None, timeout=None):
        endpoint = url.rsplit("/", 1)[-1]
        sent[endpoint] = json
        if endpoint == "search":
            return Response({"success": True, "data": {"news": [{
                "title": "Fresh", "url": "https://www.reuters.com/a", "snippet": "What happened.",
                "date": "2 hours ago", "position": 1, "imageUrl": "https://img.example.com/a.jpg",
            }]}})
        return Response({"success": True, "data": {
            "markdown": _ARTICLE_MARKDOWN,
            "metadata": {"statusCode": 200, "publishedTime": "2026-10-08T09:20:33+05:30", "url": "https://www.reuters.com/a"},
        }})

    monkeypatch.setattr(sdk_http.requests, "post", fake_post)
    monkeypatch.setattr(firecrawl_module.settings, "FIRECRAWL_API_KEY", "fc-test")

    articles = firecrawl_module.search("Adani Ports")
    page = firecrawl_module.scrape("https://www.reuters.com/a")

    assert [a["title"] for a in articles] == ["Fresh"]
    assert page == {
        "text": _ARTICLE_TEXT,
        "published_at": "2026-10-08T09:20:33+05:30",
        "url": "https://www.reuters.com/a",
    }

    assert sent["search"]["query"] == "Adani Ports"
    assert sent["search"]["sources"] == [{"type": "news"}]
    assert "includeDomains" not in sent["search"]
    assert sent["search"]["country"] == "IN"
    assert sent["scrape"]["url"] == "https://www.reuters.com/a"
    assert sent["scrape"]["formats"] == ["markdown"]
    assert sent["scrape"]["onlyMainContent"] is True
    # An empty parser list is what stops a PDF being billed per page.
    assert sent["scrape"]["parsers"] == []


# ── Full-text enrichment ─────────────────────────────────────────────────────

def test_enrichment_reads_only_the_top_articles_that_need_it(firecrawl):
    articles = [
        _article("Adani Ports Q2 profit rises 27%", "https://blog.example.org/a"),
        _article("Already summarised", "https://blog.example.org/b", summary="x" * 250),
        _article("Financial Result: Adani Ports", "https://nsearchives.nseindia.com/corporate/f.pdf", source="NSE"),
        _article("Adani Ports cargo volumes rise", "https://blog.example.org/c", summary="A snippet.", published_at="2026-10-07"),
        _article("Adani Ports profit beats estimates", "https://blog.example.org/d"),
    ]
    firecrawl.pages = {article.url: _page() for article in articles}

    enriched = enrich_articles(articles, limit=2)

    # The filing and the summarised article are passed over; the limit stops
    # the last one being read.
    assert sorted(firecrawl.scrapes) == ["https://blog.example.org/a", "https://blog.example.org/c"]
    assert enriched[0].summary == _ARTICLE_TEXT
    assert enriched[0].published_at == "2026-10-08T09:20:33+05:30"
    assert enriched[3].summary == _ARTICLE_TEXT
    assert enriched[3].published_at == "2026-10-07"
    assert [enriched[i] for i in (1, 2, 4)] == [articles[i] for i in (1, 2, 4)]


def test_enrichment_follows_a_google_news_link_to_the_article(firecrawl):
    # What the Reuters and Mint tools return: a headline and a redirect link.
    from_google = _article("Adani Ports Q2 profit rises 27%", "https://news.google.com/rss/articles/CBMi", source="Reuters")
    direct = _article("Adani Ports cargo volumes rise", "https://blog.example.org/c")
    firecrawl.pages[from_google.url] = _page(url="https://www.reuters.com/world/india/adani-ports-q2")
    firecrawl.pages[direct.url] = _page(url="https://blog.example.org/c/amp")

    followed, unchanged = enrich_articles([from_google, direct], limit=5)

    assert followed.summary == _ARTICLE_TEXT
    assert followed.url == "https://www.reuters.com/world/india/adani-ports-q2"
    # The redirect runs in JavaScript, so only that page is given time to settle.
    assert firecrawl.waits[from_google.url] == 3000
    assert firecrawl.waits[direct.url] is None
    # Only a redirect link is replaced; an article keeps the URL it came with.
    assert unchanged.url == "https://blog.example.org/c"


def test_enrichment_trims_the_body_to_its_opening(firecrawl):
    article = _article("Adani Ports Q2 profit rises 27%", "https://blog.example.org/a")
    firecrawl.pages[article.url] = _page(markdown="\n\n".join([_ARTICLE_MARKDOWN] * 10))

    [enriched] = enrich_articles([article], limit=5)

    assert enriched.summary.startswith("Adani Ports and Special Economic Zone")
    assert len(enriched.summary) <= MAX_BODY_CHARS + 1


def test_enrichment_ignores_a_page_that_is_not_the_article(firecrawl):
    article = _article("Tata Motors unveils electric hatchback", "https://blog.example.org/a")
    firecrawl.pages[article.url] = _page()

    assert enrich_articles([article], limit=5) == [article]


def test_enrichment_keeps_the_article_when_the_scrape_fails(firecrawl):
    article = _article("Adani Ports Q2 profit rises 27%", "https://blog.example.org/a", summary="A snippet.")
    firecrawl.error = RateLimitError("Rate Limit Exceeded", 429)

    assert enrich_articles([article], limit=5) == [article]


def test_enrichment_is_off_without_a_key_or_with_a_zero_limit(firecrawl, monkeypatch):
    article = _article("Adani Ports Q2 profit rises 27%", "https://blog.example.org/a")
    firecrawl.pages[article.url] = _page()

    assert enrich_articles([article], limit=0) == [article]
    monkeypatch.setattr(firecrawl_module.settings, "FIRECRAWL_API_KEY", "")
    assert enrich_articles([article], limit=5) == [article]
    assert firecrawl.scrapes == []


# ── Inflation news ───────────────────────────────────────────────────────────

def test_inflation_news_combines_firecrawl_and_tavily(firecrawl, monkeypatch):
    import src.tools.macro.inflation as inflation_module

    firecrawl.news = [
        _news("CPI eases", "https://blog.example.org/cpi", date="2 days ago"),
        _news("WPI rises", "https://blog.example.org/wpi", date="20 days ago"),
    ]
    monkeypatch.setattr(inflation_module.tavily_client, "search", lambda **kwargs: [
        {"title": "CPI eases", "url": "https://blog.example.org/cpi"},
        {"title": "Only on Tavily", "url": "https://blog.example.org/tavily"},
    ])

    articles = inflation_module.get_inflation_data.invoke({})["articles"]

    assert [a["title"] for a in articles] == ["CPI eases", "WPI rises", "Only on Tavily"]


# ── Agents ───────────────────────────────────────────────────────────────────

def test_company_agent_collects_firecrawl_news_and_reads_it_in_full(firecrawl, monkeypatch):
    import src.agents.company_news_agent as agent_module
    from src.tools.firecrawl.company_news import search_company_news_firecrawl

    assert search_company_news_firecrawl in agent_module.TOOLS

    monkeypatch.setattr(agent_module, "TOOLS", [search_company_news_firecrawl])
    firecrawl.news = [_news("Adani Ports Q2 profit rises 27%", "https://www.reuters.com/a", snippet="A snippet.", date="1 day ago")]
    firecrawl.pages["https://www.reuters.com/a"] = _page()

    news = agent_module.CompanyNewsAgent().run("Adani Ports", ticker="ADANIPORTS")

    [article] = news.articles
    assert article.source == "Reuters"
    assert article.source_type == "tier1"
    assert article.summary == _ARTICLE_TEXT
    # The search result's own date is kept over the page's.
    assert article.published_at == (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()


def test_company_agent_keeps_the_fuller_copy_of_a_duplicate():
    from src.agents.company_news_agent import CompanyNewsAgent

    bare = {"title": "Adani Ports wins contract", "summary": "", "url": "https://blog.example.org/a",
            "published_at": "", "source": "Economic Times"}
    full = {**bare, "summary": "What happened.", "published_at": "2026-10-08"}
    state = {
        "company_input": "Adani Ports",
        "ticker_input": "ADANIPORTS",
        "messages": [
            ToolMessage(content=str({"company": "Adani Ports", "articles": [bare]}), name="first", tool_call_id="1"),
            ToolMessage(content=str({"company": "Adani Ports", "articles": [full]}), name="second", tool_call_id="2"),
        ],
    }

    [article] = CompanyNewsAgent().merge_output(state)["final_result"].articles

    assert article.summary == "What happened."
    assert article.published_at == "2026-10-08"


def test_macro_agent_shows_an_article_from_both_searches_once():
    import src.agents.macro_agent as agent_module
    from src.tools.firecrawl.macro_news import search_macro_news_firecrawl

    assert search_macro_news_firecrawl in agent_module.TOOLS

    prompts = []

    class RecordingLLM:
        model = "mock-model"

        def invoke(self, messages, *args, **kwargs):
            prompts.append(messages[0].content)
            return AIMessage(content=json.dumps({
                "overall_sentiment": "Neutral", "confidence": 50, "summary": "Mock summary",
                "key_drivers": [], "market_events": [],
            }))

    article = {"title": "RBI holds the repo rate", "url": "https://blog.example.org/rbi", "summary": "", "source": "Mint"}
    agent = agent_module.MacroAgent()
    agent.formatter_llm = RecordingLLM()
    agent.format_output({"messages": [
        ToolMessage(content=str({"articles": [article]}), name="search_macro_news_tavily", tool_call_id="1"),
        ToolMessage(content=str({"articles": [article]}), name="search_macro_news_firecrawl", tool_call_id="2"),
    ]})

    assert prompts[0].count("RBI holds the repo rate") == 1
