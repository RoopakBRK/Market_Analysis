import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.company_news_agent import CompanyNewsAgent
from src.tools.company.economic_times import search_economic_times_news
from src.tools.company.investor_relations import get_investor_relations
from src.tools.company.mint import search_mint_news
from src.tools.company.moneycontrol import search_moneycontrol_news
from src.tools.company.nse import get_nse_announcements
from src.tools.company.reuters import search_reuters_news


COMPANY = "RELIANCE"

TOOLS = [
    ("Reuters", search_reuters_news),
    ("Moneycontrol", search_moneycontrol_news),
    ("Mint", search_mint_news),
    ("Economic Times", search_economic_times_news),
    ("NSE", get_nse_announcements),
    ("Investor Relations", get_investor_relations),
]


def test_tools():
    print("\n" + "=" * 70)
    print("COMPANY NEWS TOOL AUDIT")
    print("=" * 70)
    print(f"Company: {COMPANY}\n")

    results = {}

    with ThreadPoolExecutor(max_workers=len(TOOLS)) as executor:
        futures = {
            executor.submit(tool.invoke, {"company": COMPANY}): name
            for name, tool in TOOLS
        }

        for future in as_completed(futures):
            name = futures[future]

            try:
                result = future.result()

                articles = result.get("articles", []) if isinstance(result, dict) else []

                results[name] = result

                print(f"✓ {name:<20} {len(articles)} articles")

                for article in articles[:2]:
                    print(f"    - {article.get('title', '')[:100]}")

            except Exception as e:
                print(f"✗ {name:<20} ERROR: {e}")

    print("\n" + "=" * 70)
    print("RAW TOOL RESULTS")
    print("=" * 70)

    print(json.dumps(results, indent=2, default=str))

import pytest
@pytest.fixture
def raw_results():
    return {
        "Economic Times": {"articles": [{"title": "Test 1"}], "company": "RELIANCE"},
        "Moneycontrol": {"articles": [{"title": "Test 2"}], "company": "RELIANCE"}
    }

def test_agent(raw_results, monkeypatch):
    class MockLLM:
        def bind_tools(self, *args, **kwargs):
            return self
        def invoke(self, *args, **kwargs):
            from langchain_core.messages import AIMessage
            return AIMessage(content="", tool_calls=[
                {"name": "search_economic_times_news", "args": {"company": "RELIANCE"}, "id": "1"},
                {"name": "search_moneycontrol_news", "args": {"company": "RELIANCE"}, "id": "2"}
            ])
            
    import src.agents.company_news_agent
    monkeypatch.setattr("src.agents.company_news_agent.get_llm", lambda agent_name="": MockLLM())
    
    print("\n" + "=" * 60)
    print("COMPANY NEWS QUALITY AUDIT")
    print("=" * 60)

    agent = CompanyNewsAgent()
    result = agent.run(COMPANY)

    print(f"\nCompany: {result.company_name}")

    raw_articles_count = sum(len(res.get("articles", [])) if isinstance(res, dict) else 0 for res in raw_results.values())
    
    # We can approximate intermediate counts, or just show raw -> final
    # The agent hides the internals in `rank_and_filter_articles` unless we mock it or extract it.
    # We'll just show Raw vs Final, or we can mock it here for the audit.
    
    print(f"Raw articles:               {raw_articles_count}")
    print(f"Final articles:             {result.total_articles}")

    print("\nRANK | SCORE | SOURCE | DATE | TITLE")
    print("-" * 60)

    for i, article in enumerate(result.articles, 1):
        score = agent.score_article(article)
        source = (article.source or "")[:2].upper()
        date = (article.published_at or "N/A")[:10]
        title = (article.title or "")[:60]
        
        print(f"{i:<4} | {score:<5} | {source:<6} | {date:<4} | {title}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    raw_results = test_tools()
    test_agent(raw_results)
