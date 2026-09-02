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

    return results


def test_agent():
    print("\n" + "=" * 70)
    print("COMPANY NEWS AGENT TEST")
    print("=" * 70)

    agent = CompanyNewsAgent()

    result = agent.run(COMPANY)

    print(f"\nTicker:         {result.ticker}")
    print(f"Company:        {result.company_name}")
    print(f"Total Articles: {result.total_articles}")

    print("\nArticles:")
    print("-" * 70)

    for i, article in enumerate(result.articles, 1):
        print(f"\n{i}. {article.title}")
        print(f"   Source:     {article.source}")
        print(f"   Official:   {article.is_official}")
        print(f"   Published:  {article.published_at}")
        print(f"   URL:        {article.url}")

    print("\n" + "=" * 70)

    return result


if __name__ == "__main__":
    test_tools()
    test_agent()
