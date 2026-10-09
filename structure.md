# Market Analysis — Codebase Structure

## Overview

This is a **multi-agent market intelligence system** built with **LangGraph** and **LangChain**.
Given a watchlist of stock tickers it spins up a directed acyclic graph of specialised agents that:

1. Run **in parallel** to collect data from diverse sources (news feeds, exchange filings, Yahoo Finance, Reddit, Tavily news search)
2. Put each stock's day in **historical context** using a 20-year price-history store (Qdrant RAG)
3. Converge into a **SentimentAgent** that synthesises per-ticker sentiment
4. Produce a single **DailyMarketReport** via the **ReportAgent**, including a **day-over-day comparison**
5. **Fact-check** the report's commentary against the facts with a second model, removing anything unsupported

The LLM backend is **Groq**, with **Claude** (Anthropic) as an optional last-resort fallback. Most agents
are **fully deterministic** (zero LLM calls). Only `MacroAgent`, `SentimentAgent`, `ReportAgent` and
`FactCheckAgent` invoke an LLM, and the `ReportAgent` uses it for commentary only: every fact in the
report (labels, drivers, headlines, prices, timestamps, comparisons, historical passages) is attached
in code.

The watchlist lives in `src/utils/constants.py` (`WATCHLIST`: NSE symbol → company name).

---

## Directory Tree

```
Market Analysis/
│
├── apps/                            # Application entry points
│   ├── api/                         # API server (placeholder)
│   │   ├── main.py
│   │   ├── dependencies.py
│   │   └── routes.py
│   └── worker/
│       ├── run_pipeline.py          # Pipeline entry point: run graph, write PDF, save history + DB
│       └── scheduler.py             # Thin wrapper around run_pipeline
│
├── config/                          # App-level configuration
│   ├── settings.py                  # Pydantic BaseSettings — all env vars & model names
│   └── prompts.py                   # Config-level prompt constants
│
├── src/                             # Core source package
│   │
│   ├── agents/                      # One class per agent
│   │   ├── company_news_agent.py
│   │   ├── fact_check_agent.py           # Removes unsupported commentary from the report
│   │   ├── financial_data_agent.py
│   │   ├── historical_context_agent.py   # Statistics and precedents from the RAG store
│   │   ├── macro_agent.py
│   │   ├── market_data_agent.py
│   │   ├── reddit_sentiment_agent.py
│   │   ├── sentiment_agent.py
│   │   └── report_agent.py
│   │
│   ├── graph/                       # LangGraph wiring
│   │   ├── state.py                 # GraphState TypedDict (top-level shared state)
│   │   ├── messages_state.py        # AgentMessagesState TypedDict (per-agent state)
│   │   ├── nodes.py                 # Node wrapper functions — one per agent
│   │   └── workflow.py              # Builds & compiles the StateGraph
│   │
│   ├── rag/                         # Price-history RAG (see "Price-History RAG" below)
│   │   ├── universe.py              # Which companies: live NIFTY 50 list from NSE + watchlist
│   │   ├── loader.py                # 20 years of daily bars via yfinance
│   │   ├── chunking.py              # Calendar-window chunking: year / month / event
│   │   ├── history_stats.py         # Statistics computed from stored prices (no LLM)
│   │   ├── reader.py                # Exact reads: scan a ticker's chunks, fetch events by id
│   │   ├── embeddings.py            # Dense, sparse and reranker models (fastembed, local)
│   │   ├── store.py                 # Qdrant client, collection schema, upserts
│   │   ├── retriever.py             # Hybrid search + RRF fusion + cross-encoder rerank (query CLI)
│   │   ├── ingest.py                # CLI: build / refresh the store
│   │   └── query.py                 # CLI: retrieve + LLM answer
│   │
│   ├── tools/                       # LangChain @tool-decorated data fetchers
│   │   ├── common/                  # Shared utility functions
│   │   │   ├── normalization.py     # utc_now_iso(), deduplicate_article_dicts()
│   │   │   ├── source_classifier.py # Classifies a source into a reliability tier
│   │   │   ├── scraper_utils.py     # HTTP scraping helpers (headers, HTML fetch)
│   │   │   ├── yahoo_finance.py     # Quotes, daily bars, sector lookup, NSE symbol mapping
│   │   │   ├── google_news.py       # Google News RSS search (dated, per-publisher)
│   │   │   ├── citations.py         # (empty placeholder)
│   │   │   ├── date_utils.py        # (empty placeholder)
│   │   │   └── deduplicate.py       # (empty placeholder)
│   │   │
│   │   ├── company/                 # Company-specific news sources
│   │   │   ├── economic_times.py    # Economic Times topic page
│   │   │   ├── investor_relations.py# Press releases / results / investor meets, from NSE filings
│   │   │   ├── mint.py              # Mint, via Google News
│   │   │   ├── moneycontrol.py      # Moneycontrol tag page
│   │   │   ├── nse.py               # NSE corporate announcements API
│   │   │   └── reuters.py           # Reuters, via Google News
│   │   │
│   │   ├── financial_api/           # Structured financial data via external API (stub)
│   │   │   ├── client.py
│   │   │   ├── company_profile.py
│   │   │   ├── financial_metrics.py
│   │   │   └── company_events.py
│   │   │
│   │   ├── macro/                   # Macro-economic data fetchers
│   │   │   ├── crude.py             # Brent crude, with its impact on India
│   │   │   ├── fii_dii.py           # FII / DII institutional flows (NSE)
│   │   │   ├── gold.py              # Gold price
│   │   │   ├── india_market.py      # Nifty 50 and Sensex
│   │   │   ├── inflation.py         # Recent CPI / WPI news coverage (no fixed numbers)
│   │   │   ├── rbi.py               # Live RBI policy rates from rbi.org.in
│   │   │   ├── us_market.py         # S&P 500, Nasdaq, Dow
│   │   │   └── usd_inr.py           # USD/INR exchange rate
│   │   │
│   │   ├── market/                  # Per-stock market data
│   │   │   ├── price.py             # Current price, previous close, % change
│   │   │   ├── indicators.py        # RSI(14), MACD(12,26,9), 20-session VWAP
│   │   │   └── sector.py            # Day move of the matching Nifty sector index
│   │   │
│   │   ├── reddit/                  # Reddit (PRAW) tools
│   │   │   ├── client.py
│   │   │   ├── company_posts.py
│   │   │   └── market_posts.py
│   │   │
│   │   └── tavily/                  # Tavily news-search tools
│   │       ├── client.py            # News-mode search with an enforced date window
│   │       ├── company_news.py
│   │       ├── macro_news.py
│   │       └── sector_news.py
│   │
│   ├── models/                      # Pydantic output models
│   │   ├── company.py               # CompanyNews, NewsArticle
│   │   ├── fact_check.py            # FactCheckResult, UnsupportedClaim
│   │   ├── financial_data.py        # CompanyFinancials
│   │   ├── macro.py                 # MacroSummary
│   │   ├── market.py                # MarketData
│   │   ├── reddit.py                # RedditSignal, RedditPost
│   │   ├── report.py                # DailyMarketReport, CompanyIntelligence, ChangeRow, ReportNarrative
│   │   ├── sentiment.py             # SentimentResult
│   │   └── state.py                 # Misc shared state models
│   │
│   ├── prompts/                     # System-prompt strings (no logic)
│   │   ├── company.py
│   │   ├── fact_check.py            # What the fact-checker may and may not flag
│   │   ├── history.py               # Price-history Q&A (src/rag/query.py)
│   │   ├── macro.py
│   │   ├── market.py
│   │   ├── report.py
│   │   └── sentiment.py
│   │
│   ├── llm/                         # LLM abstraction layer
│   │   ├── gateway.py               # LLMGateway, RateLimitedLLM, get_llm(), get_checker_llm()
│   │   ├── anthropic_fallback.py    # Claude adapter (official Anthropic SDK) for the fallback chain
│   │   └── structured_output.py     # Structured output helpers
│   │
│   ├── storage/                     # Persistence layer (PostgreSQL)
│   │   ├── postgres.py              # SQLAlchemy engine & session factory
│   │   ├── models.py                # ORM table definitions
│   │   └── repositories.py          # Save functions + persist_pipeline_result()
│   │
│   ├── services/                    # Business-logic services
│   │   ├── change_detection.py      # Builds the day-over-day comparison rows
│   │   ├── historical_service.py    # Run-history snapshots (data/history/<date>.json)
│   │   ├── confidence_service.py    # Confidence score aggregation (unused)
│   │   └── report_service.py        # Renders DailyMarketReport to PDF
│   │
│   ├── utils/
│   │   └── constants.py             # WATCHLIST
│   │
│   ├── guardrails/                  # (empty placeholders)
│   └── observability/               # (empty placeholders)
│
├── tests/                           # Offline tests by default; `pytest -m live` for live checks
├── data/history/                    # One JSON snapshot per run day (git-ignored)
├── reports/                         # Generated PDF reports (git-ignored)
├── docker/                          # Docker / Compose files
├── docs/                            # Additional documentation
├── .env                             # Environment variables (secrets)
├── pyproject.toml                   # Project metadata + pytest config
└── requirements.txt                 # Python dependencies
```

---

## Execution Graph

```
START
  │
  ├──► MacroAgent              ← runs once
  ├──► CompanyNewsAgent        ← runs per ticker
  ├──► MarketDataAgent         ← runs per ticker ──────────┐
  ├──► FinancialDataAgent      ← runs per ticker           │
  └──► RedditSentimentAgent    ← runs per ticker           │
             │                                             ▼
             ▼   (all five branches must complete)   HistoricalContextAgent
         SentimentAgent        ← LLM call per ticker       │  ← RAG lookup per ticker, no LLM
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    ▼   (both must complete)
                               ReportAgent           ← one LLM call (commentary only)
                                    │
                                    ▼
                              FactCheckAgent         ← one LLM call, on a different model
                                    │
                                   END
```

Defined in `src/graph/workflow.py`. The graph uses a single shared `GraphState` TypedDict
(`src/graph/state.py`) to pass data between nodes.

After the graph finishes, `apps/worker/run_pipeline.py` writes the PDF, saves today's snapshot to
`data/history/`, and saves the run to PostgreSQL if a database is reachable.

---

## Agent Reference

---

### 1. `MacroAgent`
**File:** `src/agents/macro_agent.py`

**What it does:**
Collects macroeconomic signals in parallel, then makes **one LLM call** to synthesise them into a
structured `MacroSummary` (overall sentiment, confidence, key drivers, market events). Numeric tool
output is attached to the summary as `market_data` in code; news articles are given to the LLM as
dated context but not stored there. `last_updated` is set in code.

**Uses LLM?** ✅ Yes — 1 call.

**Tools (all run in parallel via `ThreadPoolExecutor`):**

| Tool function | File | Data fetched |
|---|---|---|
| `get_fii_dii_flows` | `tools/macro/fii_dii.py` | FII & DII net flows (NSE); `None` if unavailable |
| `get_crude_price` | `tools/macro/crude.py` | Brent crude, trend, and `impact_on_india` |
| `get_us_market_summary` | `tools/macro/us_market.py` | S&P 500, Nasdaq, Dow |
| `get_india_market_summary` | `tools/macro/india_market.py` | Nifty 50 and Sensex: level, previous close, change |
| `get_usd_inr_rate` | `tools/macro/usd_inr.py` | USD/INR exchange rate |
| `get_inflation_data` | `tools/macro/inflation.py` | Recent CPI / WPI news articles (35-day window) |
| `get_gold_price` | `tools/macro/gold.py` | Gold price and trend |
| `get_rbi_updates` | `tools/macro/rbi.py` | Live repo rate, SDF, MSF, bank rate, CRR, SLR from rbi.org.in |
| `search_macro_news_tavily` | `tools/tavily/macro_news.py` | Tavily news search for macro news |

**Output model:** `MacroSummary` (`src/models/macro.py`)

---

### 2. `CompanyNewsAgent`
**File:** `src/agents/company_news_agent.py`

**What it does:**
Collects news for a single company from 7 sources simultaneously, deduplicates by URL and
title+source, scores each article by relevance and source tier, and returns the top 8. News sources
are searched by **company name**; exchange-filing sources by **NSE symbol**.

**Uses LLM?** ❌ No — fully deterministic keyword scoring.

**Scoring rules** (keywords match whole words only, so "fund" does not match "refund"):
- `+10` for earnings/M&A keywords (`"earnings"`, `"results"`, `"acquisition"` …)
- `+5` if the article names the company (its ticker, or the first two words of its name)
- `+10` for `source_type == "official"` (NSE filings), `+9` for tier1, `+7` for tier2
- `−5` for generic market noise (`"nifty"`, `"sensex"`, `"stocks in news"`)
- Articles with negative final scores are dropped; max 3 per source

**Tools (all run in parallel via `ThreadPoolExecutor`):**

| Tool function | File | Data fetched |
|---|---|---|
| `search_reuters_news` | `tools/company/reuters.py` | Reuters headlines, via Google News RSS |
| `search_moneycontrol_news` | `tools/company/moneycontrol.py` | Moneycontrol tag page (unknown tags return nothing) |
| `search_mint_news` | `tools/company/mint.py` | Mint headlines, via Google News RSS |
| `search_economic_times_news` | `tools/company/economic_times.py` | Economic Times topic page |
| `get_nse_announcements` | `tools/company/nse.py` | NSE corporate announcements (last 7 days, routine filings skipped) |
| `get_investor_relations` | `tools/company/investor_relations.py` | Press releases, results, investor meets from NSE filings (30 days) |
| `search_company_news_tavily` | `tools/tavily/company_news.py` | Tavily news search (3-day window, enforced in code) |

**Output model:** `CompanyNews` with a list of `NewsArticle` (`src/models/company.py`)

---

### 3. `MarketDataAgent`
**File:** `src/agents/market_data_agent.py`

**What it does:**
Fetches price, technical indicators, and sector performance for a single ticker. Tickers are mapped
to Yahoo symbols (`TMPV` → `TMPV.NS`). The sector and industry come from Yahoo's search endpoint and
are mapped to the matching Nifty sector index (e.g. Auto Manufacturers → Nifty Auto).

**Uses LLM?** ❌ No — fully deterministic.

**Tools (all run in parallel via `ThreadPoolExecutor`):**

| Tool function | File | Data fetched |
|---|---|---|
| `get_stock_price` | `tools/market/price.py` | Current price, previous close, % day change |
| `get_technical_indicators` | `tools/market/indicators.py` | RSI(14, Wilder), MACD(12,26,9) + signal, 20-session VWAP |
| `get_sector_performance` | `tools/market/sector.py` | Day move of the matching Nifty sector index |

**Output model:** `MarketData` (`src/models/market.py`)

---

### 4. `FinancialDataAgent`
**File:** `src/agents/financial_data_agent.py`

**What it does:**
Retrieves structured fundamental financial data for a ticker from an external Financial Agent API.
The API client is currently a **stub** (no provider configured), so this returns no figures and the
report omits the "Financial Context" row.

**Uses LLM?** ❌ No — fully deterministic.

**Tools (all run in parallel via `ThreadPoolExecutor`):**

| Tool function | File | Data fetched |
|---|---|---|
| `get_company_profile` | `tools/financial_api/company_profile.py` | Company name, sector, description |
| `get_financial_metrics` | `tools/financial_api/financial_metrics.py` | Market cap, P/E, EPS, revenue, debt/equity |
| `get_company_events` | `tools/financial_api/company_events.py` | Upcoming earnings, AGMs, corporate events |

**Output model:** `CompanyFinancials` (`src/models/financial_data.py`)

---

### 5. `RedditSentimentAgent`
**File:** `src/agents/reddit_sentiment_agent.py`

**What it does:**
Fetches Reddit posts about a company and computes a weighted community sentiment signal
(Bullish / Bearish / Neutral) using keyword heuristics. Each post is weighted by its Reddit score.
Skipped when Reddit credentials are not configured.

**Uses LLM?** ❌ No — fully deterministic keyword heuristics.

**Sentiment computation** (whole-word matches only, so "calls" does not match "recalls"):
- **Bullish keywords:** `buy, bull, calls, moon, undervalued, breakout, long, hold, profit`
- **Bearish keywords:** `sell, bear, puts, overvalued, crash, short, dump, loss`
- `avg_score > 0.2` → Bullish | `< −0.2` → Bearish | otherwise → Neutral

**Tools:**

| Tool function | File | Data fetched |
|---|---|---|
| `search_company_reddit` | `tools/reddit/company_posts.py` | Top Reddit posts mentioning the company (via PRAW) |

**Output model:** `RedditSignal` with a list of `RedditPost` (`src/models/reddit.py`)

---

### 6. `HistoricalContextAgent`
**File:** `src/agents/historical_context_agent.py` (statistics in `src/rag/history_stats.py`)

**What it does:**
Puts today's session in context using the 20-year price-history store. For each company it reads that
company's chunks **exactly, by ticker and level**, not by text similarity, and computes up to four kinds
of line. Every figure is arithmetic on the stored prices and returns, done in code, so each line can be
quoted as fact.

| Line | Built from | What it says |
|---|---|---|
| **Comparable sessions** | the raw daily bars stored in every `month` chunk (the full 20-year close series) | Past sessions whose move was within ±25% of today's, same direction (a 3.5% fall matches falls of 2.6–4.4%): how many, across how many years, and how often the stock was higher 5 and 20 sessions later (and the median), **against the same share across all sessions** |
| **Precedent** (up to 3) | the same series, plus the matching `event` chunk when one exists | The closest past sessions by size of move, at most one per 20 sessions so they are separate episodes, with the 5- and 20-session follow-through. Sessions of 4% or more also show the NIFTY 50's move and the volume, read from their stored event chunk |
| **Seasonality** | `month` chunks for the current calendar month | In how many past Octobers (say) the stock rose, the median, and the best and worst year |
| **Year context** | `year` chunks | This year to date against the stock's own completed years (median, range) and the NIFTY 50, and in how many years it beat the index |

Rules that keep the numbers honest:
- The comparable-session and precedent lines appear only when today's move is **2% or more**; on a quiet
  day only seasonality and year context are produced.
- Fewer than **15** comparable sessions is reported as too few for statistics, not turned into a percentage.
- The part-year and part-month in which a stock listed are not counted as full years or months.
- If the store's last session is more than 4 days old, a note says so and to re-run the ingest.
- The baseline matters: a stock being "higher after 52% of such sessions" means little if it is higher
  after 51% of all sessions, and the line says both.

Uses no embedding model, so it adds no model loading to a pipeline run (about 1–2 seconds per company).
Semantic search (`src/rag/retriever.py`) is still used by the `python -m src.rag.query` command.

**Uses LLM?** ❌ No. **Skipped** (returns nothing) when `QDRANT_URL` is not configured.

**Output:** `historical_context: dict[ticker, list[str]]` in graph state, shown in the report under
"Historical Context" and given to the ReportAgent and the FactCheckAgent as facts.

---

### 7. `SentimentAgent`
**File:** `src/agents/sentiment_agent.py`

**What it does:**
A pure **reasoning agent** — no tools. Takes the news, financial data and Reddit signal for a single
ticker, formats them into a compact prompt, and makes **one LLM call** to produce a structured
`SentimentResult`.

Per-source signals are pinned in code: `reddit_sentiment` is copied from the Reddit agent, and
`reddit_sentiment`, `financial_data_signal` and `verified_news_sentiment` are `None` whenever that
source supplied no data, whatever the model returned.

**Uses LLM?** ✅ Yes — 1 call per ticker.

**Inputs (from graph state):**

| Input | Produced by |
|---|---|
| `CompanyNews` | `CompanyNewsAgent` |
| `CompanyFinancials` | `FinancialDataAgent` |
| `RedditSignal` | `RedditSentimentAgent` |

**Output model:** `SentimentResult` (`src/models/sentiment.py`)

---

### 8. `ReportAgent`
**File:** `src/agents/report_agent.py`

**What it does:**
Assembles the final `DailyMarketReport`. The LLM writes **only the commentary** (`ReportNarrative`:
macro overview, per-company macro relevance and interpretation, final market view, catalysts, risks).
Everything else is copied in code from upstream output: sentiment labels and confidence, positive and
negative drivers, headlines, the market-data line, the day-over-day tables, historical passages,
the macro summary, and timestamps. This stops the model altering a figure by retyping it.

If the model leaves a company out of its commentary it is asked once more; if the LLM fails entirely,
a facts-only report is still built.

**Uses LLM?** ✅ Yes — 1 call (2 if a retry is needed).

**Inputs:**

| Input | Produced by |
|---|---|
| `MacroSummary` | `MacroAgent` |
| `company_news` dict | `CompanyNewsAgent` |
| `market_data` dict | `MarketDataAgent` |
| `financial_data` dict | `FinancialDataAgent` |
| `reddit_signals` dict | `RedditSentimentAgent` |
| `sentiments` dict | `SentimentAgent` |
| `historical_context` dict | `HistoricalContextAgent` |
| `market_changes`, `run_changes` | `ChangeDetectionService` (see below) |

**Output model:** `DailyMarketReport` (`src/models/report.py`)

---

### 9. `FactCheckAgent`
**File:** `src/agents/fact_check_agent.py`

**What it does:**
The last step before the report is final. The commentary is the only free text in the report, so it
is the only place a wrong statement can still enter. This agent gives a second model the **same facts
the ReportAgent wrote from** (`build_report_facts()` in `report_agent.py`) and the commentary, split
into numbered sentences, and asks which sentences the facts do not support.

- **What is checked:** the macro overview, each company's macro relevance and interpretation, the final
  market view (each split into sentences), and every major catalyst and risk.
- **What gets flagged:** a sentence that contradicts the facts; that states a figure, date, rating,
  event or indicator not in the facts; or that attributes a fact to the wrong company. Interpretation,
  hedged reasoning and general economic logic are explicitly not flagged. Risks and catalysts are
  forward-looking by nature, so one is flagged only for a specific figure, date, rating or past event
  that the facts contradict or lack, not for naming something that might happen.
- **What happens to a flagged sentence:** it is **deleted in code**, never rewritten, so the check
  cannot introduce a new claim. Each removal and its reason is recorded in the report's
  `removed_claims` (an audit trail; not shown in the PDF) and printed in the run log.
- **Different model:** the check runs on the Groq *fallback* model (`get_checker_llm()`), not the one
  that wrote the commentary, so it is less likely to repeat the same misreading.
- **If a field is emptied:** a company's interpretation falls back to its sentiment summary and the
  macro overview to the macro summary.
- **If the check itself fails:** the report is returned unchanged.

**Uses LLM?** ✅ Yes — 1 call (none if the report has no commentary).

**Output model:** `DailyMarketReport` (the same report, with unsupported commentary removed)

---

## Day-over-Day Comparison ("What Changed")

Every report carries a "What Changed" section with two tables, built entirely in code by
`ChangeDetectionService` (`src/services/change_detection.py`). No LLM is involved; the LLM is shown
the rows so its commentary can refer to them.

**1. Markets — versus the previous close.** Always available; needs no stored history.

| Row | Source |
|---|---|
| Nifty 50, Sensex | `get_india_market_summary` |
| Gold, Brent crude, USD/INR | macro tools |
| S&P 500, Nasdaq, Dow Jones | `get_us_market_summary` |
| Each watchlist stock | `MarketDataAgent` |

Each row shows previous close, current value, and the absolute and percentage move.

**2. Signals — versus the previous run.** Needs a saved earlier run.

| Row | What is compared |
|---|---|
| Macro sentiment | label and confidence |
| FII and DII net flow | the figure, or "no new figure" if NSE's publication date is unchanged |
| RBI repo rate | change in percentage points |
| Each company's sentiment | label change ("Bearish to Bullish") or confidence change |

**Run history** (`src/services/historical_service.py`): after each run, `run_pipeline.py` writes a
small JSON snapshot to `data/history/<date>.json` holding the macro sentiment, FII/DII flows, repo
rate and each company's sentiment. The next run loads the most recent snapshot dated **before**
today. A second run on the same day overwrites that day's file and is still compared with the
previous day. Prices are not stored, because their previous value comes from the market data itself.

On the first ever run there is no earlier snapshot, so only the Markets table is shown.

---

## Price-History RAG — `src/rag/`

A retrieval-augmented store of **20 years of daily price history for the NIFTY 50**, held in
**Qdrant**. It answers questions about how a stock behaved in the past, and supplies the
`HistoricalContextAgent` with precedents for today's session.

```
yfinance ──► loader ──► chunking ──► embeddings ──► Qdrant
                                                      │
question ──► embed query ──► hybrid search (dense + BM25) ──► RRF fusion ──► cross-encoder rerank ──► passages
                                                                                                        │
                                                                              report / LLM answer ◄─────┘
```

### Data

| | |
|---|---|
| **Universe** | Current NIFTY 50 constituents, fetched live from NSE's published list, plus any watchlist stock outside the index |
| **Source** | Yahoo Finance via `yfinance`, daily bars (open, high, low, close, volume) |
| **Span** | Up to 20 years; less for companies listed more recently (e.g. Adani Ports from Nov 2007) |
| **Price basis** | Adjusted for splits and dividends, so returns are comparable across the whole span. Old prices therefore differ from the prices quoted at the time |
| **Benchmark** | NIFTY 50 (`^NSEI`) history, used for the "versus the index" sentences |

Because the universe is *today's* index membership, companies that left the index are not covered.

### Chunking: calendar-window, hierarchical, no overlap

Raw price rows are not embedded. A row of numbers carries no meaning for an embedding model, and
5,000 near-identical rows per stock would drown retrieval. Instead `src/rag/chunking.py` splits each
stock's history **along the calendar** into three levels of self-contained summary:

| Level | One chunk per | What the text states | Parent |
|---|---|---|---|
| `year` | stock × calendar year | return, closing price, high/low with dates, largest drawdown, months up, best and worst month, average volume, return vs NIFTY 50 | — |
| `month` | stock × calendar month | return, open/close, high/low with dates, sessions, best and worst session, average daily move, average volume, distance from 52-week high/low, return vs NIFTY 50 | its `year` |
| `event` | session that moved **≥ 4%** | the move, close vs previous close, volume vs its 20-session average, the next 5 and 20 sessions, NIFTY 50 move that day | its `month` |

Design choices:

- **Split by meaning, not by size.** This is structure-aware chunking on calendar boundaries rather
  than fixed-size or recursive character splitting. Each chunk is one paragraph of roughly 300–700
  characters (about 80–170 tokens), far below the embedding model's 512-token limit, so nothing is
  ever truncated or cut mid-thought.
- **No overlap.** Overlapping windows exist to preserve context across arbitrary cut points. Here
  the cut points are not arbitrary and the context comes from the hierarchy instead: each chunk
  stores its `parent_id` (event → month → year).
- **Template-generated text.** Every figure is computed in code with pandas and written into a fixed
  sentence template that names the company, ticker and period. No LLM writes or summarises the
  chunks, so a retrieved passage can be quoted in the report as fact.
- **Raw data is kept.** Each `month` chunk's payload holds its daily bars (`sessions`), so the full
  daily series is in Qdrant for exact lookups even though only the summary is embedded.
- **Periods in progress** are labelled ("month to date, through 08 Oct 2026") and flagged
  `in_progress`.
- **Deterministic ids** (UUID5 of ticker + level + period) make ingestion idempotent: re-running it
  overwrites a period's point rather than duplicating it.

Example `event` chunk:

> Adani Ports and Special Economic Zone Ltd. (ADANIPORTS) on NSE, 09 Mar 2020 (Monday). The stock
> fell 5.7% in a single session, closing at Rs 307.76 against Rs 326.40 the session before. Volume
> was 1.3 times its 20-session average. Over the following 5 sessions it moved -16.1%, and over the
> following 20 sessions -20.9%. The NIFTY 50 moved -4.9% on the same day.

Measured size for the full NIFTY 50 (October 2026): **25,497 chunks** — 956 year, 10,933 month and
13,608 event — or about 510 per stock. Event counts vary widely with volatility, from 44 to 710 per
stock (median 256). Embedding them locally takes roughly 10 minutes.

**Payload fields** (all levels): `ticker`, `company`, `industry`, `granularity`, `year`, `month`,
`period_start`, `period_end`, `in_progress`, `return_pct`, `close`, `index_return_pct`, `parent_id`,
`price_basis`, `source`, `text`. Level-specific: `open`/`high`/`low`/`sessions` (month),
`high`/`low`/`max_drawdown_pct` (year), `prev_close`/`volume_ratio`/`fwd_5d_pct`/`fwd_20d_pct` (event).
`ticker`, `granularity` and `year` have payload indexes for filtering.

### Embeddings

Both run **locally** through `fastembed` (ONNX on CPU). No embedding API key is needed.

| Vector | Model | Role |
|---|---|---|
| `dense` | `BAAI/bge-small-en-v1.5` (384-dim, cosine) | Semantic match: "sharp sell-off", "outperformed the market" |
| `bm25` | `Qdrant/bm25` sparse vectors, IDF applied by Qdrant | Exact match on tickers, company names, months and years — which dense vectors blur ("March 2020" vs "March 2021") |

### Retrieval and reranking

Retrieval is **two-stage**, in `src/rag/retriever.py`. There are two rerankers, one in each stage:

| Stage | Reranker | Type | What it does |
|---|---|---|---|
| 1. Fusion | **Reciprocal Rank Fusion (RRF)** | Rank fusion, run inside Qdrant | Qdrant runs the dense search and the BM25 search separately (30 candidates each, same payload filter) and merges the two ranked lists by rank position. Needs no score calibration between the two searches |
| 2. Rerank | **`Xenova/ms-marco-MiniLM-L-6-v2`** | Cross-encoder, run locally via `fastembed` | Reads the query and each fused candidate **together** and scores their relevance directly, which is more accurate than comparing separately-made embeddings. The top results (3 for the pipeline, 5 for the CLI) are kept |

Payload filters (`ticker`, `granularity`) are applied inside Qdrant before fusion, so the pipeline's
lookups only ever see the company in question.

**Known limit:** the rerankers judge text relevance, not arithmetic. A superlative such as
"worst year for Adani Ports" returns relevant year chunks but not reliably the numerically worst
one; such questions are better answered by sorting on the `return_pct` payload.

### Two ways to read the store

| | Used by | How it picks chunks |
|---|---|---|
| **Semantic search** (`retriever.py`): dense + BM25, RRF, cross-encoder rerank | `python -m src.rag.query` | Meaning and wording of a free-text question |
| **Exact reads** (`reader.py`): scroll by `ticker` + `granularity`, fetch by id | `HistoricalContextAgent` in the pipeline | Payload fields, so the pipeline's statistics use *all* of a company's data, not the top few matches |

The pipeline uses exact reads because its questions are numeric ("sessions like today's") and an
embedding model matches the wording of a chunk, not the size of its number.

### Commands

```bash
python -m src.rag.ingest                              # NIFTY 50, 20 years
python -m src.rag.ingest --tickers TMPV,ADANIPORTS    # selected stocks
python -m src.rag.ingest --recreate                   # drop and rebuild the collection

python -m src.rag.query "How did Adani Ports do in March 2020?"
python -m src.rag.query "sharpest one-day falls" --ticker TMPV --granularity event
python -m src.rag.query "best months" --ticker TMPV --no-answer    # passages only, no LLM
```

`query` retrieves and reranks passages, then asks the Groq LLM to answer **only from those passages**
(`src/prompts/history.py`).

---

## LLM Gateway — `src/llm/gateway.py`

All LLM calls across the entire system route through a single `LLMGateway` singleton. Every agent
gets a `RateLimitedLLM`, which wraps an ordered **fallback chain** of models:

| Order | Model | Key | Notes |
|---|---|---|---|
| 1 | Groq primary (`PRIMARY_MODEL`) | `GROQ_API_KEY` | |
| 2 | Groq fallback (`FALLBACK_MODEL`, a different model) | `GROQ_FALLBACK_API_KEY` | Covers a model outage as well as a rate-limited key |
| 3 | Claude (`ANTHROPIC_FALLBACK_MODEL`, default `claude-opus-5-5`) | `ANTHROPIC_FALLBACK_API_KEY` | Only in the chain when the key is set; used when both Groq models fail, e.g. the Groq quota has run out |

The `FactCheckAgent` uses `get_checker_llm()`, which swaps the first two (Groq fallback model first)
so the check is made by a different model from the writer.

For each call, `RateLimitedLLM`:

- Tracks per-agent call counts in `llm_usage_stats`
- Retries a Groq model up to **3 times** on 429 / 503 errors with exponential back-off + jitter
- Moves to the next model in the chain if that still fails
- Puts a failed model on a **60-second cooldown**, so later calls skip straight past it instead of
  waiting through its retries again
- **Raises** if every model fails; the graph nodes catch the error and substitute a placeholder result

Call `print_usage_stats()` at the end of a run to see a per-agent breakdown.

**Claude adapter** (`src/llm/anthropic_fallback.py`): the agents call `.invoke()` on LangChain-style
models, so the adapter exposes the same call and returns an `AIMessage`, while making the request with
the official `anthropic` SDK. It sends no `temperature` and no `thinking` field (current Claude models
reject sampling parameters and think adaptively by default), sets `effort` to `medium`, opts into
Anthropic's server-side refusal fallback, and treats a refusal or a response cut off at `max_tokens`
as an error rather than an answer. The SDK does its own retrying, so the gateway does not retry it again.

---

## Key Configuration — `config/settings.py`

| Setting | Description |
|---|---|
| `GROQ_API_KEY` | Primary Groq API key |
| `GROQ_FALLBACK_API_KEY` | Fallback Groq API key |
| `PRIMARY_MODEL` / `FALLBACK_MODEL` | Groq model names |
| `ANTHROPIC_FALLBACK_API_KEY` | Anthropic API key for the last-resort Claude fallback (optional) |
| `ANTHROPIC_FALLBACK_MODEL` | Claude model id (default `claude-opus-5-5`) |
| `ANTHROPIC_MAX_TOKENS` | Output cap for Claude calls, covering thinking and answer (default 16000) |
| `TAVILY_API_KEY` | Tavily news-search API key |
| `FINANCIAL_AGENT_API_KEY` + `FINANCIAL_AGENT_BASE_URL` | External financial data provider (stub) |
| `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` | Reddit PRAW OAuth credentials (optional) |
| `QDRANT_URL` + `QDRANT_API_KEY` | Qdrant cluster for the price-history RAG (optional) |
| `QDRANT_COLLECTION` | Collection name (default `nifty_price_history`) |
| `DATABASE_URL` | PostgreSQL connection string (optional) |

Reddit, Qdrant, PostgreSQL and the Anthropic fallback are each optional: leave them unset and the
pipeline skips that part.

---

## Agent Summary Table

| Agent | LLM? | # Tools | Output Model |
|---|---|---|---|
| `MacroAgent` | ✅ 1 call | 9 (7 macro + inflation news + Tavily) | `MacroSummary` |
| `CompanyNewsAgent` | ❌ | 7 (6 sources + Tavily) | `CompanyNews` |
| `MarketDataAgent` | ❌ | 3 (price, indicators, sector) | `MarketData` |
| `FinancialDataAgent` | ❌ | 3 (profile, metrics, events) | `CompanyFinancials` |
| `RedditSentimentAgent` | ❌ | 1 (Reddit posts) | `RedditSignal` |
| `HistoricalContextAgent` | ❌ | RAG retriever (hybrid + rerank) | `list[str]` per ticker |
| `SentimentAgent` | ✅ 1 call/ticker | 0 (reasoning only) | `SentimentResult` |
| `ReportAgent` | ✅ 1 call total | 0 (commentary only) | `DailyMarketReport` |
| `FactCheckAgent` | ✅ 1 call total | 0 (checks the commentary) | `DailyMarketReport` |
