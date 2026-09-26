# Market Analysis — Codebase Structure

## Overview

This is a **multi-agent market intelligence system** built with **LangGraph** and **LangChain**.
Given a watchlist of stock tickers it spins up a directed acyclic graph of specialised agents that:

1. Run **in parallel** to collect data from diverse sources (news scrapers, APIs, Reddit, Tavily web search)
2. Converge into a **SentimentAgent** that synthesises per-ticker sentiment
3. Finally produce a single **DailyMarketReport** via the **ReportAgent**

The LLM backend is **Groq**. Most data-collection agents are **fully deterministic** (zero LLM calls). Only `MacroAgent`, `SentimentAgent`, and `ReportAgent` invoke an LLM.

---

## Directory Tree

```
Market Analysis/
│
├── apps/                            # Application entry points
│   ├── api/                         # API server (in progress)
│   │   ├── main.py                  # Entry point — loads .env, starts server
│   │   ├── dependencies.py          # FastAPI DI helpers (placeholder)
│   │   └── routes.py                # Route definitions (placeholder)
│   └── worker/                      # Background worker (placeholder)
│
├── config/                          # App-level configuration
│   ├── settings.py                  # Pydantic BaseSettings — all env vars & model names
│   └── prompts.py                   # Config-level prompt constants
│
├── src/                             # Core source package
│   │
│   ├── agents/                      # One class per agent
│   │   ├── company_news_agent.py
│   │   ├── financial_data_agent.py
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
│   ├── tools/                       # LangChain @tool-decorated data fetchers
│   │   ├── common/                  # Shared utility functions
│   │   │   ├── normalization.py     # utc_now_iso(), deduplicate_article_dicts()
│   │   │   ├── source_classifier.py # Classifies a source URL into a tier
│   │   │   ├── scraper_utils.py     # HTTP scraping helpers (headers, retries)
│   │   │   ├── yahoo_finance.py     # Yahoo Finance convenience wrappers
│   │   │   ├── citations.py         # Citation formatting helpers
│   │   │   ├── date_utils.py        # Date parsing / formatting helpers
│   │   │   └── deduplicate.py       # Article deduplication logic
│   │   │
│   │   ├── company/                 # Company-specific news scrapers
│   │   │   ├── economic_times.py    # Economic Times scraper
│   │   │   ├── investor_relations.py# Company IR page scraper
│   │   │   ├── mint.py              # Mint (HT Media) scraper
│   │   │   ├── moneycontrol.py      # Moneycontrol scraper
│   │   │   ├── nse.py               # NSE official announcements
│   │   │   └── reuters.py           # Reuters scraper
│   │   │
│   │   ├── financial_api/           # Structured financial data via external API
│   │   │   ├── client.py            # HTTP client wrapper
│   │   │   ├── company_profile.py   # Company name, sector, description
│   │   │   ├── financial_metrics.py # Market cap, P/E, EPS, revenue, D/E
│   │   │   └── company_events.py    # Upcoming earnings dates, AGMs, etc.
│   │   │
│   │   ├── macro/                   # Macro-economic data fetchers
│   │   │   ├── crude.py             # Crude oil spot price
│   │   │   ├── fii_dii.py           # FII / DII institutional flows
│   │   │   ├── gold.py              # Gold spot price
│   │   │   ├── inflation.py         # CPI / WPI inflation data
│   │   │   ├── rbi.py               # RBI policy announcements
│   │   │   ├── us_market.py         # US index snapshot (S&P 500, NASDAQ…)
│   │   │   └── usd_inr.py           # USD/INR exchange rate
│   │   │
│   │   ├── market/                  # Per-stock market data
│   │   │   ├── price.py             # Current price, previous close, % change
│   │   │   ├── indicators.py        # RSI, MACD, VWAP
│   │   │   └── sector.py            # Sector-level performance
│   │   │
│   │   ├── reddit/                  # Reddit (PRAW) tools
│   │   │   ├── client.py            # Authenticated Reddit client
│   │   │   ├── company_posts.py     # Posts mentioning a specific company
│   │   │   └── market_posts.py      # General market/investing posts
│   │   │
│   │   └── tavily/                  # Tavily AI web-search tools
│   │       ├── client.py            # Tavily HTTP client
│   │       ├── company_news.py      # Company-specific news search
│   │       ├── macro_news.py        # Macro-level news search
│   │       └── sector_news.py       # Sector-level news search
│   │
│   ├── models/                      # Pydantic output models
│   │   ├── company.py               # CompanyNews, NewsArticle
│   │   ├── financial_data.py        # CompanyFinancials
│   │   ├── macro.py                 # MacroSummary
│   │   ├── market.py                # MarketData
│   │   ├── reddit.py                # RedditSignal, RedditPost
│   │   ├── report.py                # DailyMarketReport
│   │   ├── sentiment.py             # SentimentResult
│   │   └── state.py                 # Misc shared state models
│   │
│   ├── prompts/                     # System-prompt strings (no logic)
│   │   ├── company.py
│   │   ├── macro.py
│   │   ├── market.py
│   │   ├── report.py
│   │   └── sentiment.py
│   │
│   ├── llm/                         # LLM abstraction layer
│   │   ├── gateway.py               # LLMGateway, RateLimitedLLM, get_llm()
│   │   └── structured_output.py     # Structured output helpers
│   │
│   ├── storage/                     # Persistence layer (PostgreSQL)
│   │   ├── postgres.py              # SQLAlchemy engine & session factory
│   │   ├── models.py                # ORM table definitions
│   │   └── repositories.py         # CRUD repositories
│   │
│   ├── services/                    # Business-logic services
│   │   ├── change_detection.py      # Detects significant changes between runs
│   │   ├── confidence_service.py    # Confidence score aggregation
│   │   ├── historical_service.py    # Historical comparison (WIP)
│   │   └── report_service.py        # Report persistence / retrieval (WIP)
│   │
│   ├── guardrails/                  # Output validation guards
│   │   ├── report_guard.py          # Guards on DailyMarketReport fields
│   │   └── sentiment_guard.py       # Guards on SentimentResult fields
│   │
│   └── observability/               # Telemetry
│       ├── logging.py               # Structured logging setup
│       ├── metrics.py               # Prometheus-style metrics
│       ├── tracing.py               # OpenTelemetry tracing
│       └── cost_tracker.py          # Per-agent LLM cost tracking
│
├── tests/                           # Test suite
├── docker/                          # Docker / Compose files
├── docs/                            # Additional documentation
├── .env                             # Environment variables (secrets)
├── pyproject.toml                   # Project metadata
└── requirements.txt                 # Python dependencies
```

---

## Execution Graph

```
START
  │
  ├──► MacroAgent              ← runs once, parallel
  ├──► CompanyNewsAgent        ← runs per ticker, parallel
  ├──► MarketDataAgent         ← runs per ticker, parallel
  ├──► FinancialDataAgent      ← runs per ticker, parallel
  └──► RedditSentimentAgent    ← runs per ticker, parallel
             │
             ▼   (all five branches must complete)
         SentimentAgent        ← LLM call per ticker
             │
             ▼
         ReportAgent           ← single LLM call
             │
            END
```

Defined in `src/graph/workflow.py`. The graph uses a single shared `GraphState` TypedDict
(`src/graph/state.py`) to pass data between nodes.

---

## Agent Reference

---

### 1. `MacroAgent`
**File:** `src/agents/macro_agent.py`

**What it does:**
Collects all macroeconomic signals (commodity prices, FII/DII flows, FX rates, US markets, inflation,
RBI updates, and web news) in parallel. Then makes **one LLM call** to synthesise everything into a
structured `MacroSummary` with an overall market sentiment, confidence score, key drivers, and market events.

**Uses LLM?** ✅ Yes — 1 call to format raw data into structured JSON.

**Tools (all run in parallel via `ThreadPoolExecutor`):**

| Tool function | File | Data fetched |
|---|---|---|
| `get_fii_dii_flows` | `tools/macro/fii_dii.py` | Foreign & domestic institutional investor flows |
| `get_crude_price` | `tools/macro/crude.py` | Crude oil spot price |
| `get_us_market_summary` | `tools/macro/us_market.py` | US index levels (S&P 500, NASDAQ, Dow) |
| `get_usd_inr_rate` | `tools/macro/usd_inr.py` | USD/INR exchange rate |
| `get_inflation_data` | `tools/macro/inflation.py` | CPI / WPI inflation figures |
| `get_gold_price` | `tools/macro/gold.py` | Gold spot price |
| `get_rbi_updates` | `tools/macro/rbi.py` | RBI policy announcements |
| `search_macro_news_tavily` | `tools/tavily/macro_news.py` | Tavily AI web-search for macro news |

**Output model:** `MacroSummary` (`src/models/macro.py`)

---

### 2. `CompanyNewsAgent`
**File:** `src/agents/company_news_agent.py`

**What it does:**
Scrapes news articles for a single company from 7 sources simultaneously, deduplicates by URL and
title+source, scores each article by relevance and source tier, and returns the top 8.

**Uses LLM?** ❌ No — fully deterministic keyword scoring.

**Scoring rules:**
- `+10` for earnings/M&A keywords (`"earnings"`, `"results"`, `"acquisition"` …)
- `+10` for `source_type == "official"` (NSE/IR), `+9` for tier1, `+7` for tier2
- `−5` for generic market noise (`"nifty"`, `"sensex"`, `"stocks in news"`)
- Articles with negative final scores are dropped; max 3 per source

**Tools (all run in parallel via `ThreadPoolExecutor`):**

| Tool function | File | Data fetched |
|---|---|---|
| `search_reuters_news` | `tools/company/reuters.py` | Reuters articles for the company |
| `search_moneycontrol_news` | `tools/company/moneycontrol.py` | Moneycontrol articles |
| `search_mint_news` | `tools/company/mint.py` | Mint (HT Media) articles |
| `search_economic_times_news` | `tools/company/economic_times.py` | Economic Times articles |
| `get_nse_announcements` | `tools/company/nse.py` | Official NSE exchange filings |
| `get_investor_relations` | `tools/company/investor_relations.py` | Company IR page announcements |
| `search_company_news_tavily` | `tools/tavily/company_news.py` | Tavily AI web-search fallback |

**Output model:** `CompanyNews` with a list of `NewsArticle` (`src/models/company.py`)

---

### 3. `MarketDataAgent`
**File:** `src/agents/market_data_agent.py`

**What it does:**
Fetches real-time price, technical indicators, and sector performance for a single ticker.
The sector name is **dynamically resolved** from Yahoo Finance's `quoteSummary` API before calling
the sector tool, so the right sector string is always passed.

**Uses LLM?** ❌ No — fully deterministic.

**Tools (all run in parallel via `ThreadPoolExecutor`):**

| Tool function | File | Data fetched |
|---|---|---|
| `get_stock_price` | `tools/market/price.py` | Current price, previous close, % day change |
| `get_technical_indicators` | `tools/market/indicators.py` | RSI, MACD, VWAP |
| `get_sector_performance` | `tools/market/sector.py` | Sector-level return / performance |

**Output model:** `MarketData` (`src/models/market.py`)

---

### 4. `FinancialDataAgent`
**File:** `src/agents/financial_data_agent.py`

**What it does:**
Retrieves structured fundamental financial data for a ticker from a dedicated external Financial Agent API.
Runs profile, metrics, and events calls in parallel then merges them.

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

**Uses LLM?** ❌ No — fully deterministic keyword heuristics.

**Sentiment computation:**
- **Bullish keywords:** `buy, bull, calls, moon, undervalued, breakout, long, hold, profit`
- **Bearish keywords:** `sell, bear, puts, overvalued, crash, short, dump, loss`
- `avg_score > 0.2` → Bullish | `< −0.2` → Bearish | otherwise → Neutral

**Tools:**

| Tool function | File | Data fetched |
|---|---|---|
| `search_company_reddit` | `tools/reddit/company_posts.py` | Top Reddit posts mentioning the ticker (via PRAW) |

**Output model:** `RedditSignal` with a list of `RedditPost` (`src/models/reddit.py`)

---

### 6. `SentimentAgent`
**File:** `src/agents/sentiment_agent.py`

**What it does:**
A pure **reasoning agent** — no tools. Takes the outputs of all four data-collection agents for a
single ticker, formats them into a compact prompt, and makes **one LLM call** to produce a structured
`SentimentResult`.

**Uses LLM?** ✅ Yes — 1 call per ticker.

**Inputs (from graph state):**

| Input | Produced by |
|---|---|
| `CompanyNews` | `CompanyNewsAgent` |
| `CompanyFinancials` | `FinancialDataAgent` |
| `RedditSignal` | `RedditSentimentAgent` |

**Output fields:** `sentiment`, `impact`, `confidence`, `summary`, `positive_drivers`,
`negative_drivers`, `verified_news_sentiment`, `financial_data_signal`, `reddit_sentiment`,
`articles_analyzed`, `source_breakdown`

**Output model:** `SentimentResult` (`src/models/sentiment.py`)

---

### 7. `ReportAgent`
**File:** `src/agents/report_agent.py`

**What it does:**
A pure **reasoning agent** — no tools. Combines all outputs from the entire graph and makes
**one LLM call** to generate the final `DailyMarketReport`. Raw `market_data` is excluded from the
macro payload to reduce token usage.

**Uses LLM?** ✅ Yes — 1 call total (not per ticker).

**Inputs (from graph state):**

| Input | Produced by |
|---|---|
| `MacroSummary` | `MacroAgent` |
| `company_news` dict | `CompanyNewsAgent` |
| `market_data` dict | `MarketDataAgent` |
| `financial_data` dict | `FinancialDataAgent` |
| `reddit_signals` dict | `RedditSentimentAgent` |
| `sentiments` dict | `SentimentAgent` |

**Output model:** `DailyMarketReport` (`src/models/report.py`)

---

## LLM Gateway — `src/llm/gateway.py`

All LLM calls across the entire system route through a single `LLMGateway` singleton:

- Configures a **primary** and **fallback** Groq model from `config/settings.py`
- Every agent call is wrapped in `RateLimitedLLM` which:
  - Tracks per-agent call counts in `llm_usage_stats`
  - Retries up to **3 times** on 429 / 503 errors with exponential back-off + jitter
  - Returns an empty `AIMessage` (instead of raising) on permanent failure so the graph keeps running
- Call `print_usage_stats()` at the end of a run to see a per-agent breakdown

---

## Key Configuration — `config/settings.py`

| Setting | Description |
|---|---|
| `GROQ_API_KEY` | Primary Groq API key |
| `GROQ_FALLBACK_API_KEY` | Fallback Groq API key |
| `PRIMARY_MODEL` / `FALLBACK_MODEL` | Groq model names |
| `TAVILY_API_KEY` | Tavily web-search API key |
| `FINANCIAL_AGENT_API_KEY` + `FINANCIAL_AGENT_BASE_URL` | External financial data provider |
| `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` | Reddit PRAW OAuth credentials |
| `DATABASE_URL` | PostgreSQL connection string |

---

## Agent Summary Table

| Agent | LLM? | # Tools | Output Model |
|---|---|---|---|
| `MacroAgent` | ✅ 1 call | 8 (7 macro + Tavily) | `MacroSummary` |
| `CompanyNewsAgent` | ❌ | 7 (6 scrapers + Tavily) | `CompanyNews` |
| `MarketDataAgent` | ❌ | 3 (price, indicators, sector) | `MarketData` |
| `FinancialDataAgent` | ❌ | 3 (profile, metrics, events) | `CompanyFinancials` |
| `RedditSentimentAgent` | ❌ | 1 (Reddit posts) | `RedditSignal` |
| `SentimentAgent` | ✅ 1 call/ticker | 0 (reasoning only) | `SentimentResult` |
| `ReportAgent` | ✅ 1 call total | 0 (reasoning only) | `DailyMarketReport` |
