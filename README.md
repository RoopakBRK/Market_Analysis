# Market Analysis

A multi-agent financial intelligence system for **daily sentiment analysis of the Indian stock market**, focused on NIFTY 50 companies.

## What It Does

The system collects and analyzes:

* Company-specific financial news
* Indian macroeconomic and market news
* Regulatory and sector developments
* Global events affecting Indian markets

It evaluates the **sentiment and potential impact** of these events on individual NIFTY 50 companies.

## How It Works

The system uses specialized agents to:

1. Collect relevant company and macro news.
2. Identify important events and their affected companies.
3. Analyze positive, negative, and neutral sentiment.
4. Combine company-level signals with broader market context.
5. Generate a daily market intelligence report.

The system focuses on **daily news-driven sentiment**, not stock-price prediction, valuation, or technical analysis.

---

## Architecture

```
START
  ├── MacroAgent            → Macroeconomic signals + Tavily news
  ├── CompanyNewsAgent      → Parallel news retrieval (Reuters, ET, NSE, Tavily...)
  ├── MarketDataAgent       → Price, RSI, MACD, VWAP, sector index via Yahoo Finance
  ├── FinancialDataAgent    → PE, EPS, Revenue, upcoming events
  └── RedditSentimentAgent  → Community sentiment (r/IndiaInvestments etc.)
          │
          ▼
    SentimentAgent          → LLM reasoning over all signals
    HistoricalContextAgent  → Precedents from 20 years of price history (Qdrant RAG)
          │
          ▼
     ReportAgent            → Final daily market intelligence report
```

The LLM writes only the commentary. Every fact in the report — sentiment
labels, drivers, headlines, prices, timestamps — is copied in code from the
agents that collected it, so the model cannot alter a figure by retyping it.

The companies covered are set in `src/utils/constants.py` (`WATCHLIST`: NSE
symbol → company name).

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/RoopakBRK/Market_Analysis.git
cd Market_Analysis
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env             # if .env.example exists, otherwise create .env
```

Edit `.env` and fill in your keys:

```env
# LLM (Groq)
GROQ_API_KEY=your_groq_api_key
GROQ_FALLBACK_API_KEY=your_fallback_key

# Tavily — web/news retrieval
TAVILY_API_KEY=your_tavily_key

# Financial Agent API (optional — stub until provider is confirmed)
FINANCIAL_AGENT_API_KEY=your_key
FINANCIAL_AGENT_BASE_URL=

# Reddit API (optional — pipeline continues without it)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=MarketAnalysisBot/1.0

# Qdrant — price-history RAG (optional — pipeline continues without it)
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# PostgreSQL (optional — pipeline continues without it)
DATABASE_URL=postgresql://user:password@localhost:5432/market_analysis
```

### 5. Database (optional)

Set `DATABASE_URL` and each run is saved to PostgreSQL; the tables are
created on the first run. Leave it unset (or unreachable) and the pipeline
simply skips saving.

---

## Running the Pipeline

Run it either as a script or as a module — both work from the repository root:

```bash
python apps/worker/run_pipeline.py
# or
python -m apps.worker.run_pipeline
```

The pipeline will:
- Collect macro + company intelligence in parallel
- Synthesise sentiment per company
- Print the full daily report to stdout
- Write a formatted PDF report to `reports/market_report_<date>.pdf`
- Save a snapshot to `data/history/<date>.json`, used by the next day's
  "What Changed" comparison
- Save the run to PostgreSQL (if `DATABASE_URL` is set and reachable). A
  second run on the same day replaces that day's report.

A single run takes a few minutes (data collection is paced with short
delays between companies, plus real LLM calls for sentiment + report
synthesis).

### What Changed (day-over-day)

Every report has a "What Changed" section. Price moves (Nifty 50, Sensex,
gold, crude, USD/INR, US indices, each watchlist stock) are compared with the
previous close. Sentiment, FII/DII flows and the RBI repo rate are compared
with the previous run, read from `data/history/`; on the very first run there
is no earlier snapshot, so that second table starts the next day.

### Price-history RAG (optional)

The pipeline can add historical precedent to each company's section from a
Qdrant store holding 20 years of NIFTY 50 price history. To set it up:

```bash
# 1. Put QDRANT_URL and QDRANT_API_KEY in .env
# 2. Build the store (downloads history with yfinance; embeddings run locally)
python -m src.rag.ingest
# 3. Ask it something
python -m src.rag.query "How did Adani Ports do in March 2020?"
```

Without `QDRANT_URL` the pipeline logs `[RAG] Skipped` and runs as before.
The chunking, embedding and reranking design is described in `structure.md`.

### Reddit is optional

If `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` are left blank in `.env`,
the pipeline logs `[Reddit] Skipped — REDDIT_CLIENT_ID/SECRET not
configured.` and continues normally — Reddit sentiment is simply reported
as `Unknown` for that run. No crash, no manual flag needed.

---

## Running Tests

```bash
pytest tests/ -v --tb=short      # offline tests: no network, no API keys
pytest tests/ -m live -v -s      # live checks: real APIs, needs the keys in .env
```

> The default run is fully offline — external calls are mocked. Tests that
> call real services (scrapers, Tavily, the LLM, the full pipeline) are
> marked `live` and skipped unless you ask for them with `-m live`.

---

## Quick Reference — Terminal Commands

```bash
# 1. Activate the virtual environment (from the repo root)
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 2. Install / update dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (prints report to stdout + writes a PDF to reports/)
python apps/worker/run_pipeline.py

# 4. Run the test suite (offline; add `-m live` for the live checks)
pytest tests/ -v --tb=short

# 5. Open the most recently generated PDF report
open reports/market_report_$(date +%Y-%m-%d).pdf     # macOS
# xdg-open reports/market_report_$(date +%Y-%m-%d).pdf  # Linux
# start reports\market_report_%date%.pdf                # Windows

```

---

## Project Structure

```
Market_Analysis/
├── apps/
│   └── worker/
│       └── run_pipeline.py       # Pipeline entry point
├── config/
│   └── settings.py               # Pydantic settings (env-driven)
├── src/
│   ├── agents/                   # 7 intelligent agents
│   ├── graph/                    # LangGraph workflow + state
│   ├── models/                   # Pydantic domain models
│   ├── prompts/                  # LLM system prompts
│   ├── rag/                      # Price-history RAG (yfinance → Qdrant)
│   ├── services/                 # PDF rendering, day-over-day comparison, run history
│   ├── storage/                  # PostgreSQL persistence layer
│   ├── tools/                    # Data retrieval tools (Tavily, Reddit, NSE...)
│   └── llm/                      # LLM gateway + usage tracking
├── tests/                        # Unit tests
├── .env                          # Local credentials (never commit)
└── requirements.txt
```
