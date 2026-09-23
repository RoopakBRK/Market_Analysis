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
  ├── MarketDataAgent       → Price, RSI, MACD, VWAP via Yahoo Finance
  ├── FinancialDataAgent    → PE, EPS, Revenue, upcoming events
  └── RedditSentimentAgent  → Community sentiment (r/IndiaInvestments etc.)
          │
          ▼
    SentimentAgent          → LLM reasoning over all signals
          │
          ▼
     ReportAgent            → Final daily market intelligence report
```

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

# PostgreSQL (optional — pipeline continues without it)
DATABASE_URL=postgresql://user:password@localhost:5432/market_analysis
```

### 5. Initialise the database (optional)

```bash
python -c "from src.storage.postgres import init_db; init_db()"
```

---

## Running the Pipeline

```bash
python apps/worker/run_pipeline.py
```

The pipeline will:
- Collect macro + company intelligence in parallel
- Synthesise sentiment per company
- Print the full daily report to stdout
- Persist results to PostgreSQL (if `DATABASE_URL` is set)

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

> Tests do **not** require live API keys — all external calls are mocked.

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
│   ├── storage/                  # PostgreSQL persistence layer
│   ├── tools/                    # Data retrieval tools (Tavily, Reddit, NSE...)
│   └── llm/                      # LLM gateway + usage tracking
├── tests/                        # Unit tests
├── .env                          # Local credentials (never commit)
└── requirements.txt
```
