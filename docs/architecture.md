# System Architecture

## Overview
Market Intelligence Agent is a multi-agent system designed for automated financial and market analysis synthesis.

## Core Components
- **Agents**: Macro, Company News, Market Data, Sentiment, and Synthesis Report agents.
- **Graph Workflow**: State graph orchestration managing multi-agent workflow.
- **LLM Gateway**: Primary and fallback LLM routing via LangChain/Groq.
- **Tools**: Specialized fetching and parsing tools for Macro, Company, Market, and Common utilities.
- **Services & Storage**: Confidence estimation, historical analysis, change detection, and PostgreSQL storage.
