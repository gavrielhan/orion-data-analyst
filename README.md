# Orion - Data Analysis LangGraph Agent

An AI-powered Data Analysis Agent that connects to Google BigQuery's public e-commerce dataset and performs intelligent data exploration through natural language interaction.

🔗 **Repository**: https://github.com/gavrielhan/orion-data-analyst

## Overview

Orion is an intelligent business analyst that:
- Connects to BigQuery's `thelook_ecommerce` dataset
- Generates dynamic SQL queries from natural language
- Performs statistical analysis and data visualization
- Provides actionable business insights

## Architecture

Built with **LangGraph**, Orion uses a modular node-based architecture:

```
User Query → InputNode → QueryBuilderNode → BigQueryExecutorNode → OutputNode
```

Each node handles a distinct analytical step, creating a directed graph of reasoning.

## Features (MVP)

✅ Natural language query processing  
✅ Dynamic SQL generation with Gemini  
✅ BigQuery integration  
✅ Basic result display  
✅ CLI interface  

## Setup

### Prerequisites

- Python 3.10+
- Google account for Google Cloud and Gemini API access

### Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Get your API keys** - Follow this guide:
   - **👉 [GETTING_KEYS.md](GETTING_KEYS.md) - Start here!** 
   
   Or see [SETUP.md](SETUP.md) for detailed setup instructions.

3. **Configure your `.env` file**:
```bash
cp .env.example .env
# Edit .env with your credentials
```

4. **Run Orion**:
```bash
python -m src.cli
```

## Usage

Start the interactive CLI:

```bash
python -m src.cli
```

Example queries:
- "What are total sales?"
- "Show me the number of orders by status"
- "List the top 10 products by revenue"

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py          # LangGraph orchestration
│   │   ├── nodes.py          # All agent nodes
│   │   └── state.py          # Agent state management
│   ├── config.py             # Configuration loader
│   └── utils/
│       ├── __init__.py
│       └── bigquery.py       # BigQuery utilities
├── tests/                     # Test suite
├── requirements.txt
├── .env.example
└── README.md
```

## Dataset

The project uses Google BigQuery's public e-commerce dataset:
- **Dataset**: `bigquery-public-data.thelook_ecommerce`
- **Tables**: orders, order_items, products, users

## Development

Run tests:
```bash
pytest tests/
```

## License

MIT

## Roadmap

- [x] Milestone 1: Foundation & Happy Path MVP
- [ ] Milestone 2: Validation & Error Handling
- [ ] Milestone 3: Advanced Analysis & Visualization
- [ ] Milestone 4: Conversation Memory
- [ ] Milestone 5: Production Readiness

