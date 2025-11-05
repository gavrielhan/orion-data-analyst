# 🌟 Orion - AI-Powered Data Analysis Agent

[![PyPI version](https://badge.fury.io/py/orion-data-analyst.svg)](https://pypi.org/project/orion-data-analyst/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/🦜_LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/🦜_LangChain-0.3+-green.svg)](https://www.langchain.com/)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-BigQuery-4285F4?logo=google-cloud)](https://cloud.google.com/bigquery)
[![Gemini AI](https://img.shields.io/badge/Gemini_AI-2.0_Flash-8E75B2?logo=google)](https://ai.google.dev/)
[![Powered by AI](https://img.shields.io/badge/Powered_by-AI-orange.svg)](https://github.com/gavrielhan/orion-data-analyst)

An intelligent data analysis agent that transforms natural language questions into SQL queries, executes them on BigQuery, performs statistical analysis, and generates actionable business insights.

🔗 **GitHub**: https://github.com/gavrielhan/orion-data-analyst  
📦 **PyPI**: https://pypi.org/project/orion-data-analyst/

---

## ✨ What is Orion?

![Orion Interface](assets/orion_face.png)

Orion is your AI business analyst that:
- **Understands natural language** - Ask questions in plain English
- **Generates smart SQL** - Powered by Google Gemini AI
- **Analyzes data automatically** - Statistical analysis, trends, segmentation
- **Provides insights** - Actionable recommendations with business context
- **Creates visualizations** - Charts saved automatically
- **Self-heals errors** - Automatically fixes and retries failed queries
- **Remembers conversations** - Handles follow-up questions with context

Built with **LangGraph** for modular AI reasoning and **Google BigQuery** for data warehousing.

---

## 🚀 Quick Start

### Installation

**Option 1: Install from PyPI (Recommended)**
```bash
pip install orion-data-analyst
```

**Option 2: Install from Source**
```bash
git clone https://github.com/gavrielhan/orion-data-analyst.git
cd orion-data-analyst
pip install -e .
```

### Setup

1. **Get API Keys** (see [GETTING_KEYS.md](GETTING_KEYS.md)):
   - Google Cloud Project ID
   - Google Cloud service account JSON key
   - Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. **Configure `.env` file**:
```bash
# Copy example
cp .env.example .env

# Edit with your credentials
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GEMINI_API_KEY=your-gemini-api-key
```

3. **Run Orion**:
```bash
orion
```

---

## 💡 Usage Examples

### Basic Queries
```
You: show me top 10 products by revenue
Orion: [Generates SQL, executes, analyzes, and displays ranked results]

You: what are the sales trends for the last 6 months?
Orion: [Creates time-series analysis with month-over-month growth]

You: segment customers by purchase behavior
Orion: [Performs customer segmentation and analysis]
```

### Follow-up Questions
```
You: show top customers
Orion: [Displays ranked customer list]

You: show the same for the last quarter
Orion: [Uses conversation context to apply date filter]

You: break that down by region
Orion: [Further segments the previous results]
```

### Visualizations & Exports
```
You: create a bar chart of sales by category
Orion: [Generates chart and saves to ~/orion_results/]

You: save this as csv
Orion: [Exports results to ~/orion_results/results_TIMESTAMP.csv]
```

### Meta-Questions (Instant Responses)
```
You: what can you do?
Orion: [Explains capabilities without querying database]

You: which datasets can you query?
Orion: [Lists available tables and schemas]
```

---

## 🏗️ Architecture

Orion uses a **modular node-based architecture** powered by LangGraph:

### High-Level Architecture

![High-Level Schema](assets/high_level_schema.png)

### Detailed Graph Schema

![Graph Schema](assets/graph_schema.png)


See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed component descriptions.

---

## 🎯 Key Features

### 🤖 Intelligent SQL Generation
- Natural language to SQL using Google Gemini
- Automatic schema context injection
- Self-healing with error feedback loops (max 3 retries)
- Handles complex JOINs across multiple tables

### 🛡️ Safety & Validation
- Blocks malicious queries (DROP, DELETE, etc.)
- BigQuery cost estimation before execution
- Query syntax validation with dry-run
- Row limits to prevent runaway queries
- Human-in-the-loop approval for expensive operations

### 📊 Advanced Analytics
- **Ranking**: Top N analysis with contribution %
- **Trends**: Time-series with growth rates
- **Segmentation**: Group-by analysis
- **RFM Analysis**: Customer segmentation (Recency, Frequency, Monetary)
- **Anomaly Detection**: Outlier identification
- **Comparative Analysis**: Period-over-period comparison

### 💬 Conversation Memory
- Remembers last 5 interactions
- Context-aware follow-up questions
- Session save/load for long conversations
- Automatic context pruning for token efficiency

### 📈 Visualizations
- **Chart Types**: Bar, Line, Pie, Scatter, Box, Candle
- Auto-saved to `~/orion_results/` (configurable)
- Smart chart type selection based on data
- CSV export for further analysis

### ⚡ Performance Optimizations
- **Query Caching**: Instant responses for repeated queries (1-hour TTL)
- **Schema Caching**: Reduces API calls to BigQuery metadata
- **Rate Limiting**: Token bucket algorithm for Gemini API
- **Streaming**: Large result handling

### 🎨 Polished UX
- Colored terminal output with formatted text
- Progress indicators at each step
- Helpful error messages with action links
- Startup validation with setup guidance

---

## 🗂️ Project Structure

```
orion-data-analyst/
├── assets/                        # Images and diagrams
│   ├── orion_face.png            # Main interface screenshot
│   ├── high_level_schema.png     # High-level architecture diagram
│   └── graph_schema.png          # Detailed graph flow diagram
├── src/
│   ├── __init__.py
│   ├── cli.py                    # CLI interface with session management
│   ├── config.py                 # Configuration loader (.env)
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py              # LangGraph workflow orchestration
│   │   ├── nodes.py              # All 10 agent nodes
│   │   └── state.py              # Centralized AgentState (TypedDict)
│   └── utils/
│       ├── __init__.py
│       ├── cache.py              # Query result caching
│       ├── formatter.py          # ANSI terminal formatting
│       ├── rate_limiter.py       # API rate limiting
│       ├── schema_fetcher.py     # BigQuery schema utilities
│       └── visualizer.py         # Chart generation (matplotlib/seaborn)
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_nodes.py             # Node unit tests
│   └── test_graph.py             # Graph integration tests
├── .env.example                  # Configuration template
├── requirements.txt              # Dependencies
├── setup.py                      # PyPI packaging
├── pyproject.toml                # Modern Python packaging
├── install.sh                    # One-line installer
├── ARCHITECTURE.md               # Detailed architecture docs
├── GETTING_KEYS.md               # API key setup guide
└── README.md                     # This file
```

---

## ⚙️ Configuration

All configuration via `.env` file:

```bash
# REQUIRED
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GEMINI_API_KEY=your-gemini-api-key

# OPTIONAL
GEMINI_MODEL=gemini-2.0-flash-exp              # Choose Gemini model
ORION_OUTPUT_DIR=~/orion_results               # Results directory
BIGQUERY_DATASET=bigquery-public-data.thelook_ecommerce
MAX_QUERY_ROWS=10000                           # Row limit
QUERY_TIMEOUT=300                              # Timeout (seconds)
```

---

## 📊 Dataset

Uses Google BigQuery's public e-commerce dataset:
- **Dataset**: `bigquery-public-data.thelook_ecommerce`
- **Tables**: `orders`, `order_items`, `products`, `users`
- **Schema**: Automatically loaded with column descriptions

---

## 🔧 Development

### Run from Source
```bash
git clone https://github.com/gavrielhan/orion-data-analyst.git
cd orion-data-analyst
pip install -e .
orion
```
---

## 📝 Commands

In the Orion CLI:
- `exit` / `quit` / `q` - Exit Orion
- `save session` - Save conversation history
- `load session [path]` - Load previous session
- `clear cache` - Clear query cache

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **AI Orchestration** | LangGraph |
| **LLM Integration** | LangChain |
| **AI Model** | Google Gemini 2.0 Flash |
| **Data Warehouse** | Google BigQuery |
| **Data Processing** | pandas |
| **Visualization** | matplotlib, seaborn |
| **State Management** | TypedDict (Python) |
| **Configuration** | python-dotenv |
| **Packaging** | setuptools, PyPI |

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) by LangChain
- Powered by [Google Gemini](https://ai.google.dev/)
- Data from [BigQuery Public Datasets](https://cloud.google.com/bigquery/public-data)

---

**Made with ❤️ by Gavriel Hannuna**
