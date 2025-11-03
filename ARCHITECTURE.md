# Orion Architecture

## Overview

Orion is built using LangGraph, which enables modular reasoning through connected nodes with managed state transitions. Each node encapsulates a distinct analytical step.

## Node Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Query                                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        InputNode                                      │
│  • Receives user query                                               │
│  • Classifies intent (aggregation, ranking, trend, etc.)            │
│  • Normalizes input                                                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     QueryBuilderNode                                 │
│  • Uses Gemini LLM to generate SQL                                  │
│  • Provides schema context                                           │
│  • Validates query structure                                         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BigQueryExecutorNode                              │
│  • Executes SQL on BigQuery                                          │
│  • Returns DataFrame results                                         │
│  • Handles errors and timeouts                                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OutputNode                                    │
│  • Formats results for display                                       │
│  • Handles errors                                                   │
│  • Returns final output                                              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │  END          │
                        └───────────────┘
```

## Components

### Core Modules

- **src/agent/state.py**: Defines `AgentState` - TypedDict shared across all nodes
- **src/agent/nodes.py**: Implementation of all four nodes
- **src/agent/graph.py**: LangGraph orchestration and workflow
- **src/config.py**: Configuration management and environment loading
- **src/cli.py**: Command-line interface

### Node Details

#### InputNode
- **Purpose**: Query classification and normalization
- **Input**: Raw user query string
- **Output**: Query intent, normalized message
- **Logic**: Keyword-based intent classification

#### QueryBuilderNode
- **Purpose**: Dynamic SQL generation
- **Input**: User query + schema context
- **Output**: Generated SQL query
- **Tools**: Gemini 1.5 Flash LLM via LangChain
- **Context**: Hardcoded schema for MVP (orders, order_items, products, users)

#### BigQueryExecutorNode
- **Purpose**: Execute SQL on BigQuery
- **Input**: SQL query
- **Output**: pandas DataFrame or error
- **Tools**: google-cloud-bigquery SDK
- **Features**: Row limits, timeout handling

#### OutputNode
- **Purpose**: Format results for display
- **Input**: DataFrame or error
- **Output**: Formatted string
- **Features**: Empty result handling, error formatting

## State Management

`AgentState` contains:

```python
{
    "user_query": str,           # Original user input
    "query_intent": str,         # Classified intent
    "sql_query": str,            # Generated SQL
    "query_result": DataFrame,   # BigQuery results
    "query_error": str,          # Any errors
    "analysis_result": str,      # Analysis (future)
    "final_output": str,         # Final formatted output
    "messages": List,            # Conversation history
    "retry_count": int           # Retry tracking
}
```

## Data Flow

1. User submits query via CLI
2. `InputNode` classifies and normalizes
3. `QueryBuilderNode` generates SQL using Gemini
4. `BigQueryExecutorNode` executes on BigQuery
5. `OutputNode` formats and displays results
6. Result returned to CLI

## Future Enhancements

For Milestone 2+:
- **ValidationNode**: Pre-execution SQL validation and security checks
- **ContextNode**: Dynamic schema retrieval and conversation history
- **ResultCheckNode**: Result validation and routing logic
- **AnalysisNode**: Statistical analysis on results
- **InsightGeneratorNode**: Natural language insights generation

## Technology Stack

- **LangGraph**: Workflow orchestration
- **LangChain**: LLM integration
- **Google Generative AI**: Gemini 1.5 Flash
- **BigQuery**: Data warehouse
- **pandas**: Data manipulation
- **python-dotenv**: Configuration management

## Design Principles

1. **Modularity**: Each node has single responsibility
2. **Composability**: Nodes can be added/removed easily
3. **State Management**: Centralized state across nodes
4. **Error Handling**: Graceful degradation at each step
5. **Extensibility**: Easy to add new nodes and capabilities

