"""Agent state management for Orion."""

from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    """State shared across all nodes in the LangGraph."""
    
    # User input
    user_query: str
    query_intent: str
    
    # SQL generation
    sql_query: str
    
    # BigQuery execution
    query_result: Optional[Any]
    query_error: Optional[str]
    
    # Analysis
    analysis_result: Optional[str]
    
    # Output
    final_output: str
    
    # Metadata
    retry_count: int

