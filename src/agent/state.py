"""Agent state management for Orion."""

from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    """State shared across all nodes in the LangGraph."""
    
    # User input
    user_query: str
    query_intent: str
    
    # Schema context
    schema_context: Optional[str]
    schema_cache_timestamp: Optional[float]
    
    # SQL generation
    sql_query: str
    
    # Validation
    validation_passed: Optional[bool]
    estimated_cost_gb: Optional[float]
    
    # BigQuery execution
    query_result: Optional[Any]
    query_error: Optional[str]
    
    # Analysis
    analysis_result: Optional[str]
    
    # Output
    final_output: str
    
    # Metadata
    retry_count: int
    execution_time_sec: Optional[float]

