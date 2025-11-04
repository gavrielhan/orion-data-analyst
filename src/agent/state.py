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
    analysis_type: Optional[str]  # trends, ranking, segmentation, aggregation
    has_empty_results: Optional[bool]
    key_findings: Optional[list]  # Structured insights
    visualization_path: Optional[str]  # Path to saved chart
    visualization_suggestion: Optional[dict]  # {chart_type, x_col, y_col, title}
    
    # Output
    final_output: str

    # Discovery queries for data exploration
    discovery_query: Optional[str]        # Query to discover data values
    discovery_result: Optional[str]       # Results from discovery query
    discovery_count: int                  # Track discovery queries to prevent infinite loops

    # Conversation memory (limited to last 5 interactions)
    conversation_history: Optional[list]  # [{query, result, timestamp}]
    
    # Human-in-the-loop
    requires_approval: Optional[bool]
    approval_reason: Optional[str]
    
    # Metadata
    retry_count: int
    execution_time_sec: Optional[float]
    error_history: Optional[list]

