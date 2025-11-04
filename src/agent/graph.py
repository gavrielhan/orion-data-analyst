"""LangGraph orchestration for Orion agent."""

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    InputNode,
    ContextNode,
    QueryBuilderNode,
    ValidationNode,
    ApprovalNode,
    BigQueryExecutorNode,
    ResultCheckNode,
    AnalysisNode,
    InsightGeneratorNode,
    OutputNode,
    query_builder_node,
    approval_node,
    result_check_node,
    analysis_node,
    insight_generator_node
)


class OrionGraph:
    """Orion agent graph orchestration."""
    
    def __init__(self):
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _route_from_context(self, state: AgentState) -> str:
        """Route from context - skip to output if meta-question was detected early."""
        # Check if InputNode already answered (fast-path meta-questions)
        if state.get("final_output"):
            return "output"
        return "query_builder"
    
    def _route_from_query_builder(self, state: AgentState) -> str:
        """Route from query builder - check for meta answers, discovery, or SQL."""
        final_output = state.get("final_output", "")
        sql_query = state.get("sql_query", "")
        discovery_query = state.get("discovery_query", "")
        query_error = state.get("query_error")
        retry_count = state.get("retry_count", 0)
        
        # If it's a meta answer, go directly to output
        if final_output and final_output.strip():
            return "output"
        
        # If it's a discovery query, execute it first
        if discovery_query and discovery_query.strip():
            return "bigquery_executor"  # Execute discovery, then loop back
        
        # If there's an error, check if we should retry
        if query_error:
            if not sql_query:
                should_retry = (
                    ("Invalid response format" in query_error and retry_count < 3) or
                    ("Rate limit" in query_error and retry_count < 3)
                )
                if should_retry:
                    return "query_builder"
                return "output"
        
        # SQL generated successfully, proceed to validation
        return "validation"
    
    def _route_from_validation(self, state: AgentState) -> str:
        """Route from validation - check approval if passed, output if failed."""
        validation_passed = state.get("validation_passed", False)
        query_error = state.get("query_error")
        
        if query_error or not validation_passed:
            return "output"
        
        return "approval"
    
    def _route_from_bigquery_executor(self, state: AgentState) -> str:
        """Route from bigquery executor - check if discovery or main query."""
        discovery_result = state.get("discovery_result")
        
        # If we just completed a discovery query, go back to query builder
        if discovery_result:
            return "query_builder"
        
        # Regular query completed, check results
        return "result_check"
    
    def _route_from_result_check(self, state: AgentState) -> str:
        """
        Route from result check based on execution outcome.
        Implements self-healing retry logic and empty result handling.
        """
        query_error = state.get("query_error")
        retry_count = state.get("retry_count", 0)
        has_empty_results = state.get("has_empty_results", False)
        
        # Case 1: Error occurred and retry limit not reached - self-heal by retrying
        if query_error and retry_count < 3:
            return "query_builder"
        
        # Case 2: Empty results - generate insight explanation
        if has_empty_results:
            return "insight_generator"
        
        # Case 3: Success with data - analyze it
        return "analysis"
    
    def _build_graph(self) -> StateGraph:
        """
        Build conversational LangGraph with human-in-the-loop approval.
        
        Flow: input → context → query_builder → validation → approval → executor → result_check
        Approval node flags expensive queries for user confirmation (handled in CLI).
        """
        workflow = StateGraph(AgentState)
        
        # Add all nodes
        workflow.add_node("input", InputNode.execute)
        workflow.add_node("context", ContextNode.execute)
        workflow.add_node("query_builder", query_builder_node.execute)
        workflow.add_node("validation", ValidationNode.execute)
        workflow.add_node("approval", approval_node.execute)
        workflow.add_node("bigquery_executor", BigQueryExecutorNode.execute)
        workflow.add_node("result_check", result_check_node.execute)
        workflow.add_node("analysis", analysis_node.execute)
        workflow.add_node("insight_generator", insight_generator_node.execute)
        workflow.add_node("output", OutputNode.execute)
        
        # Define edges with conditional routing
        workflow.set_entry_point("input")
        workflow.add_edge("input", "context")
        workflow.add_conditional_edges(
            "context",
            self._route_from_context,
            {
                "query_builder": "query_builder",
                "output": "output"  # Fast-path for instant meta-questions
            }
        )
        workflow.add_conditional_edges(
            "query_builder",
            self._route_from_query_builder,
            {
                "output": "output",
                "validation": "validation",
                "query_builder": "query_builder",  # Self-healing retry loop
                "bigquery_executor": "bigquery_executor"  # Discovery query execution
            }
        )
        workflow.add_conditional_edges(
            "validation",
            self._route_from_validation,
            {
                "approval": "approval",
                "output": "output"
            }
        )
        # Approval always proceeds to executor (CLI handles user confirmation)
        workflow.add_edge("approval", "bigquery_executor")
        
        # Executor routes based on whether it's discovery or main query
        workflow.add_conditional_edges(
            "bigquery_executor",
            self._route_from_bigquery_executor,
            {
                "query_builder": "query_builder",     # After discovery, loop back
                "result_check": "result_check"        # After main query, check results
            }
        )
        
        # Result check implements smart routing based on outcome
        workflow.add_conditional_edges(
            "result_check",
            self._route_from_result_check,
            {
                "query_builder": "query_builder",           # Retry with error context
                "insight_generator": "insight_generator",   # Explain empty results
                "analysis": "analysis"                      # Success - analyze data
            }
        )
        
        # Analysis and insight generation pipeline
        workflow.add_edge("analysis", "insight_generator")
        workflow.add_edge("insight_generator", "output")
        workflow.add_edge("output", END)
        
        return workflow
    
    def invoke(self, user_query: str, conversation_history: list = None, verbose: bool = True) -> dict:
        """Execute the agent with a user query and optional conversation history."""
        initial_state: AgentState = {
            "user_query": user_query,
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "",
            "discovery_query": None,
            "discovery_result": None,
            "validation_passed": None,
            "estimated_cost_gb": None,
            "query_result": None,
            "query_error": None,
            "analysis_result": None,
            "analysis_type": None,
            "has_empty_results": None,
            "key_findings": None,
            "visualization_path": None,
            "visualization_suggestion": None,
            "conversation_history": conversation_history or [],
            "requires_approval": None,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": [],
            "_verbose": verbose  # For progress updates
        }
        
        result = self.app.invoke(initial_state)
        return result

