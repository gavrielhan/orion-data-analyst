"""LangGraph orchestration for Orion agent."""

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    InputNode,
    ContextNode,
    QueryBuilderNode,
    ValidationNode,
    BigQueryExecutorNode,
    ResultCheckNode,
    InsightGeneratorNode,
    OutputNode,
    query_builder_node,
    result_check_node,
    insight_generator_node
)


class OrionGraph:
    """Orion agent graph orchestration."""
    
    def __init__(self):
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _route_from_context(self, state: AgentState) -> str:
        """Route from context to query builder."""
        return "query_builder"
    
    def _route_from_query_builder(self, state: AgentState) -> str:
        """Route from query builder - check for meta answers or SQL."""
        final_output = state.get("final_output", "")
        sql_query = state.get("sql_query", "")
        query_error = state.get("query_error")
        retry_count = state.get("retry_count", 0)
        
        # If it's a meta answer, go directly to output
        if final_output and final_output.strip():
            return "output"
        
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
        """Route from validation - execute if passed, output if failed."""
        validation_passed = state.get("validation_passed", False)
        query_error = state.get("query_error")
        
        if query_error or not validation_passed:
            return "output"
        
        return "bigquery_executor"
    
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
        
        # Case 3: Success with data or max retries reached
        return "output"
    
    def _build_graph(self) -> StateGraph:
        """
        Build self-healing LangGraph with error recovery and resilience.
        
        Flow: input → context → query_builder → validation → executor → result_check
        Result_check routes to:
          - query_builder (retry with error context, max 3 times)
          - insight_generator (empty results explanation)
          - output (success with data)
        """
        workflow = StateGraph(AgentState)
        
        # Add all nodes
        workflow.add_node("input", InputNode.execute)
        workflow.add_node("context", ContextNode.execute)
        workflow.add_node("query_builder", query_builder_node.execute)
        workflow.add_node("validation", ValidationNode.execute)
        workflow.add_node("bigquery_executor", BigQueryExecutorNode.execute)
        workflow.add_node("result_check", result_check_node.execute)
        workflow.add_node("insight_generator", insight_generator_node.execute)
        workflow.add_node("output", OutputNode.execute)
        
        # Define edges with conditional routing
        workflow.set_entry_point("input")
        workflow.add_edge("input", "context")
        workflow.add_conditional_edges(
            "context",
            self._route_from_context,
            {"query_builder": "query_builder"}
        )
        workflow.add_conditional_edges(
            "query_builder",
            self._route_from_query_builder,
            {
                "output": "output",
                "validation": "validation",
                "query_builder": "query_builder"  # Self-healing retry loop
            }
        )
        workflow.add_conditional_edges(
            "validation",
            self._route_from_validation,
            {
                "bigquery_executor": "bigquery_executor",
                "output": "output"
            }
        )
        # Executor always goes to result_check for evaluation
        workflow.add_edge("bigquery_executor", "result_check")
        
        # Result check implements smart routing based on outcome
        workflow.add_conditional_edges(
            "result_check",
            self._route_from_result_check,
            {
                "query_builder": "query_builder",      # Retry with error context
                "insight_generator": "insight_generator",  # Explain empty results
                "output": "output"                      # Success
            }
        )
        
        # Both insight generator and output are terminal nodes
        workflow.add_edge("insight_generator", "output")
        workflow.add_edge("output", END)
        
        return workflow
    
    def invoke(self, user_query: str) -> dict:
        """Execute the agent with a user query."""
        initial_state: AgentState = {
            "user_query": user_query,
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "",
            "validation_passed": None,
            "estimated_cost_gb": None,
            "query_result": None,
            "query_error": None,
            "analysis_result": None,
            "has_empty_results": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": []
        }
        
        result = self.app.invoke(initial_state)
        return result

