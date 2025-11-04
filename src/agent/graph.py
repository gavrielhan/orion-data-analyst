"""LangGraph orchestration for Orion agent."""

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    InputNode,
    ContextNode,
    QueryBuilderNode,
    ValidationNode,
    BigQueryExecutorNode,
    OutputNode,
    query_builder_node
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
    
    def _route_from_executor(self, state: AgentState) -> str:
        """Route from executor - retry on error or output on success."""
        query_error = state.get("query_error")
        retry_count = state.get("retry_count", 0)
        
        if query_error and retry_count < 3:
            return "query_builder"
        
        return "output"
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph with all nodes and edges."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("input", InputNode.execute)
        workflow.add_node("context", ContextNode.execute)
        workflow.add_node("query_builder", query_builder_node.execute)
        workflow.add_node("validation", ValidationNode.execute)
        workflow.add_node("bigquery_executor", BigQueryExecutorNode.execute)
        workflow.add_node("output", OutputNode.execute)
        
        # Define edges
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
                "query_builder": "query_builder"  # For retries
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
        workflow.add_conditional_edges(
            "bigquery_executor",
            self._route_from_executor,
            {
                "query_builder": "query_builder",
                "output": "output"
            }
        )
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
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": None
        }
        
        result = self.app.invoke(initial_state)
        return result

