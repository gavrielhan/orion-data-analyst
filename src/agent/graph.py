"""LangGraph orchestration for Orion agent."""

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    InputNode,
    QueryBuilderNode,
    BigQueryExecutorNode,
    OutputNode,
    query_builder_node
)


class OrionGraph:
    """Orion agent graph orchestration."""
    
    def __init__(self):
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _check_input_routing(self, state: AgentState) -> str:
        """Route from input to query builder."""
        # All queries go to query_builder first - it will decide if meta or SQL
        return "query_builder"
    
    def _check_if_direct_output(self, state: AgentState) -> str:
        """Check if QueryBuilder returned a direct output (error message)."""
        # Check if final_output exists and is not empty
        final_output = state.get("final_output", "")
        sql_query = state.get("sql_query", "")
        query_error = state.get("query_error")
        retry_count = state.get("retry_count", 0)
        
        if final_output and final_output.strip():
            return "output"
        
        if query_error:
            # If there's an error and no SQL query, check if we should retry
            if not sql_query:
                # Check if we should retry (for missing prefix errors or rate limits)
                should_retry = (
                    ("Invalid response format" in query_error and retry_count < 3) or
                    ("Rate limit" in query_error and retry_count < 3)
                )
                if should_retry:
                    return "query_builder"
                # Otherwise, go to output with error
                return "output"
        
        # No error, proceed to BigQuery execution
        return "bigquery_executor"
    
    def _check_retry_needed(self, state: AgentState) -> str:
        """Check if we should retry after BigQuery error."""
        query_error = state.get("query_error")
        retry_count = state.get("retry_count", 0)
        
        # If there's an error and we haven't exceeded max retries, retry
        if query_error and retry_count < 3:
            return "query_builder"
        
        # Otherwise, go to output
        return "output"
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph with all nodes and edges."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("input", InputNode.execute)
        workflow.add_node("query_builder", query_builder_node.execute)
        workflow.add_node("bigquery_executor", BigQueryExecutorNode.execute)
        workflow.add_node("output", OutputNode.execute)
        
        # Define edges
        workflow.set_entry_point("input")
        workflow.add_conditional_edges(
            "input",
            self._check_input_routing,
            {
                "output": "output",
                "query_builder": "query_builder"
            }
        )
        workflow.add_conditional_edges(
            "query_builder",
            self._check_if_direct_output,
            {
                "output": "output",
                "bigquery_executor": "bigquery_executor"
            }
        )
        workflow.add_conditional_edges(
            "bigquery_executor",
            self._check_retry_needed,
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
            "sql_query": "",
            "query_result": None,
            "query_error": None,
            "analysis_result": None,
            "final_output": "",
            "retry_count": 0
        }
        
        result = self.app.invoke(initial_state)
        return result

