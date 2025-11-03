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
        workflow.add_edge("input", "query_builder")
        workflow.add_edge("query_builder", "bigquery_executor")
        workflow.add_edge("bigquery_executor", "output")
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
            "messages": [],
            "retry_count": 0
        }
        
        result = self.app.invoke(initial_state)
        return result

