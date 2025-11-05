"""Integration tests for Orion graph workflow."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

try:
    from src.agent.graph import OrionGraph
    from src.agent.state import AgentState
    GRAPH_AVAILABLE = True
except ImportError:
    # Skip if dependencies not available
    GRAPH_AVAILABLE = False


@unittest.skipIf(not GRAPH_AVAILABLE, "Graph dependencies not available")
class TestOrionGraph(unittest.TestCase):
    """Tests for Orion graph workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        if GRAPH_AVAILABLE:
            self.graph = OrionGraph()
    
    def test_graph_initialization(self):
        """Test that graph initializes correctly."""
        self.assertIsNotNone(self.graph.graph)
        self.assertIsNotNone(self.graph.app)
    
    @patch('src.agent.nodes.ContextNode.get_schema_context')
    @patch('src.agent.nodes.InputNode.execute')
    def test_meta_question_fast_path(self, mock_input, mock_context):
        """Test that meta questions skip to output directly."""
        mock_context.return_value = "Schema context"
        mock_input.return_value = {
            "query_intent": "meta_question",
            "final_output": "I can help you with..."
        }
        
        initial_state: AgentState = {
            "user_query": "help",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "",
            "discovery_query": None,
            "discovery_result": None,
            "discovery_count": 0,
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
            "conversation_history": [],
            "requires_approval": None,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": [],
            "_verbose": False
        }
        
        # Mock all nodes to avoid actual execution
        with patch('src.agent.nodes.query_builder_node.execute') as mock_qb, \
             patch('src.agent.nodes.OutputNode.execute') as mock_output:
            
            mock_output.return_value = {"final_output": "I can help you with..."}
            
            result = self.graph.invoke("help", [], verbose=False)
            
            # Should have final_output from meta question
            self.assertIn("final_output", result)
    
    def test_routing_from_context(self):
        """Test routing logic from context node."""
        # Test meta question route
        state: AgentState = {
            "final_output": "Meta answer",
            "user_query": "help",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "",
            "discovery_query": None,
            "discovery_result": None,
            "discovery_count": 0,
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
            "conversation_history": [],
            "requires_approval": None,
            "approval_reason": None,
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": [],
            "_verbose": False
        }
        
        route = self.graph._route_from_context(state)
        self.assertEqual(route, "output")
        
        # Test normal query route
        state["final_output"] = ""
        route = self.graph._route_from_context(state)
        self.assertEqual(route, "query_builder")
    
    def test_routing_from_validation(self):
        """Test routing logic from validation node."""
        # Test validation passed -> approval
        state: AgentState = {
            "validation_passed": True,
            "query_error": None,
            "final_output": "",
            "user_query": "test",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "",
            "discovery_query": None,
            "discovery_result": None,
            "discovery_count": 0,
            "estimated_cost_gb": None,
            "query_result": None,
            "analysis_result": None,
            "analysis_type": None,
            "has_empty_results": None,
            "key_findings": None,
            "visualization_path": None,
            "visualization_suggestion": None,
            "conversation_history": [],
            "requires_approval": None,
            "approval_reason": None,
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": [],
            "_verbose": False
        }
        
        route = self.graph._route_from_validation(state)
        self.assertEqual(route, "approval")
        
        # Test validation failed -> output
        state["validation_passed"] = False
        route = self.graph._route_from_validation(state)
        self.assertEqual(route, "output")
        
        # Test with query error -> output
        state["validation_passed"] = True
        state["query_error"] = "Some error"
        route = self.graph._route_from_validation(state)
        self.assertEqual(route, "output")
    
    def test_routing_from_result_check(self):
        """Test routing logic from result check node."""
        # Test error with retries available -> query_builder
        state: AgentState = {
            "query_error": "Some error",
            "retry_count": 1,  # Less than 3
            "has_empty_results": False,
            "final_output": "",
            "user_query": "test",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "",
            "discovery_query": None,
            "discovery_result": None,
            "discovery_count": 0,
            "validation_passed": None,
            "estimated_cost_gb": None,
            "query_result": None,
            "analysis_result": None,
            "analysis_type": None,
            "key_findings": None,
            "visualization_path": None,
            "visualization_suggestion": None,
            "conversation_history": [],
            "requires_approval": None,
            "approval_reason": None,
            "execution_time_sec": None,
            "error_history": [],
            "_verbose": False
        }
        
        route = self.graph._route_from_result_check(state)
        self.assertEqual(route, "query_builder")
        
        # Test empty results -> insight_generator
        state["query_error"] = None
        state["has_empty_results"] = True
        route = self.graph._route_from_result_check(state)
        self.assertEqual(route, "insight_generator")
        
        # Test success -> analysis
        state["has_empty_results"] = False
        state["query_result"] = pd.DataFrame({"col1": [1, 2, 3]})
        route = self.graph._route_from_result_check(state)
        self.assertEqual(route, "analysis")
        
        # Test retry limit exceeded -> output
        state["query_error"] = "Some error"
        state["retry_count"] = 3  # Max retries
        route = self.graph._route_from_result_check(state)
        self.assertEqual(route, "output")
    
    def test_routing_from_bigquery_executor(self):
        """Test routing logic from BigQuery executor."""
        # Test discovery result -> query_builder
        state: AgentState = {
            "discovery_result": "Discovered values: ...",
            "discovery_query": None,
            "sql_query": "",
            "final_output": "",
            "user_query": "test",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "discovery_count": 0,
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
            "conversation_history": [],
            "requires_approval": None,
            "approval_reason": None,
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": [],
            "_verbose": False
        }
        
        route = self.graph._route_from_bigquery_executor(state)
        self.assertEqual(route, "query_builder")
        
        # Test regular query -> result_check
        state["discovery_result"] = None
        state["sql_query"] = "SELECT * FROM orders"
        state["query_result"] = pd.DataFrame({"col1": [1, 2, 3]})
        route = self.graph._route_from_bigquery_executor(state)
        self.assertEqual(route, "result_check")
    
    def test_routing_from_query_builder(self):
        """Test routing logic from query builder."""
        # Test meta answer -> output
        state: AgentState = {
            "final_output": "Meta answer",
            "sql_query": "",
            "discovery_query": None,
            "query_error": None,
            "retry_count": 0,
            "discovery_count": 0,
            "user_query": "test",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "discovery_result": None,
            "validation_passed": None,
            "estimated_cost_gb": None,
            "query_result": None,
            "analysis_result": None,
            "analysis_type": None,
            "has_empty_results": None,
            "key_findings": None,
            "visualization_path": None,
            "visualization_suggestion": None,
            "conversation_history": [],
            "requires_approval": None,
            "approval_reason": None,
            "execution_time_sec": None,
            "error_history": [],
            "_verbose": False
        }
        
        route = self.graph._route_from_query_builder(state)
        self.assertEqual(route, "output")
        
        # Test discovery query -> bigquery_executor
        state["final_output"] = ""
        state["discovery_query"] = "SELECT DISTINCT gender FROM users"
        route = self.graph._route_from_query_builder(state)
        self.assertEqual(route, "bigquery_executor")
        
        # Test SQL query -> validation
        state["discovery_query"] = None
        state["sql_query"] = "SELECT * FROM orders"
        route = self.graph._route_from_query_builder(state)
        self.assertEqual(route, "validation")
    
    def test_initial_state_creation(self):
        """Test that initial state is created correctly."""
        result = self.graph.invoke("test query", [], verbose=False)
        
        # Should have all required fields
        self.assertIn("user_query", result)
        self.assertEqual(result["user_query"], "test query")
        self.assertIn("conversation_history", result)


if __name__ == '__main__':
    unittest.main()

