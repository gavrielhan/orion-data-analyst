"""Unit tests for Orion agent nodes."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from pathlib import Path

from src.agent.nodes import (
    InputNode,
    ContextNode,
    ApprovalNode,
    ValidationNode,
    OutputNode,
)
from src.agent.state import AgentState


class TestInputNode(unittest.TestCase):
    """Tests for InputNode."""
    
    def test_meta_question_help(self):
        """Test that help questions are detected as meta."""
        state: AgentState = {
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
            "conversation_history": None,
            "requires_approval": None,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": None,
            "_verbose": False
        }
        
        result = InputNode.execute(state)
        self.assertEqual(result["query_intent"], "meta_question")
        self.assertIn("final_output", result)
        self.assertIn("I can analyze", result["final_output"])
    
    def test_sales_query_intent(self):
        """Test that sales queries are classified as aggregation."""
        state: AgentState = {
            "user_query": "show me total sales",
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
            "conversation_history": None,
            "requires_approval": None,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": None,
            "_verbose": False
        }
        
        result = InputNode.execute(state)
        self.assertEqual(result["query_intent"], "aggregation")
    
    def test_top_query_intent(self):
        """Test that 'top' queries are classified as ranking."""
        state: AgentState = {
            "user_query": "top 10 products",
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
            "conversation_history": None,
            "requires_approval": None,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": None,
            "_verbose": False
        }
        
        result = InputNode.execute(state)
        self.assertEqual(result["query_intent"], "ranking")


class TestContextNode(unittest.TestCase):
    """Tests for ContextNode."""
    
    @patch('src.agent.nodes.ContextNode.SCHEMA_CACHE_FILE')
    def test_schema_context_loaded(self, mock_file):
        """Test that schema context is loaded from file."""
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = "Test schema context"
        
        state: AgentState = {
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
            "error_history": None,
            "_verbose": False
        }
        
        result = ContextNode.execute(state)
        self.assertEqual(result["schema_context"], "Test schema context")
        self.assertIn("schema_cache_timestamp", result)
    
    def test_schema_fallback(self):
        """Test fallback when schema file doesn't exist."""
        state: AgentState = {
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
            "error_history": None,
            "_verbose": False
        }
        
        # Use actual file path, but it might not exist
        schema_file = Path(__file__).parent.parent / "schema_context.txt"
        if not schema_file.exists():
            # Test fallback behavior
            with patch.object(ContextNode, 'SCHEMA_CACHE_FILE', schema_file):
                result = ContextNode.execute(state)
                # Should still return something
                self.assertIn("schema_context", result)


class TestApprovalNode(unittest.TestCase):
    """Tests for ApprovalNode."""
    
    def test_approval_required_high_cost(self):
        """Test that approval is required for queries > 5GB."""
        state: AgentState = {
            "user_query": "test",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "SELECT * FROM orders",
            "discovery_query": None,
            "discovery_result": None,
            "discovery_count": 0,
            "validation_passed": True,
            "estimated_cost_gb": 7.5,  # Above threshold
            "query_result": None,
            "query_error": None,
            "analysis_result": None,
            "analysis_type": None,
            "has_empty_results": None,
            "key_findings": None,
            "visualization_path": None,
            "visualization_suggestion": None,
            "conversation_history": None,
            "requires_approval": None,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": None,
            "_verbose": False
        }
        
        result = ApprovalNode.execute(state)
        self.assertTrue(result["requires_approval"])
        self.assertIsNotNone(result["approval_reason"])
        self.assertIn("7.5", result["approval_reason"])
    
    def test_no_approval_low_cost(self):
        """Test that approval is not required for queries < 5GB."""
        state: AgentState = {
            "user_query": "test",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "SELECT * FROM orders LIMIT 100",
            "discovery_query": None,
            "discovery_result": None,
            "discovery_count": 0,
            "validation_passed": True,
            "estimated_cost_gb": 2.5,  # Below threshold
            "query_result": None,
            "query_error": None,
            "analysis_result": None,
            "analysis_type": None,
            "has_empty_results": None,
            "key_findings": None,
            "visualization_path": None,
            "visualization_suggestion": None,
            "conversation_history": None,
            "requires_approval": None,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": None,
            "_verbose": False
        }
        
        result = ApprovalNode.execute(state)
        self.assertFalse(result["requires_approval"])
        self.assertIsNone(result["approval_reason"])
    
    def test_skips_approval_when_validation_failed(self):
        """Test that approval check is skipped when validation fails."""
        state: AgentState = {
            "user_query": "test",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "SELECT * FROM orders",
            "discovery_query": None,
            "discovery_result": None,
            "discovery_count": 0,
            "validation_passed": False,  # Validation failed
            "estimated_cost_gb": 10.0,  # High cost but validation failed
            "query_result": None,
            "query_error": None,
            "analysis_result": None,
            "analysis_type": None,
            "has_empty_results": None,
            "key_findings": None,
            "visualization_path": None,
            "visualization_suggestion": None,
            "conversation_history": None,
            "requires_approval": None,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": None,
            "_verbose": False
        }
        
        result = ApprovalNode.execute(state)
        self.assertEqual(result, {})  # Should return empty dict


class TestValidationNode(unittest.TestCase):
    """Tests for ValidationNode."""
    
    @patch('src.agent.nodes.bigquery.Client')
    def test_blocks_dangerous_keywords(self, mock_client_class):
        """Test that dangerous SQL keywords are blocked."""
        state: AgentState = {
            "user_query": "test",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "DROP TABLE orders",
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
            "conversation_history": None,
            "requires_approval": None,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": None,
            "_verbose": False
        }
        
        result = ValidationNode.execute(state)
        self.assertFalse(result["validation_passed"])
        self.assertIn("Security violation", result["query_error"])
        self.assertIn("DROP", result["query_error"])
    
    @patch('src.agent.nodes.bigquery.Client')
    def test_blocks_too_expensive_queries(self, mock_client_class):
        """Test that queries exceeding MAX_COST_GB are blocked."""
        mock_client = mock_client_class.return_value
        mock_query_job = Mock()
        mock_query_job.total_bytes_processed = 15 * (1024 ** 3)  # 15 GB
        mock_client.query.return_value = mock_query_job
        
        state: AgentState = {
            "user_query": "test",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "SELECT * FROM `bigquery-public-data.thelook_ecommerce.orders`",
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
            "conversation_history": None,
            "requires_approval": None,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": None,
            "_verbose": False
        }
        
        result = ValidationNode.execute(state)
        self.assertFalse(result["validation_passed"])
        self.assertIn("too expensive", result["query_error"].lower())
        self.assertGreater(result["estimated_cost_gb"], ValidationNode.MAX_COST_GB)
    
    @patch('src.agent.nodes.bigquery.Client')
    def test_validates_successful_query(self, mock_client_class):
        """Test that valid queries pass validation."""
        mock_client = mock_client_class.return_value
        mock_query_job = Mock()
        mock_query_job.total_bytes_processed = 2 * (1024 ** 3)  # 2 GB
        mock_client.query.return_value = mock_query_job
        
        state: AgentState = {
            "user_query": "test",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "SELECT * FROM `bigquery-public-data.thelook_ecommerce.orders` LIMIT 100",
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
            "conversation_history": None,
            "requires_approval": None,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": None,
            "_verbose": False
        }
        
        result = ValidationNode.execute(state)
        self.assertTrue(result["validation_passed"])
        self.assertAlmostEqual(result["estimated_cost_gb"], 2.0, places=1)


class TestOutputNode(unittest.TestCase):
    """Tests for OutputNode."""
    
    def test_output_with_data_less_than_50_rows(self):
        """Test output when there are 30 rows (should show all)."""
        df = pd.DataFrame({
            "product_id": range(30),
            "sales": range(30, 60)
        })
        
        state: AgentState = {
            "user_query": "test",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "SELECT * FROM products",
            "discovery_query": None,
            "discovery_result": None,
            "discovery_count": 0,
            "validation_passed": True,
            "estimated_cost_gb": 1.0,
            "query_result": df,
            "query_error": None,
            "analysis_result": None,
            "analysis_type": None,
            "has_empty_results": False,
            "key_findings": None,
            "visualization_path": None,
            "visualization_suggestion": None,
            "conversation_history": None,
            "requires_approval": False,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": 1.5,
            "error_history": None,
            "_verbose": False
        }
        
        result = OutputNode.execute(state)
        output = result["final_output"]
        self.assertIn("30 rows", output)
        self.assertNotIn("showing first 50", output)
        self.assertNotIn("more rows", output)
    
    def test_output_with_data_more_than_50_rows(self):
        """Test output when there are 100 rows (should show only first 50)."""
        df = pd.DataFrame({
            "product_id": range(100),
            "sales": range(100, 200)
        })
        
        state: AgentState = {
            "user_query": "test",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "SELECT * FROM products",
            "discovery_query": None,
            "discovery_result": None,
            "discovery_count": 0,
            "validation_passed": True,
            "estimated_cost_gb": 1.0,
            "query_result": df,
            "query_error": None,
            "analysis_result": None,
            "analysis_type": None,
            "has_empty_results": False,
            "key_findings": None,
            "visualization_path": None,
            "visualization_suggestion": None,
            "conversation_history": None,
            "requires_approval": False,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": 1.5,
            "error_history": None,
            "_verbose": False
        }
        
        result = OutputNode.execute(state)
        output = result["final_output"]
        self.assertIn("100 rows, showing first 50", output)
        self.assertIn("50 more rows", output)
    
    def test_output_with_empty_results(self):
        """Test output when query returns no results."""
        df = pd.DataFrame()
        
        state: AgentState = {
            "user_query": "test",
            "query_intent": "",
            "schema_context": None,
            "schema_cache_timestamp": None,
            "sql_query": "SELECT * FROM products WHERE 1=0",
            "discovery_query": None,
            "discovery_result": None,
            "discovery_count": 0,
            "validation_passed": True,
            "estimated_cost_gb": 1.0,
            "query_result": df,
            "query_error": None,
            "analysis_result": None,
            "analysis_type": None,
            "has_empty_results": True,
            "key_findings": None,
            "visualization_path": None,
            "visualization_suggestion": None,
            "conversation_history": None,
            "requires_approval": False,
            "approval_reason": None,
            "final_output": "",
            "retry_count": 0,
            "execution_time_sec": 1.5,
            "error_history": None,
            "_verbose": False
        }
        
        result = OutputNode.execute(state)
        output = result["final_output"]
        self.assertIn("No results found", output)
    
    def test_output_with_existing_final_output(self):
        """Test that existing final_output is returned as-is."""
        state: AgentState = {
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
            "query_error": None,
            "analysis_result": None,
            "analysis_type": None,
            "has_empty_results": None,
            "key_findings": None,
            "visualization_path": None,
            "visualization_suggestion": None,
            "conversation_history": None,
            "requires_approval": None,
            "approval_reason": None,
            "final_output": "Pre-existing output",
            "retry_count": 0,
            "execution_time_sec": None,
            "error_history": None,
            "_verbose": False
        }
        
        result = OutputNode.execute(state)
        self.assertEqual(result["final_output"], "Pre-existing output")


if __name__ == '__main__':
    unittest.main()

