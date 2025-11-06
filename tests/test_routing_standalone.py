#!/usr/bin/env python3
"""
Standalone routing test script for Orion graph workflow.
This script tests routing logic without requiring full graph execution.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.agent.graph import OrionGraph
    from src.agent.state import AgentState
    import pandas as pd
    GRAPH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    GRAPH_AVAILABLE = False
    sys.exit(1)


def create_base_state() -> AgentState:
    """Create a base state for testing."""
    return {
        "user_query": "test query",
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


def test_validation_routing():
    """Test routing from validation node."""
    print("\n=== Testing Validation Routing ===")
    graph = OrionGraph()
    
    # Test 1: Validation passed -> approval
    state = create_base_state()
    state["validation_passed"] = True
    state["query_error"] = None
    route = graph._route_from_validation(state)
    print(f"✓ Validation passed -> {route} (expected: approval)")
    assert route == "approval", f"Expected 'approval', got '{route}'"
    
    # Test 2: Validation failed without error -> output
    state["validation_passed"] = False
    state["query_error"] = None
    route = graph._route_from_validation(state)
    print(f"✓ Validation failed (no error) -> {route} (expected: output)")
    assert route == "output", f"Expected 'output', got '{route}'"
    
    # Test 3: Validation failed with error, retry_count < 3 -> query_builder
    state["validation_passed"] = False
    state["query_error"] = "Syntax error"
    state["retry_count"] = 1
    route = graph._route_from_validation(state)
    print(f"✓ Validation failed (error, retry_count=1) -> {route} (expected: query_builder)")
    assert route == "query_builder", f"Expected 'query_builder', got '{route}'"
    
    # Test 4: Validation failed with error, retry_count >= 3 -> output
    state["retry_count"] = 3
    route = graph._route_from_validation(state)
    print(f"✓ Validation failed (error, retry_count=3) -> {route} (expected: output)")
    assert route == "output", f"Expected 'output', got '{route}'"


def test_result_check_routing():
    """Test routing from result check node."""
    print("\n=== Testing Result Check Routing ===")
    graph = OrionGraph()
    
    # Test 1: Error with retry_count < 3 -> query_builder
    state = create_base_state()
    state["query_error"] = "Execution error"
    state["retry_count"] = 1
    state["has_empty_results"] = False
    route = graph._route_from_result_check(state)
    print(f"✓ Error (retry_count=1) -> {route} (expected: query_builder)")
    assert route == "query_builder", f"Expected 'query_builder', got '{route}'"
    
    # Test 2: Error with retry_count >= 3 -> output
    state["retry_count"] = 3
    route = graph._route_from_result_check(state)
    print(f"✓ Error (retry_count=3) -> {route} (expected: output)")
    assert route == "output", f"Expected 'output', got '{route}'"
    
    # Test 3: Empty results with retry_count < 3 -> query_builder
    state["query_error"] = None
    state["has_empty_results"] = True
    state["retry_count"] = 1
    route = graph._route_from_result_check(state)
    print(f"✓ Empty results (retry_count=1) -> {route} (expected: query_builder)")
    assert route == "query_builder", f"Expected 'query_builder', got '{route}'"
    
    # Test 4: Empty results with retry_count >= 3 -> insight_generator
    state["retry_count"] = 3
    route = graph._route_from_result_check(state)
    print(f"✓ Empty results (retry_count=3) -> {route} (expected: insight_generator)")
    assert route == "insight_generator", f"Expected 'insight_generator', got '{route}'"
    
    # Test 5: Success with data -> analysis
    state["has_empty_results"] = False
    state["query_result"] = pd.DataFrame({"col1": [1, 2, 3]})
    route = graph._route_from_result_check(state)
    print(f"✓ Success with data -> {route} (expected: analysis)")
    assert route == "analysis", f"Expected 'analysis', got '{route}'"


def test_bigquery_executor_routing():
    """Test routing from BigQuery executor node."""
    print("\n=== Testing BigQuery Executor Routing ===")
    graph = OrionGraph()
    
    # Test 1: Discovery result -> query_builder
    state = create_base_state()
    state["discovery_result"] = "Discovered: F, M"
    state["discovery_query"] = None
    state["sql_query"] = ""
    route = graph._route_from_bigquery_executor(state)
    print(f"✓ Discovery result -> {route} (expected: query_builder)")
    assert route == "query_builder", f"Expected 'query_builder', got '{route}'"
    
    # Test 2: Discovery query still exists (error case) -> output
    state["discovery_query"] = "SELECT DISTINCT gender FROM users"
    state["discovery_result"] = None
    route = graph._route_from_bigquery_executor(state)
    print(f"✓ Discovery query still exists -> {route} (expected: output)")
    assert route == "output", f"Expected 'output', got '{route}'"
    
    # Test 3: Regular query -> result_check
    state["discovery_result"] = None
    state["discovery_query"] = None
    state["sql_query"] = "SELECT * FROM orders"
    state["query_result"] = pd.DataFrame({"col1": [1, 2, 3]})
    route = graph._route_from_bigquery_executor(state)
    print(f"✓ Regular query -> {route} (expected: result_check)")
    assert route == "result_check", f"Expected 'result_check', got '{route}'"


def test_query_builder_routing():
    """Test routing from query builder node."""
    print("\n=== Testing Query Builder Routing ===")
    graph = OrionGraph()
    
    # Test 1: Meta answer -> output
    state = create_base_state()
    state["final_output"] = "I can help you with..."
    state["sql_query"] = ""
    state["discovery_query"] = None
    route = graph._route_from_query_builder(state)
    print(f"✓ Meta answer -> {route} (expected: output)")
    assert route == "output", f"Expected 'output', got '{route}'"
    
    # Test 2: Discovery query -> bigquery_executor
    state["final_output"] = ""
    state["discovery_query"] = "SELECT DISTINCT gender FROM users"
    route = graph._route_from_query_builder(state)
    print(f"✓ Discovery query -> {route} (expected: bigquery_executor)")
    assert route == "bigquery_executor", f"Expected 'bigquery_executor', got '{route}'"
    
    # Test 3: SQL query -> validation
    state["discovery_query"] = None
    state["sql_query"] = "SELECT * FROM orders"
    route = graph._route_from_query_builder(state)
    print(f"✓ SQL query -> {route} (expected: validation)")
    assert route == "validation", f"Expected 'validation', got '{route}'"


def test_context_routing():
    """Test routing from context node."""
    print("\n=== Testing Context Routing ===")
    graph = OrionGraph()
    
    # Test 1: Meta question -> output
    state = create_base_state()
    state["final_output"] = "Meta answer"
    route = graph._route_from_context(state)
    print(f"✓ Meta question -> {route} (expected: output)")
    assert route == "output", f"Expected 'output', got '{route}'"
    
    # Test 2: Normal query -> query_builder
    state["final_output"] = ""
    route = graph._route_from_context(state)
    print(f"✓ Normal query -> {route} (expected: query_builder)")
    assert route == "query_builder", f"Expected 'query_builder', got '{route}'"


def main():
    """Run all routing tests."""
    print("=" * 60)
    print("Orion Graph Routing Tests")
    print("=" * 60)
    
    try:
        test_context_routing()
        test_query_builder_routing()
        test_validation_routing()
        test_bigquery_executor_routing()
        test_result_check_routing()
        
        print("\n" + "=" * 60)
        print("✓ All routing tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

