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
        from src.utils.formatter import OutputFormatter
        import sys
        
        final_output = state.get("final_output", "")
        sql_query = state.get("sql_query", "")
        discovery_query = state.get("discovery_query", "")
        discovery_result = state.get("discovery_result")
        discovery_count = state.get("discovery_count", 0)
        query_error = state.get("query_error")
        retry_count = state.get("retry_count", 0)
        verbose = state.get("_verbose", False)
        
        def _vprint(msg):
            if verbose:
                print(msg)
                sys.stdout.flush()
        
        _vprint(OutputFormatter.info(f"  → Routing from query_builder: sql_query={bool(sql_query)}, discovery_query={bool(discovery_query)}, discovery_result={bool(discovery_result)}, discovery_count={discovery_count}, retry_count={retry_count}, error={bool(query_error)}"))
        
        print(f"\n🔍 [ROUTING] _route_from_query_builder")
        print(f"   final_output: {bool(final_output)}")
        print(f"   sql_query: {bool(sql_query)}")
        print(f"   discovery_query: {bool(discovery_query)}")
        print(f"   discovery_result: {bool(discovery_result)}")
        print(f"   discovery_count: {discovery_count}")
        print(f"   retry_count: {retry_count}")
        print(f"   query_error: {bool(query_error)}")
        if query_error:
            print(f"   error_msg: {query_error[:100]}...")
        sys.stdout.flush()
        
        # If it's a meta answer, go directly to output
        if final_output and final_output.strip():
            _vprint(OutputFormatter.info("  → Routing to OUTPUT (meta answer)"))
            print(f"🔍 [ROUTING] → OUTPUT (meta answer)")
            sys.stdout.flush()
            return "output"
        
        # If discovery_count is too high, prevent infinite loops
        if discovery_count >= 2:
            _vprint(OutputFormatter.warning("  → Routing to OUTPUT (discovery_count >= 2)"))
            print(f"🔍 [ROUTING] → OUTPUT (discovery_count >= 2)")
            sys.stdout.flush()
            return "output"  # Force output instead of looping
        
        # If discovery_result exists but we've tried multiple times to generate SQL, force output
        if discovery_result and not sql_query:
            # If we've already tried once (retry_count > 0), force output to prevent loops
            if retry_count >= 2:
                _vprint(OutputFormatter.warning(f"  → Routing to OUTPUT (discovery_result exists, retry_count={retry_count} >= 2)"))
                print(f"🔍 [ROUTING] → OUTPUT (discovery_result exists, retry_count >= 2)")
                sys.stdout.flush()
                return "output"  # Prevent infinite loop after discovery
            # If discovery_result exists but no SQL, we should be generating SQL now
            # This is normal - allow it to proceed
            _vprint(OutputFormatter.info("  → Discovery result exists, will generate SQL"))
            print(f"🔍 [ROUTING] → Continue (discovery_result exists, will generate SQL)")
            sys.stdout.flush()
        
        # If it's a discovery query, execute it first
        if discovery_query and discovery_query.strip():
            _vprint(OutputFormatter.info("  → Routing to BIGQUERY_EXECUTOR (discovery query)"))
            print(f"🔍 [ROUTING] → BIGQUERY_EXECUTOR (discovery query)")
            sys.stdout.flush()
            return "bigquery_executor"  # Execute discovery, then loop back
        
        # If discovery_result exists but no SQL yet, this is normal - we'll generate SQL
        # But if discovery_result exists AND we have SQL, we should proceed
        
        # If there's an error, check if we should retry
        if query_error:
            if not sql_query:
                # CRITICAL: "Discovery already completed" means LLM ignored discovery_result
                # Force output immediately - don't retry (would loop infinitely)
                if "Discovery already completed" in query_error:
                    _vprint(OutputFormatter.error("  → Routing to OUTPUT (LLM tried to generate DISCOVER after discovery completed)"))
                    print(f"🔍 [ROUTING] → OUTPUT (Discovery already completed - no retry)")
                    sys.stdout.flush()
                    return "output"
                
                # CRITICAL: Rate limit errors should NEVER retry through routing
                # QueryBuilderNode already handles exponential backoff internally
                # If we hit rate limit, output the error immediately - don't loop
                if "Rate limit" in query_error or "429" in query_error or "Resource exhausted" in query_error:
                    _vprint(OutputFormatter.error("  → Routing to OUTPUT (Rate limit - no retry through routing)"))
                    print(f"🔍 [ROUTING] → OUTPUT (Rate limit - no retry through routing)")
                    sys.stdout.flush()
                    return "output"
                
                # Only retry for format errors (not rate limits)
                should_retry = (
                    "Invalid response format" in query_error and retry_count < 3
                )
                if should_retry:
                    _vprint(OutputFormatter.info(f"  → Routing to QUERY_BUILDER (retry {retry_count + 1}/3)"))
                    print(f"🔍 [ROUTING] → QUERY_BUILDER (retry {retry_count + 1}/3)")
                    sys.stdout.flush()
                    return "query_builder"
                _vprint(OutputFormatter.warning(f"  → Routing to OUTPUT (max retries reached or non-retryable error)"))
                print(f"🔍 [ROUTING] → OUTPUT (max retries reached or non-retryable error)")
                sys.stdout.flush()
                return "output"
        
        # SQL generated successfully, proceed to validation
        _vprint(OutputFormatter.info("  → Routing to VALIDATION (SQL generated)"))
        print(f"🔍 [ROUTING] → VALIDATION (SQL generated)")
        sys.stdout.flush()
        return "validation"
    
    def _route_from_validation(self, state: AgentState) -> str:
        """Route from validation - check approval if passed, output if failed."""
        # Absolute terminator: if final_output is set, always go to output
        if state.get("final_output"):
            return "output"
        
        validation_passed = state.get("validation_passed", False)
        query_error = state.get("query_error")
        
        if query_error or not validation_passed:
            return "output"
        
        return "approval"
    
    def _route_from_bigquery_executor(self, state: AgentState) -> str:
        """Route from bigquery executor - check if discovery or main query."""
        from src.utils.formatter import OutputFormatter
        import sys
        
        verbose = state.get("_verbose", False)
        final_output = state.get("final_output")
        discovery_query = state.get("discovery_query")
        discovery_result = state.get("discovery_result")
        sql_query = state.get("sql_query")
        
        def _vprint(msg):
            if verbose:
                print(msg)
                sys.stdout.flush()
        
        _vprint(OutputFormatter.info(f"  → Routing from bigquery_executor: discovery_query={bool(discovery_query)}, discovery_result={bool(discovery_result)}, sql_query={bool(sql_query)}"))
        
        print(f"\n🔍 [ROUTING] _route_from_bigquery_executor")
        print(f"   final_output: {bool(final_output)}")
        print(f"   discovery_query: {bool(discovery_query)}")
        print(f"   discovery_result: {bool(discovery_result)}")
        print(f"   sql_query: {bool(sql_query)}")
        sys.stdout.flush()
        
        # Absolute terminator: if final_output is set, always go to output
        if final_output:
            _vprint(OutputFormatter.info("  → Routing to OUTPUT (final_output set)"))
            print(f"🔍 [ROUTING] → OUTPUT (final_output set)")
            sys.stdout.flush()
            return "output"
        
        # If we just completed a discovery query (no sql_query yet), go back to query builder
        # BUT: if sql_query exists, we've already moved past discovery, so check results
        if discovery_result and not sql_query and not discovery_query:
            # Discovery completed, no SQL yet - go back to generate SQL
            _vprint(OutputFormatter.info("  → Routing to QUERY_BUILDER (discovery completed, need SQL)"))
            print(f"🔍 [ROUTING] → QUERY_BUILDER (discovery completed, need SQL)")
            sys.stdout.flush()
            return "query_builder"
        
        # If discovery_query still exists, something went wrong - prevent infinite loop
        if discovery_query:
            # Discovery query wasn't cleared - this shouldn't happen, but prevent loop
            _vprint(OutputFormatter.warning("  → Routing to OUTPUT (discovery_query not cleared - safety)"))
            print(f"🔍 [ROUTING] → OUTPUT (discovery_query not cleared - safety)")
            sys.stdout.flush()
            return "output"
        
        # Regular query completed, check results
        _vprint(OutputFormatter.info("  → Routing to RESULT_CHECK (main query completed)"))
        print(f"🔍 [ROUTING] → RESULT_CHECK (main query completed)")
        sys.stdout.flush()
        return "result_check"
    
    def _route_from_result_check(self, state: AgentState) -> str:
        """
        Route from result check based on execution outcome.
        Implements self-healing retry logic and empty result handling.
        """
        from src.utils.formatter import OutputFormatter
        import sys
        
        verbose = state.get("_verbose", False)
        final_output = state.get("final_output")
        query_error = state.get("query_error")
        retry_count = state.get("retry_count", 0)
        has_empty_results = state.get("has_empty_results", False)
        
        def _vprint(msg):
            if verbose:
                print(msg)
                sys.stdout.flush()
        
        _vprint(OutputFormatter.info(f"  → Routing from result_check: error={bool(query_error)}, retry_count={retry_count}, has_empty={has_empty_results}"))
        
        print(f"\n🔍 [ROUTING] _route_from_result_check")
        print(f"   final_output: {bool(final_output)}")
        print(f"   query_error: {bool(query_error)}")
        print(f"   retry_count: {retry_count}")
        print(f"   has_empty_results: {has_empty_results}")
        if query_error:
            print(f"   error_msg: {query_error[:100]}...")
        sys.stdout.flush()
        
        # Absolute terminator: if final_output is set, always go to output
        if final_output:
            _vprint(OutputFormatter.info("  → Routing to OUTPUT (final_output set)"))
            print(f"🔍 [ROUTING] → OUTPUT (final_output set)")
            sys.stdout.flush()
            return "output"
        
        # Case 1: Error occurred and retry limit not reached - self-heal by retrying
        if query_error and retry_count < 3:
            _vprint(OutputFormatter.warning(f"  → Routing to QUERY_BUILDER (retry {retry_count}/3)"))
            print(f"🔍 [ROUTING] → QUERY_BUILDER (retry {retry_count}/3)")
            sys.stdout.flush()
            return "query_builder"
        
        # Case 2: Retry limit exceeded - output error
        if query_error and retry_count >= 3:
            _vprint(OutputFormatter.error(f"  → Routing to OUTPUT (max retries reached: {retry_count})"))
            print(f"🔍 [ROUTING] → OUTPUT (max retries reached: {retry_count})")
            sys.stdout.flush()
            return "output"
        
        # Case 3: Empty results - generate insight explanation
        if has_empty_results:
            _vprint(OutputFormatter.info("  → Routing to INSIGHT_GENERATOR (empty results)"))
            print(f"🔍 [ROUTING] → INSIGHT_GENERATOR (empty results)")
            sys.stdout.flush()
            return "insight_generator"
        
        # Case 4: Success with data - analyze it
        _vprint(OutputFormatter.success("  → Routing to ANALYSIS (success with data)"))
        print(f"🔍 [ROUTING] → ANALYSIS (success with data)")
        sys.stdout.flush()
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
            "discovery_count": 0,  # Track discovery queries to prevent infinite loops
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
        
        # Set recursion limit to prevent infinite loops
        config = {"recursion_limit": 50}  # Increase from default 25
        result = self.app.invoke(initial_state, config)
        return result

