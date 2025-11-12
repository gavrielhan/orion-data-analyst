"""Focused tests for ApprovalNode."""

import os
import sys
import types
import unittest

# Ensure project root is on sys.path so `src` can be imported when running via `python tests/...`.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Stub external dependencies so ApprovalNode can be imported without
# requiring the real Google libraries during unit tests.
# ---------------------------------------------------------------------------
if "google" not in sys.modules:
    google_module = types.ModuleType("google")
    sys.modules["google"] = google_module
else:
    google_module = sys.modules["google"]

generativeai_module = types.ModuleType("google.generativeai")
generativeai_module.configure = lambda **kwargs: None
generativeai_module.GenerativeModel = lambda *args, **kwargs: None
google_module.generativeai = generativeai_module
sys.modules["google.generativeai"] = generativeai_module

cloud_module = types.ModuleType("google.cloud")
bigquery_module = types.ModuleType("google.cloud.bigquery")

class _DummyQueryJob:
    total_bytes_processed = 0

class _DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    def query(self, *args, **kwargs):
        return _DummyQueryJob()

bigquery_module.Client = _DummyClient
cloud_module.bigquery = bigquery_module
sys.modules["google.cloud"] = cloud_module
sys.modules["google.cloud.bigquery"] = bigquery_module

# Import after stubbing external modules
from src.agent.nodes import ApprovalNode
from src.agent.state import AgentState


def _base_state(**overrides) -> AgentState:
    """Helper to build a minimal AgentState for approval tests."""
    state: AgentState = {
        "user_query": "test",
        "query_intent": "",
        "schema_context": None,
        "schema_cache_timestamp": None,
        "sql_query": "SELECT 1",
        "discovery_query": None,
        "discovery_result": None,
        "discovery_count": 0,
        "validation_passed": True,
        "estimated_cost_gb": 0.0,
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
        "approval_granted": None,
        "final_output": "",
        "retry_count": 0,
        "execution_time_sec": None,
        "error_history": None,
        "_verbose": False,
    }
    state.update(overrides)
    return state


class TestApprovalNode(unittest.TestCase):
    """Unit tests dedicated to ApprovalNode behaviour."""

    def test_below_threshold_skips_approval(self):
        """Queries under the threshold should not require approval."""
        state = _base_state(estimated_cost_gb=1.5)
        result = ApprovalNode.execute(state)

        self.assertFalse(result["requires_approval"])
        self.assertTrue(result["approval_granted"])
        self.assertIsNone(result["approval_reason"])

    def test_above_threshold_requests_approval(self):
        """Queries over the threshold should request user approval."""
        state = _base_state(estimated_cost_gb=10.25)
        print("\n[TEST] Approval required for high-cost query. Please respond in the prompt.")
        result = ApprovalNode.execute(state)

        self.assertTrue(result["requires_approval"])
        self.assertIn("10.25", result["approval_reason"])
        self.assertIsNotNone(result["approval_granted"])


if __name__ == "__main__":
    unittest.main()


