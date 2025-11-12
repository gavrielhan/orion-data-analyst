"""Token usage tracking with persistence for rate monitoring."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class TokenTracker:
    """
    Tracks Gemini API token usage with timestamps.
    Persists to disk for cross-session tracking.
    """
    
    def __init__(self, log_file: Path = None):
        self.log_file = log_file or (Path.home() / ".orion" / "token_usage.jsonl")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.current_session_tokens = {
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0
        }
    
    def record_usage(
        self,
        prompt_tokens: int = 0,
        response_tokens: int = 0,
        total_tokens: Optional[int] = None
    ) -> None:
        """Record token usage with timestamp."""
        if total_tokens is None:
            total_tokens = prompt_tokens + response_tokens
        
        # Update session counters
        self.current_session_tokens["prompt_tokens"] += prompt_tokens
        self.current_session_tokens["response_tokens"] += response_tokens
        self.current_session_tokens["total_tokens"] += total_tokens
        
        # Persist to disk
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass  # Silent fail on logging errors
    
    def get_current_query_usage(self) -> Dict[str, int]:
        """Get token usage for current query (since last reset)."""
        return self.current_session_tokens.copy()
    
    def get_usage_last_minute(self) -> Dict[str, int]:
        """Calculate token usage in the last 60 seconds from persisted log."""
        self._cleanup_on_read()
        return self._get_usage_since(datetime.now() - timedelta(seconds=60))
    
    def get_usage_last_hour(self) -> Dict[str, int]:
        """Calculate token usage in the last hour from persisted log."""
        return self._get_usage_since(datetime.now() - timedelta(hours=1))
    
    def _get_usage_since(self, cutoff: datetime) -> Dict[str, int]:
        """Calculate token usage since a given cutoff time."""
        usage = {"prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0, "call_count": 0}
        
        if not self.log_file.exists():
            return usage
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if datetime.fromisoformat(entry["timestamp"]) >= cutoff:
                            usage["prompt_tokens"] += entry.get("prompt_tokens", 0)
                            usage["response_tokens"] += entry.get("response_tokens", 0)
                            usage["total_tokens"] += entry.get("total_tokens", 0)
                            usage["call_count"] += 1
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
        except Exception:
            pass
        
        return usage
    
    def _cleanup_on_read(self) -> None:
        """Remove entries older than 1 hour."""
        if not self.log_file.exists():
            return
        
        cutoff = datetime.now() - timedelta(hours=1)
        temp_file = self.log_file.with_suffix('.tmp')
        
        try:
            kept_lines = []
            total_lines = 0
            with open(self.log_file, 'r') as f:
                for line in f:
                    total_lines += 1
                    try:
                        entry = json.loads(line.strip())
                        if datetime.fromisoformat(entry["timestamp"]) >= cutoff:
                            kept_lines.append(line)
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
            
            # Only rewrite if we filtered something
            if len(kept_lines) < total_lines:
                with open(temp_file, 'w') as f_out:
                    f_out.writelines(kept_lines)
                temp_file.replace(self.log_file)
        except Exception:
            if temp_file.exists():
                temp_file.unlink()
    
    def reset_query_counter(self) -> None:
        """Reset current query token counter (call at start of new query)."""
        self.current_session_tokens = {"prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0}


# Global singleton
_global_token_tracker = None


def get_token_tracker() -> TokenTracker:
    """Get the global token tracker instance."""
    global _global_token_tracker
    if _global_token_tracker is None:
        _global_token_tracker = TokenTracker()
    return _global_token_tracker


def track_llm_call(model, prompt: str, **generation_kwargs):
    """
    Helper to track token usage for an LLM call.
    Returns the response object.
    """
    tracker = get_token_tracker()
    
    # Estimate prompt tokens
    estimated_tokens = 0
    try:
        token_count = model.count_tokens(prompt)
        estimated_tokens = getattr(token_count, "total_tokens", 0)
    except Exception:
        pass
    
    # Make the LLM call
    response = model.generate_content(prompt, **generation_kwargs)
    
    # Record actual usage from response metadata
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        tracker.record_usage(
            prompt_tokens=getattr(usage_metadata, "prompt_token_count", 0),
            response_tokens=getattr(usage_metadata, "candidates_token_count", 0),
            total_tokens=getattr(usage_metadata, "total_token_count", 0),
        )
    elif estimated_tokens > 0:
        tracker.record_usage(prompt_tokens=estimated_tokens, total_tokens=estimated_tokens)
    
    return response

