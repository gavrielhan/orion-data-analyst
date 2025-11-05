"""Rate limiter for API calls to prevent quota exhaustion."""

import time
from collections import deque
from typing import Optional


# Global shared rate limiter for all Gemini API calls
_global_rate_limiter = None


def get_global_rate_limiter(max_calls: int = 30, window_seconds: int = 60):
    """
    Get the global shared rate limiter instance.
    Default: 30 calls per 60 seconds (very conservative, leaves large safety margin for Gemini's 60/min limit)
    This prevents hitting rate limits even if previous queries exhausted quota.
    """
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter(max_calls, window_seconds)
    return _global_rate_limiter


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    Prevents hitting Gemini API rate limits.
    """
    
    def __init__(self, max_calls: int = 60, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum calls allowed in window
            window_seconds: Time window in seconds
        """
        self.max_calls = max_calls
        self.window = window_seconds
        self.calls = deque()  # Timestamps of recent calls
    
    def wait_if_needed(self, verbose: bool = False) -> Optional[float]:
        """
        Wait if rate limit would be exceeded.
        Returns wait time in seconds, or None if no wait needed.
        """
        now = time.time()
        
        # Remove calls outside the window
        while self.calls and now - self.calls[0] > self.window:
            self.calls.popleft()
        
        # Check current call count
        current_calls = len(self.calls)
        
        # Check if we're at the limit (with small buffer to be extra safe)
        if current_calls >= self.max_calls:
            # Calculate wait time until oldest call expires
            oldest_call = self.calls[0]
            wait_time = self.window - (now - oldest_call) + 0.5  # Add 0.5s buffer
            
            if wait_time > 0:
                if verbose or current_calls >= self.max_calls:
                    print(f"⏱️  Rate limiter: {current_calls}/{self.max_calls} calls used. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                # Remove old calls after waiting
                now = time.time()
                while self.calls and now - self.calls[0] > self.window:
                    self.calls.popleft()
                # Record this call
                self.calls.append(now)
                return wait_time
        
        # Record this call BEFORE making API call
        self.calls.append(now)
        
        if verbose:
            print(f"⏱️  Rate limiter: {current_calls + 1}/{self.max_calls} calls used")
        
        return None
    
    def get_status(self) -> dict:
        """Get current rate limiter status."""
        now = time.time()
        
        # Remove old calls
        while self.calls and now - self.calls[0] > self.window:
            self.calls.popleft()
        
        return {
            "current_calls": len(self.calls),
            "max_calls": self.max_calls,
            "window_seconds": self.window,
            "can_proceed": len(self.calls) < self.max_calls
        }
    
    def can_proceed(self) -> bool:
        """Check if we can make a call without waiting."""
        now = time.time()
        
        # Remove old calls
        while self.calls and now - self.calls[0] > self.window:
            self.calls.popleft()
        
        return len(self.calls) < self.max_calls
    
    def reset(self):
        """Reset rate limiter."""
        self.calls.clear()

