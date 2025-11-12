"""Simple query cache with TTL for performance optimization."""

import hashlib
import json
import time
import pickle
import base64
from pathlib import Path
from typing import Optional, Any
import pandas as pd


class QueryCache:
    """
    In-memory cache with disk persistence for query results.
    Provides instant responses for repeated queries.
    """
    
    def __init__(self, cache_dir: Path = None, ttl_seconds: int = 5 * 3600):
        self.ttl = ttl_seconds
        self.cache_dir = cache_dir or (Path.home() / ".orion" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}  # In-memory for speed
        self._last_cleanup = 0.0
        # Clean up immediately on startup to enforce TTL across sessions
        self.enforce_ttl(force=True)
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def enforce_ttl(self, force: bool = False) -> None:
        """
        Remove expired cache entries from memory and disk.
        Runs automatically but can be forced when the agent becomes active.
        """
        now = time.time()
        cleanup_interval = max(300, min(self.ttl // 4, 3600))
        
        if not force and (now - self._last_cleanup) < cleanup_interval:
            return
        
        self._last_cleanup = now
        
        # Clean in-memory cache
        expired_keys = [
            key for key, entry in list(self.memory_cache.items())
            if now - entry.get('timestamp', 0) >= self.ttl
        ]
        for key in expired_keys:
            self.memory_cache.pop(key, None)
        
        # Clean persisted cache files
        for cache_file in list(self.cache_dir.glob("*.pkl")):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                if now - entry.get('timestamp', 0) >= self.ttl:
                    cache_file.unlink()
            except Exception:
                continue
        
        for cache_file in list(self.cache_dir.glob("*.json")):
            try:
                with open(cache_file, 'r') as f:
                    entry = json.load(f)
                timestamp = entry.get('timestamp')
                if timestamp is None or now - timestamp >= self.ttl:
                    cache_file.unlink()
            except Exception:
                continue
    
    def get(self, query: str) -> Optional[Any]:
        """Retrieve cached result if valid."""
        # Periodic cleanup for TTL enforcement
        self.enforce_ttl()
        
        key = self._get_cache_key(query)
        
        # Check memory first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                data = entry['data']
                # Validate DataFrames in cached data
                if isinstance(data, dict) and 'query_result' in data:
                    query_result = data.get('query_result')
                    if query_result is not None:
                        if isinstance(query_result, str):
                            # DataFrame was corrupted, remove from cache
                            del self.memory_cache[key]
                            return None
                        if not isinstance(query_result, pd.DataFrame):
                            # Not a DataFrame, might be corrupted
                            del self.memory_cache[key]
                            return None
                return data
            else:
                del self.memory_cache[key]
        
        # Check disk cache - try pickle first (new format), then JSON (old format)
        cache_file = self.cache_dir / f"{key}.pkl"
        if not cache_file.exists():
            # Fallback to old JSON format
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        entry = json.load(f)
                    
                    if time.time() - entry['timestamp'] < self.ttl:
                        # JSON format doesn't preserve DataFrames - skip corrupted cache
                        return None
                    else:
                        cache_file.unlink()  # Delete expired
                except Exception as e:
                    # Log cache read error but don't fail - cache is optional
                    import logging
                    logging.getLogger(__name__).debug(f"Cache read error (JSON format): {e}")
                    pass
                return None
        
        if cache_file.exists() and cache_file.suffix == '.pkl':
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                if time.time() - entry['timestamp'] < self.ttl:
                    # Restore to memory
                    self.memory_cache[key] = entry
                    # Validate and restore DataFrame if needed
                    data = entry['data']
                    if isinstance(data, dict) and 'query_result' in data:
                        query_result = data.get('query_result')
                        if isinstance(query_result, str):
                            # DataFrame was serialized incorrectly, skip cache
                            return None
                        # Ensure it's a DataFrame if it should be
                        if query_result is not None and not isinstance(query_result, pd.DataFrame):
                            # Try to restore from string representation (fallback)
                            return None
                    return data
                else:
                    cache_file.unlink()  # Delete expired
            except Exception as e:
                # Log cache read error but don't fail - cache is optional
                import logging
                logging.getLogger(__name__).debug(f"Cache read error (pickle format): {e}")
                pass
        
        return None
    
    def set(self, query: str, data: Any) -> None:
        """Cache query result."""
        # Cleanup before inserting new data to keep cache bounded
        self.enforce_ttl()
        
        key = self._get_cache_key(query)
        
        # Validate data before caching - ensure DataFrames are preserved
        if isinstance(data, dict) and 'query_result' in data:
            query_result = data.get('query_result')
            if query_result is not None and not isinstance(query_result, pd.DataFrame):
                # Don't cache if query_result is not a DataFrame (could be corrupted)
                return
        
        entry = {
            'timestamp': time.time(),
            'data': data,
            'query': query
        }
        
        # Store in memory
        self.memory_cache[key] = entry
        
        # Persist to disk using pickle (preserves DataFrames properly)
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            # Log cache write error but don't fail - cache is optional
            import logging
            logging.getLogger(__name__).debug(f"Cache write error: {e}")
            pass  # Silent fail on cache write
    
    def clear(self) -> None:
        """Clear all cache."""
        self.memory_cache.clear()
        # Clear both JSON (old format) and pickle (new format) files
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

