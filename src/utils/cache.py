"""Simple query cache with TTL for performance optimization."""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Any


class QueryCache:
    """
    In-memory cache with disk persistence for query results.
    Provides instant responses for repeated queries.
    """
    
    def __init__(self, cache_dir: Path = None, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self.cache_dir = cache_dir or (Path.home() / ".orion" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}  # In-memory for speed
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Any]:
        """Retrieve cached result if valid."""
        key = self._get_cache_key(query)
        
        # Check memory first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['data']
            else:
                del self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    entry = json.load(f)
                
                if time.time() - entry['timestamp'] < self.ttl:
                    # Restore to memory
                    self.memory_cache[key] = entry
                    return entry['data']
                else:
                    cache_file.unlink()  # Delete expired
            except Exception:
                pass
        
        return None
    
    def set(self, query: str, data: Any) -> None:
        """Cache query result."""
        key = self._get_cache_key(query)
        entry = {
            'timestamp': time.time(),
            'data': data,
            'query': query
        }
        
        # Store in memory
        self.memory_cache[key] = entry
        
        # Persist to disk (async in production, sync for simplicity)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(entry, f, default=str)
        except Exception:
            pass  # Silent fail on cache write
    
    def clear(self) -> None:
        """Clear all cache."""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

