"""TTL-based caching infrastructure for Doxa data pipelines.

Provides a ``CacheBackend`` Protocol for pluggable backends and an
``InMemoryCache`` default implementation with TTL expiration.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

TTL_PRICE_DATA: int = 86_400
"""24 hours — current quotes and price history."""

TTL_FINANCIALS: int = 7_776_000
"""90 days — income statements, balance sheets, cash flows."""

TTL_SEC_FILINGS: int = 15_552_000
"""180 days — SEC EDGAR 10-K/10-Q filings."""


@dataclass
class CacheStats:
    """Tracks cache hit/miss counts for observability."""

    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        """Total lookups."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Hit rate as percentage (0-100). Returns 0.0 if no lookups."""
        if self.total == 0:
            return 0.0
        return self.hits / self.total * 100


class CacheBackend(Protocol):
    """Protocol for pluggable cache backends."""

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` on miss/expiry."""
        ...

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store *value* under *key* with a TTL in seconds."""
        ...

    def delete(self, key: str) -> None:
        """Remove a single key."""
        ...

    def clear(self) -> None:
        """Remove all entries."""
        ...


class InMemoryCache:
    """Thread-safe in-memory cache with TTL-based expiration.

    Args:
        max_entries: Maximum stored entries. Oldest evicted when full.
            Use ``0`` for unlimited.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        """Initialise cache with optional size limit."""
        self._store: dict[str, tuple[Any, float]] = {}
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` on miss/expiry."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expiry = entry
            if time.monotonic() > expiry:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store *value* under *key* with a TTL in seconds."""
        with self._lock:
            if (
                self._max_entries > 0
                and key not in self._store
                and len(self._store) >= self._max_entries
            ):
                self._evict_oldest()
            self._store[key] = (
                value,
                time.monotonic() + ttl_seconds,
            )

    def delete(self, key: str) -> None:
        """Remove a single key (no-op if absent)."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()

    def _evict_oldest(self) -> None:
        """Evict the first-inserted entry (FIFO). Caller must hold lock."""
        if self._store:
            oldest = next(iter(self._store))
            del self._store[oldest]


_cache_instance: InMemoryCache | None = None
_cache_lock = threading.Lock()


def get_cache(max_entries: int = 1000) -> InMemoryCache:
    """Return the module-level singleton ``InMemoryCache``.

    This function implements the singleton pattern — only the **first**
    call creates the cache instance. Subsequent calls return the same
    instance and ignore the ``max_entries`` parameter.

    Args:
        max_entries: Size limit (only honored on first call).

    Returns:
        The shared cache instance.

    Note:
        Currently always returns ``InMemoryCache``. The ``CACHE_BACKEND``
        environment variable exists for future Redis support (Epic 4-5)
        but is not yet used. If you need different cache sizes for
        different contexts, create separate ``InMemoryCache()`` instances
        directly instead of using this singleton factory.
    """
    global _cache_instance  # noqa: PLW0603
    with _cache_lock:
        if _cache_instance is None:
            _cache_instance = InMemoryCache(max_entries=max_entries)
        return _cache_instance


def cached_fetch[T](
    cache: InMemoryCache,
    key: str,
    fetcher: Callable[[], T],
    ttl_seconds: int,
) -> T:
    """Cache-aside helper: return cached value or call *fetcher*.

    Args:
        cache: Cache backend instance.
        key: Cache key.
        fetcher: Produces the value on a cache miss.
        ttl_seconds: TTL for the cached entry.

    Returns:
        The cached or freshly-fetched value.

    Raises:
        Exception: Re-raises any exception from *fetcher*
            (the value is **not** cached on error).
    """
    hit = cache.get(key)
    if hit is not None:
        cache.stats.hits += 1
        logger.debug("Cache hit: %s", key)
        return hit  # type: ignore[no-any-return]

    cache.stats.misses += 1
    logger.debug("Cache miss: %s", key)
    value = fetcher()
    cache.set(key, value, ttl_seconds)
    return value
