"""Tests for doxa_shared.utils.cache module."""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest
from doxa_shared.utils.cache import (
    CacheStats,
    InMemoryCache,
    cached_fetch,
    get_cache,
)


class TestCacheStats:
    """CacheStats dataclass tests."""

    def test_initial_values(self) -> None:
        """New stats start at zero."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0

    def test_total(self) -> None:
        """Total is sum of hits and misses."""
        stats = CacheStats(hits=3, misses=7)
        assert stats.total == 10

    def test_hit_rate(self) -> None:
        """Hit rate is percentage of hits over total."""
        stats = CacheStats(hits=8, misses=2)
        assert stats.hit_rate == pytest.approx(80.0)

    def test_hit_rate_no_lookups(self) -> None:
        """Hit rate is 0 when no lookups occurred."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0


class TestInMemoryCache:
    """InMemoryCache unit tests."""

    def test_get_empty_returns_none(self) -> None:
        """Getting a missing key returns None."""
        cache = InMemoryCache()
        assert cache.get("missing") is None

    def test_set_then_get(self) -> None:
        """Stored value is retrievable."""
        cache = InMemoryCache()
        cache.set("k", "v", ttl_seconds=60)
        assert cache.get("k") == "v"

    def test_set_overwrites(self) -> None:
        """Setting the same key replaces the value."""
        cache = InMemoryCache()
        cache.set("k", "old", ttl_seconds=60)
        cache.set("k", "new", ttl_seconds=60)
        assert cache.get("k") == "new"

    @patch("doxa_shared.utils.cache.time")
    def test_ttl_expiration(self, mock_time: object) -> None:
        """Expired entries return None and are deleted."""
        mock_time.monotonic.side_effect = [  # type: ignore[union-attr]
            100.0,   # set → expiry = 101
            100.5,   # get → 100.5 < 101 → hit
            101.5,   # get → 101.5 > 101 → expired
        ]
        cache = InMemoryCache()
        cache.set("k", "v", ttl_seconds=1)
        assert cache.get("k") == "v"
        assert cache.get("k") is None

    def test_delete(self) -> None:
        """Delete removes the entry."""
        cache = InMemoryCache()
        cache.set("k", "v", ttl_seconds=60)
        cache.delete("k")
        assert cache.get("k") is None

    def test_delete_missing_key_no_error(self) -> None:
        """Deleting a missing key is a no-op."""
        cache = InMemoryCache()
        cache.delete("nope")

    def test_clear(self) -> None:
        """Clear removes all entries."""
        cache = InMemoryCache()
        cache.set("a", 1, ttl_seconds=60)
        cache.set("b", 2, ttl_seconds=60)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_max_entries_eviction(self) -> None:
        """Oldest entry is evicted when cache is full."""
        cache = InMemoryCache(max_entries=2)
        cache.set("a", 1, ttl_seconds=60)
        cache.set("b", 2, ttl_seconds=60)
        cache.set("c", 3, ttl_seconds=60)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_unlimited_entries(self) -> None:
        """max_entries=0 allows unlimited storage."""
        cache = InMemoryCache(max_entries=0)
        for i in range(2000):
            cache.set(f"k{i}", i, ttl_seconds=60)
        assert cache.get("k0") == 0
        assert cache.get("k1999") == 1999

    def test_thread_safety(self) -> None:
        """Concurrent set/get must not corrupt data."""
        cache = InMemoryCache(max_entries=0)
        errors: list[str] = []

        def worker(tid: int) -> None:
            for i in range(50):
                key = f"t{tid}-{i}"
                val = tid * 1000 + i
                cache.set(key, val, ttl_seconds=60)
                got = cache.get(key)
                if got != val:
                    errors.append(
                        f"{key}: expected {val}, got {got}"
                    )

        threads = [
            threading.Thread(target=worker, args=(t,))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"


class TestCachedFetch:
    """cached_fetch helper tests."""

    def test_miss_calls_fetcher(self) -> None:
        """On miss, fetcher is called and result cached."""
        cache = InMemoryCache()
        calls: list[int] = []

        def fetcher() -> str:
            calls.append(1)
            return "data"

        result = cached_fetch(cache, "k", fetcher, ttl_seconds=60)
        assert result == "data"
        assert len(calls) == 1
        assert cache.stats.misses == 1

    def test_hit_skips_fetcher(self) -> None:
        """On hit, fetcher is not called."""
        cache = InMemoryCache()
        cache.set("k", "cached", ttl_seconds=60)
        calls: list[int] = []

        def fetcher() -> str:
            calls.append(1)
            return "fresh"

        result = cached_fetch(cache, "k", fetcher, ttl_seconds=60)
        assert result == "cached"
        assert len(calls) == 0
        assert cache.stats.hits == 1

    def test_fetcher_exception_not_cached(self) -> None:
        """Fetcher errors are re-raised, value not cached."""
        cache = InMemoryCache()

        def bad_fetcher() -> str:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            cached_fetch(cache, "k", bad_fetcher, ttl_seconds=60)

        assert cache.get("k") is None
        assert cache.stats.misses == 1


class TestGetCache:
    """get_cache factory tests."""

    def test_returns_same_instance(self) -> None:
        """Singleton returns the same object on repeated calls."""
        import doxa_shared.utils.cache as mod

        mod._cache_instance = None
        c1 = get_cache()
        c2 = get_cache()
        assert c1 is c2
        mod._cache_instance = None
