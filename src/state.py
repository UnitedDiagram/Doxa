"""Re-export shared research state for backwards compatibility."""

from __future__ import annotations

from doxa_shared.types.state import ResearchState, create_initial_state

__all__ = ["ResearchState", "create_initial_state"]
