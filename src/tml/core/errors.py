from __future__ import annotations


class TmlError(RuntimeError):
    """Base user-facing TheML error."""


class ContextError(TmlError):
    """Raised when current project/run context cannot be resolved."""
