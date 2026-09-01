from __future__ import annotations


class AnalysisCanceled(RuntimeError):
    """Raised when a user requests cancellation of an active analysis."""


class TaskOwnershipLost(RuntimeError):
    """Raised when a worker no longer owns the lease for an analysis task."""

