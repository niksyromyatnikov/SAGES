"""Importable runner API for SAGES.

Use this from notebooks instead of invoking the CLI.

Example:
    from runner import PipelineOptions, run_pipeline

    report = run_pipeline(PipelineOptions(repo=r"D:\path\to\repo", heuristic_activation=True))
    report
"""

from __future__ import annotations

from typing import Any, Dict

from main import PipelineOptions, run_pipeline

__all__ = ["PipelineOptions", "run_pipeline"]


def run(repo: str, **kwargs: Any) -> Dict[str, Any]:
    """Convenience wrapper: run_pipeline(PipelineOptions(repo=repo, **kwargs))."""

    return run_pipeline(PipelineOptions(repo=repo, **kwargs))
