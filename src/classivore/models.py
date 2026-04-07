#!/usr/bin/env python3
"""Shared data models for inter-module communication.

Dataclasses used by the agent and coverage analysis. Existing modules
continue to use dicts internally — these models provide typed interfaces
at module boundaries.
"""

from dataclasses import dataclass, field


@dataclass
class CategoryGap:
    """A taxonomy category that needs more labeled pages."""
    name: str
    current_count: int
    target_count: int
    deficit: int
    tier1_name: str


@dataclass
class CoverageReport:
    """Snapshot of label coverage across the taxonomy."""
    total_categories: int
    covered_categories: int
    satisfied_categories: int
    total_labeled_pages: int
    gaps: list[CategoryGap]
    timestamp: str

    @property
    def coverage_pct(self) -> float:
        if self.total_categories == 0:
            return 0.0
        return self.satisfied_categories / self.total_categories * 100

    @property
    def worst_gaps(self) -> list[CategoryGap]:
        """Top 50 gaps — categories with fewest labels."""
        return self.gaps[:50]


@dataclass
class IterationPlan:
    """Thin audit record of what the agent intended for one iteration."""
    iteration: int
    target_categories: list[str]
    use_llm_queries: bool = False


@dataclass
class IterationResult:
    """Outcome of one agent iteration."""
    iteration: int
    pages_collected: int
    pages_labeled: int
    categories_satisfied_before: int
    categories_satisfied_after: int
    gaps_before: int
    gaps_after: int
    collection_summary: dict = field(default_factory=dict)
    labeling_summary: dict = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Stop conditions and budget for the agent."""
    max_iterations: int = 10
    target_per_category: int = 50
    min_yield_per_iteration: int = 5
    max_consecutive_zero_yield: int = 2
