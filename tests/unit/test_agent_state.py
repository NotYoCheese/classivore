#!/usr/bin/env python3
"""Tests for agent state persistence and stop conditions."""

import json

import pytest

from classivore.agent.state import AgentState
from classivore.models import AgentConfig, IterationPlan, IterationResult


def _make_plan(iteration=0):
    return IterationPlan(
        iteration=iteration,
        target_categories=["Sedan", "SUV"],
    )


def _make_result(iteration=0, collected=10, labeled=8, gaps_before=50, gaps_after=48):
    return IterationResult(
        iteration=iteration,
        pages_collected=collected,
        pages_labeled=labeled,
        categories_satisfied_before=0,
        categories_satisfied_after=2,
        gaps_before=gaps_before,
        gaps_after=gaps_after,
    )


class TestAgentState:
    """Test agent state persistence."""

    def test_init_fresh(self, tmp_path):
        state = AgentState(tmp_path / "agent")
        assert state.current_iteration() == 0
        assert state.iterations == []

    def test_persistence_roundtrip(self, tmp_path):
        state_dir = tmp_path / "agent"
        state = AgentState(state_dir)
        state.start_iteration(_make_plan(0))
        state.complete_iteration(_make_result(0))

        # Reload
        state2 = AgentState(state_dir)
        assert len(state2.iterations) == 1
        assert state2.iterations[0]["result"]["pages_collected"] == 10
        assert state2.started_at is not None

    def test_iteration_tracking(self, tmp_path):
        state = AgentState(tmp_path / "agent")

        state.start_iteration(_make_plan(0))
        # Incomplete iteration is dropped on current_iteration() (resume safety)
        assert state.current_iteration() == 0

        # Re-start and complete
        state.start_iteration(_make_plan(0))
        state.complete_iteration(_make_result(0))
        assert state.current_iteration() == 1
        assert state.iterations[0]["result"] is not None
        assert state.iterations[0]["completed_at"] is not None

    def test_summary(self, tmp_path):
        state = AgentState(tmp_path / "agent")
        state.start_iteration(_make_plan(0))
        state.complete_iteration(_make_result(0, collected=10, labeled=8))
        state.start_iteration(_make_plan(1))
        state.complete_iteration(_make_result(1, collected=5, labeled=3))

        summary = state.summary()
        assert summary["iterations_completed"] == 2
        assert summary["total_pages_collected"] == 15
        assert summary["total_pages_labeled"] == 11


class TestStopConditions:
    """Test agent stop condition evaluation."""

    def test_max_iterations(self, tmp_path):
        state = AgentState(tmp_path / "agent")
        config = AgentConfig(max_iterations=2)

        for i in range(2):
            state.start_iteration(_make_plan(i))
            state.complete_iteration(_make_result(i))

        should_stop, reason = state.should_stop(config, start_iteration=0)
        assert should_stop
        assert "max iterations" in reason

    def test_max_iterations_is_relative(self, tmp_path):
        """Previous iterations don't count toward the limit."""
        state = AgentState(tmp_path / "agent")
        config = AgentConfig(max_iterations=1)

        # 3 iterations from a previous session
        for i in range(3):
            state.start_iteration(_make_plan(i))
            state.complete_iteration(_make_result(i))

        # New session starts at iteration 3 — hasn't done any yet
        should_stop, reason = state.should_stop(config, start_iteration=3)
        assert not should_stop

    def test_all_satisfied(self, tmp_path):
        state = AgentState(tmp_path / "agent")
        config = AgentConfig()

        state.start_iteration(_make_plan(0))
        state.complete_iteration(_make_result(0, gaps_after=0))

        should_stop, reason = state.should_stop(config)
        assert should_stop
        assert "all categories satisfied" in reason

    def test_consecutive_zero_yield(self, tmp_path):
        state = AgentState(tmp_path / "agent")
        config = AgentConfig(max_consecutive_zero_yield=2)

        for i in range(2):
            state.start_iteration(_make_plan(i))
            state.complete_iteration(_make_result(i, labeled=0))

        should_stop, reason = state.should_stop(config)
        assert should_stop
        assert "zero pages labeled" in reason

    def test_below_min_yield(self, tmp_path):
        state = AgentState(tmp_path / "agent")
        config = AgentConfig(min_yield_per_iteration=5)

        state.start_iteration(_make_plan(0))
        state.complete_iteration(_make_result(0, labeled=2))

        should_stop, reason = state.should_stop(config)
        assert should_stop
        assert "below minimum" in reason

    def test_no_stop_when_progressing(self, tmp_path):
        state = AgentState(tmp_path / "agent")
        config = AgentConfig(max_iterations=10, min_yield_per_iteration=5)

        state.start_iteration(_make_plan(0))
        state.complete_iteration(_make_result(0, labeled=20, gaps_after=30))

        should_stop, reason = state.should_stop(config)
        assert not should_stop
        assert reason == ""

    def test_no_stop_on_first_run(self, tmp_path):
        state = AgentState(tmp_path / "agent")
        config = AgentConfig()

        should_stop, reason = state.should_stop(config)
        assert not should_stop

    def test_zero_yield_not_triggered_below_threshold(self, tmp_path):
        """One zero-yield iteration doesn't trigger stop with threshold of 2."""
        state = AgentState(tmp_path / "agent")
        config = AgentConfig(max_consecutive_zero_yield=2)

        state.start_iteration(_make_plan(0))
        state.complete_iteration(_make_result(0, labeled=0))

        should_stop, reason = state.should_stop(config)
        # labeled=0 but min_yield check: 0 < 0 is False, so no stop from min_yield
        # consecutive zero: only 1, need 2
        assert not should_stop

    def test_errored_iterations_excluded_from_zero_yield(self, tmp_path):
        """Errored iterations don't count toward the zero-yield streak —
        they signal a transient failure (API outage, credit exhaustion),
        not exhausted yield."""
        state = AgentState(tmp_path / "agent")
        config = AgentConfig(max_consecutive_zero_yield=2)

        for i in range(2):
            state.start_iteration(_make_plan(i))
            r = _make_result(i, labeled=0)
            r.errored = True
            state.complete_iteration(r)

        should_stop, reason = state.should_stop(config)
        assert not should_stop, (
            f"errored iterations triggered stop with reason: {reason!r}"
        )
