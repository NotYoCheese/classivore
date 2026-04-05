#!/usr/bin/env python3
"""Agent state persistence and stop condition evaluation.

Tracks iteration history, cumulative progress, and determines when
the agent should stop. Persists atomically for crash recovery.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from classivore.models import AgentConfig, IterationPlan, IterationResult
from classivore.persistence import atomic_json_save


class AgentState:
    """Persists agent run progress across crashes."""

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "agent_state.json"
        self.started_at: str | None = None
        self.last_checkpoint_at: str | None = None
        self.iterations: list[dict] = []

        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.started_at = data.get("started_at")
            self.last_checkpoint_at = data.get("last_checkpoint_at")
            self.iterations = data.get("iterations", [])

    def save(self) -> None:
        """Atomically save state to disk."""
        now = datetime.now(timezone.utc).isoformat()
        if not self.started_at:
            self.started_at = now
        self.last_checkpoint_at = now

        data = {
            "started_at": self.started_at,
            "last_checkpoint_at": self.last_checkpoint_at,
            "iterations": self.iterations,
        }
        atomic_json_save(data, self.state_file, directory=self.state_dir)

    def start_iteration(self, plan: IterationPlan) -> None:
        """Record the start of an iteration."""
        self.iterations.append({
            "iteration": plan.iteration,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "plan": asdict(plan),
            "result": None,
        })
        self.save()

    def complete_iteration(self, result: IterationResult) -> None:
        """Record iteration outcome."""
        if self.iterations and self.iterations[-1]["result"] is None:
            self.iterations[-1]["completed_at"] = datetime.now(timezone.utc).isoformat()
            self.iterations[-1]["result"] = asdict(result)
        self.save()

    def current_iteration(self) -> int:
        """Current iteration number (0-indexed).

        Incomplete iterations (started but not completed) are not counted,
        so they get retried on resume.
        """
        # Drop incomplete trailing iteration so it gets retried
        if self.iterations and self.iterations[-1]["result"] is None:
            self.iterations.pop()
            self.save()
        return len(self.iterations)

    def should_stop(self, config: AgentConfig) -> tuple[bool, str]:
        """Evaluate stop conditions.

        Returns:
            Tuple of (should_stop, reason).
        """
        completed = [i for i in self.iterations if i.get("result") is not None]

        # Max iterations
        if len(completed) >= config.max_iterations:
            return True, f"max iterations reached ({config.max_iterations})"

        # All categories satisfied (check last iteration)
        if completed:
            last = completed[-1]["result"]
            if last["gaps_after"] == 0:
                return True, "all categories satisfied"

        # Consecutive zero yield
        if len(completed) >= config.max_consecutive_zero_yield:
            recent = completed[-config.max_consecutive_zero_yield:]
            if all(i["result"]["pages_labeled"] == 0 for i in recent):
                return True, (
                    f"{config.max_consecutive_zero_yield} consecutive iterations "
                    f"with zero pages labeled"
                )

        # Below minimum yield (only check if we have at least one completed)
        if completed:
            last_yield = completed[-1]["result"]["pages_labeled"]
            if 0 < last_yield < config.min_yield_per_iteration:
                return True, (
                    f"yield ({last_yield}) below minimum "
                    f"({config.min_yield_per_iteration})"
                )

        return False, ""

    def summary(self) -> dict:
        """Summary for CLI output."""
        completed = [i for i in self.iterations if i.get("result") is not None]
        total_collected = sum(i["result"]["pages_collected"] for i in completed)
        total_labeled = sum(i["result"]["pages_labeled"] for i in completed)

        return {
            "iterations_completed": len(completed),
            "total_pages_collected": total_collected,
            "total_pages_labeled": total_labeled,
            "started_at": self.started_at,
            "last_checkpoint_at": self.last_checkpoint_at,
        }
