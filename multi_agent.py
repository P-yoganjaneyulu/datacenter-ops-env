"""Simple multi-agent team wrappers used by app.py endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from grader import HeuristicAgent, RandomAgent
from models import DataCenterAction


@dataclass
class MultiAgentTeam:
    """Thin team adapter with shared policy for current agent turn."""

    policy_type: str = "heuristic"
    total_reward: float = 0.0
    steps: int = 0
    actions_taken: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.policy_type == "random":
            self._policy = RandomAgent(seed=42)
        else:
            self._policy = HeuristicAgent()

    def reset(self) -> None:
        self.total_reward = 0.0
        self.steps = 0
        self.actions_taken.clear()
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def select_action(self, observation) -> DataCenterAction:
        action = self._policy.select_action(observation)
        self.actions_taken.append(action.action_type.value)
        return action

    def update_all(self, observation, action: DataCenterAction, reward: float) -> None:
        self.total_reward += reward
        self.steps += 1

    def get_team_metrics(self) -> Dict[str, float]:
        return {
            "steps": self.steps,
            "avg_reward_per_step": (self.total_reward / self.steps) if self.steps else 0.0,
            "actions_logged": len(self.actions_taken),
        }


def create_team(agent_type: str = "heuristic") -> MultiAgentTeam:
    return MultiAgentTeam(policy_type=agent_type)
