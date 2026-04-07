"""
DataCenterOps — OpenEnv Environment Server
===========================================
Wraps the Gymnasium-style DataCenterOpsEnv simulation in the
openenv.core.env_server.interfaces.Environment interface, enabling:

  - WebSocket-based persistent sessions via FastAPI
  - Concurrent multi-session support
  - Typed Pydantic Action/Observation
  - state() endpoint for full readable state

Used by server/app.py via create_app().
"""

import sys
import os
from uuid import uuid4
from typing import Optional

# Allow importing from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.interfaces import Environment

from models import (
    DataCenterAction, DataCenterObservation, DataCenterState,
    TaskTier
)
from environment import DataCenterOpsEnv  # Gymnasium simulation engine


class DataCenterEnvironment(Environment[DataCenterAction, DataCenterObservation, DataCenterState]):
    """
    OpenEnv-compliant environment for DataCenterOps.

    Each WebSocket session gets its own instance (SUPPORTS_CONCURRENT_SESSIONS=True).
    The task difficulty is read from the DATACENTER_TASK environment variable
    (default: "easy"). Overridable per-reset via options dict.

    Coordination patterns implemented:
        ALERT         → watcher broadcasts anomaly to all agents
        INVESTIGATE   → watcher runs deep sensor analysis
        DIAGNOSE      → responder identifies root cause
        FIX           → responder attempts automated remediation
        REQUEST_HELP  → responder asks coordinator for technician
        DISPATCH      → coordinator sends technician to site
        ESCALATE      → coordinator notifies management
        RESOLVE       → coordinator closes incident
        DISAGREE      → coordinate_message with disagreement on hard task
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        task = os.environ.get("DATACENTER_TASK", "easy")
        assert task in ("easy", "medium", "hard"), f"DATACENTER_TASK must be easy|medium|hard, got: {task}"
        self._task_tier = TaskTier(task)
        self._env: Optional[DataCenterOpsEnv] = None
        self._episode_id: Optional[str] = None
        self._terminated = False
        self._truncated = False

    # ------------------------------------------------------------------
    # OpenEnv Interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> DataCenterObservation:
        """Reset environment and return initial observation."""
        # Allow task override via kwargs
        task = kwargs.get("task", None)
        if task:
            self._task_tier = TaskTier(task)
        
        self._env = DataCenterOpsEnv(task_tier=self._task_tier, seed=seed)
        obs = self._env.reset(seed=seed)

        self._episode_id = episode_id or str(uuid4())
        self._terminated = False
        self._truncated = False

        return obs

    def step(self, action: DataCenterAction) -> DataCenterObservation:
        """Execute action and return observation with reward."""
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._terminated = terminated
        self._truncated = truncated

        # Update observation with done flags
        obs.done = terminated
        obs.truncated = truncated
        
        return obs

    @property
    def state(self) -> DataCenterState:
        """Return full readable environment state."""
        if self._env is None:
            # Pre-reset state
            return DataCenterState(
                episode_id=self._episode_id or "",
                task_tier=self._task_tier,
                seed=None,
                step_number=0,
                max_steps=0,
                current_agent=None,
                equipment=[],
                incidents=[],
                technicians=[],
                agent_states={},
                message_history=[],
                all_evidence=[],
                metrics=None,
                total_reward=0.0,
                coordination_score=0.0,
                cascade_count=0,
            )

        return self._env.state()
