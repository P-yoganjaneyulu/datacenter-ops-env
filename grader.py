"""
DataCenterOps — Grading System
==============================

Comprehensive grading and benchmarking system following the SRE Incident
Environment's approach to deterministic scoring.

Features:
- Per-task grading with 0-1 scores
- Benchmark execution across all tiers
- Baseline agent comparison
- Detailed breakdown and notes
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from models import (
    ActionType,
    AgentRole,
    BenchmarkResult,
    DataCenterAction,
    GradingResult,
    safe_score,
    TaskDefinition,
    TaskTier,
)
from environment import DataCenterOpsEnv, TASK_CONFIG

if TYPE_CHECKING:
    from environment import DataCenterOpsEnv


# =============================================================================
# Task Definitions
# =============================================================================

TASK_DEFINITIONS: Dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        task_id="easy",
        tier=TaskTier.EASY,
        name="Basic Incident Response",
        description="Single incident, full pipeline execution. Agent must detect, alert, investigate, diagnose, request help, dispatch, and resolve.",
        max_steps=24,
        max_incidents=1,
        cascade_probability=0.0,
        technicians_available=5,
        repair_steps_required=4,
        min_resolution_rate=1.0,
        min_coordination_score=0.5,
        supports_seed_variants=True,
    ),
    "medium": TaskDefinition(
        task_id="medium",
        tier=TaskTier.MEDIUM,
        name="Multi-Incident Coordination",
        description="Multiple incidents with cascade risk. Agent must prioritize and coordinate response.",
        max_steps=42,
        max_incidents=3,
        cascade_probability=0.20,
        technicians_available=4,
        repair_steps_required=6,
        min_resolution_rate=0.7,
        min_coordination_score=0.6,
        supports_seed_variants=True,
        has_red_herrings=False,
    ),
    "hard": TaskDefinition(
        task_id="hard",
        tier=TaskTier.HARD,
        name="Crisis Management",
        description="Multiple concurrent incidents with cascade risk and resource constraints. Agent must prevent cascade while resolving incidents.",
        max_steps=60,
        max_incidents=5,
        cascade_probability=0.35,
        technicians_available=3,
        repair_steps_required=9,
        min_resolution_rate=0.6,
        min_coordination_score=0.7,
        supports_seed_variants=True,
        has_red_herrings=True,
        requires_root_cause=True,
    ),
}


# =============================================================================
# Baseline Agents
# =============================================================================

class RandomAgent:
    """Random action selection baseline."""
    
    name = "random"
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def select_action(self, observation) -> DataCenterAction:
        """Select random valid action."""
        valid_actions = observation.valid_actions
        action_type = self.rng.choice(valid_actions)
        
        return DataCenterAction(
            action_type=action_type,
            confidence=self.rng.random(),
        )


class HeuristicAgent:
    """Rule-based heuristic agent that follows the correct pipeline."""
    
    name = "heuristic"
    
    def __init__(self):
        self.state = "start"
    
    def reset(self):
        self.state = "start"
    
    def select_action(self, observation) -> DataCenterAction:
        """Select action based on pipeline state."""
        agent = observation.current_agent
        
        # Get agent states
        watcher = observation.agent_states.get("watcher")
        responder = observation.agent_states.get("responder")
        coordinator = observation.agent_states.get("coordinator")
        
        # Watcher logic
        if agent == AgentRole.WATCHER:
            if not watcher.alert_sent and observation.active_incidents:
                return DataCenterAction(
                    action_type=ActionType.WATCHER_ALERT,
                    reasoning="Active incident detected, sending alert",
                    confidence=0.9,
                )
            elif watcher.alert_sent and not watcher.investigation_complete:
                return DataCenterAction(
                    action_type=ActionType.WATCHER_INVESTIGATE,
                    reasoning="Investigating reported incident",
                    confidence=0.85,
                )
            else:
                return DataCenterAction(
                    action_type=ActionType.WATCHER_MONITOR,
                    reasoning="Monitoring system status",
                    confidence=0.6,
                )
        
        # Responder logic
        elif agent == AgentRole.RESPONDER:
            if watcher and watcher.investigation_complete and not responder.diagnosis_complete:
                return DataCenterAction(
                    action_type=ActionType.RESPONDER_DIAGNOSE,
                    reasoning="Diagnosing based on investigation results",
                    confidence=0.85,
                )
            elif responder.diagnosis_complete and not responder.help_requested:
                return DataCenterAction(
                    action_type=ActionType.RESPONDER_REQUEST_HELP,
                    reasoning="Requesting coordinator assistance for manual fix",
                    confidence=0.8,
                )
            else:
                return DataCenterAction(
                    action_type=ActionType.RESPONDER_FIX,
                    reasoning="Attempting automated fix",
                    confidence=0.5,
                )
        
        # Coordinator logic
        elif agent == AgentRole.COORDINATOR:
            # Check if can dispatch
            if (watcher and watcher.alert_sent and 
                watcher.investigation_complete and
                responder and responder.diagnosis_complete and
                responder.help_requested and
                not coordinator.dispatch_complete):
                
                # Find available technician
                if observation.technicians_available > 0:
                    return DataCenterAction(
                        action_type=ActionType.COORDINATOR_DISPATCH,
                        reasoning="Dispatching technician to resolve incident",
                        confidence=0.9,
                    )
            
            # Check if can resolve
            for incident in observation.active_incidents:
                if incident.assigned_technician and incident.dispatch_step:
                    steps_since = observation.step_number - incident.dispatch_step
                    repair_steps = TASK_CONFIG[observation.task_tier]["repair_steps"]
                    if steps_since >= repair_steps:
                        return DataCenterAction(
                            action_type=ActionType.COORDINATOR_RESOLVE,
                            reasoning=f"Technician has completed repair work",
                            confidence=0.95,
                        )
            
            # Otherwise coordinate
            return DataCenterAction(
                action_type=ActionType.COORDINATOR_MESSAGE,
                message="Status check: all teams please report",
                reasoning="Coordinating team efforts",
                confidence=0.5,
            )
        
        # Fallback
        return DataCenterAction(
            action_type=observation.valid_actions[0],
            confidence=0.3,
        )


# =============================================================================
# Grader Class
# =============================================================================

class Grader:
    """
    Grading system for evaluating agents.
    
    Provides deterministic scoring across multiple episodes with
    detailed breakdown and comparison.
    """
    
    def __init__(self, n_episodes: int = 5):
        self.n_episodes = n_episodes
        self.results: Dict[str, GradingResult] = {}
    
    def grade_agent(
        self,
        agent,
        task_id: str,
        verbose: bool = False,
    ) -> GradingResult:
        """
        Grade an agent on a specific task.
        
        Args:
            agent: Agent with select_action(observation) method
            task_id: Task identifier (easy, medium, hard)
            verbose: Whether to print progress
        
        Returns:
            GradingResult with score and breakdown
        """
        task = TASK_DEFINITIONS.get(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")
        
        rewards = []
        resolutions = []
        coordination_scores = []
        notes = []
        
        for episode in range(self.n_episodes):
            seed = episode * 1000 + hash(task_id) % 1000
            
            env = DataCenterOpsEnv(task_tier=task.tier, seed=seed)
            obs = env.reset()
            
            if hasattr(agent, 'reset'):
                agent.reset()
            
            episode_reward = 0.0
            done = False
            
            while not done and obs.step_number < task.max_steps:
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                if verbose and obs.step_number % 10 == 0:
                    print(f"  Episode {episode+1}, Step {obs.step_number}: reward={reward:.2f}")
            
            # Record results
            replay = env.get_replay()
            if replay.result:
                rewards.append(episode_reward)
                resolutions.append(replay.result.incidents_resolved / max(1, replay.result.incidents_total))
                coordination_scores.append(replay.result.coordination_score)
            
            notes.append(f"Episode {episode+1}: reward={episode_reward:.2f}, resolved={replay.result.incidents_resolved if replay.result else 0}/{replay.result.incidents_total if replay.result else 0}")
        
        # Compute final score
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        resolution_rate_raw = sum(resolutions) / len(resolutions) if resolutions else 0
        avg_coordination_raw = sum(coordination_scores) / len(coordination_scores) if coordination_scores else 0
        
        # Score formula: weighted combination
        score_raw = (
            0.4 * resolution_rate_raw +
            0.3 * min(1.0, avg_reward / 20.0) +  # Normalize reward
            0.3 * avg_coordination_raw
        )
        
        # Clamp ALL metrics strictly within (0, 1)
        score = safe_score(score_raw)
        resolution_rate = safe_score(resolution_rate_raw)
        coordination_score = safe_score(avg_coordination_raw)
        
        passed = (
            resolution_rate_raw >= task.min_resolution_rate and
            avg_coordination_raw >= task.min_coordination_score
        )
        
        return GradingResult(
            task_id=task_id,
            tier=task.tier,
            score=score,
            passed=passed,
            episodes_run=self.n_episodes,
            avg_reward=round(avg_reward, 3),
            resolution_rate=resolution_rate,
            coordination_score=coordination_score,
            breakdown={
                "avg_reward": round(avg_reward, 3),
                "resolution_rate": resolution_rate,
                "coordination_score": coordination_score,
            },
            notes=notes,
        )
    
    def run_benchmark(
        self,
        agent,
        agent_type: str = "custom",
        agent_config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """
        Run full benchmark across all tasks.
        
        Args:
            agent: Agent with select_action(observation) method
            agent_type: Type identifier for the agent
            agent_config: Configuration dict for the agent
            verbose: Whether to print progress
        
        Returns:
            BenchmarkResult with scores for all tasks
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Benchmark: {agent_type}")
            print(f"{'='*60}\n")
        
        results = {}
        for task_id in ["easy", "medium", "hard"]:
            if verbose:
                print(f"\nGrading task: {task_id}")
            
            result = self.grade_agent(agent, task_id, verbose=verbose)
            results[task_id] = result
            self.results[task_id] = result
            
            if verbose:
                print(f"  Score: {result.score:.3f} ({'PASS' if result.passed else 'FAIL'})")
        
        duration = time.time() - start_time
        
        # Compute overall score
        overall_raw = sum(r.score for r in results.values()) / len(results)
        overall = safe_score(overall_raw)
        
        return BenchmarkResult(
            agent_type=agent_type,
            agent_config=agent_config or {},
            easy=results.get("easy"),
            medium=results.get("medium"),
            hard=results.get("hard"),
            overall_score=overall,
            duration_seconds=round(duration, 2),
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def run_baseline_benchmark(verbose: bool = True) -> BenchmarkResult:
    """Run benchmark with heuristic baseline agent."""
    grader = Grader(n_episodes=3)
    agent = HeuristicAgent()
    return grader.run_benchmark(agent, agent_type="heuristic", verbose=verbose)


def run_random_benchmark(verbose: bool = True) -> BenchmarkResult:
    """Run benchmark with random baseline agent."""
    grader = Grader(n_episodes=3)
    agent = RandomAgent(seed=42)
    return grader.run_benchmark(agent, agent_type="random", verbose=verbose)


def get_tasks() -> List[TaskDefinition]:
    """Get all task definitions."""
    return list(TASK_DEFINITIONS.values())


def get_task(task_id: str) -> Optional[TaskDefinition]:
    """Get specific task definition."""
    return TASK_DEFINITIONS.get(task_id)
