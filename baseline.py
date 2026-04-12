#!/usr/bin/env python3
"""
DataCenterOps Baseline Agents
==============================
Two baseline agents to bracket the learning problem:

  RandomAgent    — mask-aware random. Picks a valid action for the
                   current agent's turn but ignores all state. This is
                   the true lower bound. Scores ~0.10–0.20.

  HeuristicAgent — rule-based pipeline. Knows the correct sequence
                   (alert → investigate → diagnose → fix → request_help
                   → dispatch → resolve) and always picks the right next
                   step for whoever's turn it is. Scores ~0.75–0.85.

The gap between these two is the RL learning opportunity.

Run:
    python baseline.py                   # heuristic on all tasks
    python baseline.py --agent rand      # random agent
    python baseline.py --task hard       # single task
    python baseline.py --render          # step-by-step rendering
    python baseline.py --seed 7          # specific seed
"""

import argparse
import json
import random
import time

from environment import DataCenterOpsEnv
from models import TaskTier, ActionType, AgentRole, DataCenterAction
from grader import Grader, HeuristicAgent as GraderHeuristicAgent, RandomAgent as GraderRandomAgent


# ---------------------------------------------------------------------------
# Random Agent (true lower bound)
# ---------------------------------------------------------------------------

class RandomAgent:
    """
    Mask-aware random agent — the correct lower bound for multi-agent envs.
    Samples uniformly from valid actions for the current agent's turn.
    Never wastes a turn on a cross-agent action, but ignores all state.
    Expected score: ~0.10–0.20 (much lower than heuristic due to
    wrong-order actions: diagnosing before alerting, dispatching before
    diagnosis, resolving before dispatch, etc.)
    """

    name = "RandomAgent"

    def __init__(self, env: DataCenterOpsEnv):
        self.env = env

    def act(self, obs, info: dict) -> DataCenterAction:
        """Select random valid action."""
        valid_actions = obs.valid_actions
        action_type = random.choice(valid_actions)
        return DataCenterAction(
            action_type=action_type,
            confidence=random.random(),
        )


# ---------------------------------------------------------------------------
# Heuristic Agent (strong rule-based baseline)
# ---------------------------------------------------------------------------

class HeuristicAgent:
    """
    Turn-aware rule-based agent. Reads current_agent from observation
    and executes the optimal action for that agent's turn given current state.

    Encodes the correct coordination pipeline:
        Watcher:     alert (first) → investigate → monitor (idle)
        Responder:   diagnose → fix → request_help → idle
        Coordinator: dispatch → escalate (if critical) → resolve → sync

    Expected score: ~0.75–0.85 across all tasks.
    """

    name = "HeuristicAgent"

    def __init__(self, env: DataCenterOpsEnv):
        self.env = env

    def act(self, obs, info: dict) -> DataCenterAction:
        """Select optimal action based on priority and pipeline state."""
        from models import Severity
        
        # Helper: Check if incident has been alerted
        def is_alerted(inc_id):
            return any(m.message_type == "alert" and f"#{inc_id}" in m.content for m in obs.message_history)
        
        # Helper: Check if incident has been investigated
        def is_investigated(inc_id):
            return any(m.message_type == "investigation_complete" and f"#{inc_id}" in m.content for m in obs.message_history)

        # Helper: Check if help has been requested
        def is_help_requested(inc_id):
            return any(m.message_type == "help_request" and f"#{inc_id}" in m.content for m in obs.message_history)
        
        # Sort active incidents by priority (Severity + Age)
        def get_priority(inc):
            sev_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            age = (obs.step_number - inc.step_started) / 20.0
            return sev_map.get(inc.severity.value, 1) + age

        ranked_incidents = sorted(obs.active_incidents, key=get_priority, reverse=True)
        
        # Find the best target for the current agent's stage
        state = self.env.state()
        w = state.agent_states.get("watcher")
        r = state.agent_states.get("responder")
        c = state.agent_states.get("coordinator")
        
        current_agent = obs.current_agent
        
        # ── WATCHER TURN ──────────────────────────────────────────────
        if current_agent == AgentRole.WATCHER:
            # Watcher's job: Alert then Investigate
            for inc in ranked_incidents:
                if not is_alerted(inc.id):
                    return DataCenterAction(
                        action_type=ActionType.WATCHER_ALERT,
                        incident_id=inc.id,
                        reasoning=f"New {inc.severity.value} incident #{inc.id} detected on {inc.equipment_name}. Alerting team.",
                        confidence=0.95,
                    )
            
            for inc in ranked_incidents:
                if is_alerted(inc.id) and not is_investigated(inc.id):
                    return DataCenterAction(
                        action_type=ActionType.WATCHER_INVESTIGATE,
                        incident_id=inc.id,
                        reasoning=f"Investigating alerted incident #{inc.id} on {inc.equipment_name} for root cause.",
                        confidence=0.9,
                    )
                    
            return DataCenterAction(
                action_type=ActionType.WATCHER_MONITOR,
                reasoning="Monitoring system status.",
                confidence=0.6,
            )

        # ── RESPONDER TURN ────────────────────────────────────────────
        if current_agent == AgentRole.RESPONDER:
            # Responder's job: Diagnose then Request Help
            for inc in ranked_incidents:
                if is_investigated(inc.id) and not is_help_requested(inc.id):
                    if not r.diagnosis_complete:
                         return DataCenterAction(
                            action_type=ActionType.RESPONDER_DIAGNOSE,
                            incident_id=inc.id,
                            reasoning=f"Diagnosing incident #{inc.id} on {inc.equipment_name}.",
                            confidence=0.9,
                        )
                    
                    return DataCenterAction(
                        action_type=ActionType.RESPONDER_REQUEST_HELP,
                        incident_id=inc.id,
                        reasoning=f"Requesting technician for incident #{inc.id} after diagnosis.",
                        confidence=0.85,
                    )
            
            return DataCenterAction(
                action_type=ActionType.RESPONDER_FIX,
                reasoning="Standby for investigations.",
                confidence=0.5,
            )

        # ── COORDINATOR TURN ──────────────────────────────────────────
        if current_agent == AgentRole.COORDINATOR:
            # 1. Resolve completed repairs
            for inc in ranked_incidents:
                if inc.assigned_technician and inc.dispatch_step is not None:
                    from environment import TASK_CONFIG
                    steps_since = obs.step_number - inc.dispatch_step
                    repair_steps = TASK_CONFIG[obs.task_tier]["repair_steps"]
                    if steps_since >= repair_steps:
                        return DataCenterAction(
                            action_type=ActionType.COORDINATOR_RESOLVE,
                            incident_id=inc.id,
                            reasoning=f"Repair complete for incident #{inc.id}. Resolving.",
                            confidence=0.98,
                        )

            # 2. Dispatch for help requests
            for inc in ranked_incidents:
                if not inc.assigned_technician and is_help_requested(inc.id):
                    if obs.technicians_available > 0:
                        # Specialist matching
                        best_tech = None
                        available_techs = [t for t in state.technicians if t.available]
                        for tech in available_techs:
                            if inc.incident_type.value in tech.specialization:
                                best_tech = tech
                                break
                        tech_id = best_tech.id if best_tech else available_techs[0].id
                        
                        return DataCenterAction(
                            action_type=ActionType.COORDINATOR_DISPATCH,
                            incident_id=inc.id,
                            technician_id=tech_id,
                            reasoning=f"Dispatching specialist to incident #{inc.id}.",
                            confidence=0.92,
                        )

            # 3. Escalate high severity
            if not c.escalated:
                for inc in ranked_incidents:
                    if inc.severity in [Severity.HIGH, Severity.CRITICAL]:
                        return DataCenterAction(
                            action_type=ActionType.COORDINATOR_ESCALATE,
                            incident_id=inc.id,
                            reasoning=f"Escalating {inc.severity.value} incident #{inc.id}.",
                            confidence=0.8,
                        )

            return DataCenterAction(
                action_type=ActionType.COORDINATOR_MESSAGE,
                message="Team status sync.",
                reasoning="Coordinating response.",
                confidence=0.5,
            )

        # Fallback
        return DataCenterAction(
            action_type=obs.valid_actions[0],
            confidence=0.3,
        )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(agent, task: str, seed: int = 42, render: bool = False) -> dict:
    """Run a single episode with the given agent."""
    task_tier = TaskTier(task)
    env = DataCenterOpsEnv(task_tier=task_tier)
    grader = Grader(n_episodes=1)

    obs = env.reset(seed=seed)
    agent.env = env

    total_reward = 0.0
    final_info   = {}
    steps        = 0

    print(f"\n{'─'*66}")
    print(f"  Agent: {agent.name}  |  Task: {task.upper()}  |  Seed: {seed}")
    print(f"{'─'*66}")

    while True:
        action = agent.act(obs, final_info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        final_info    = info
        steps        += 1
        if terminated or truncated:
            break

    # Get replay for grading
    replay = env.get_replay()
    score = replay.result.score if replay.result else 0.0
    correct = replay.result.solved if replay.result else False

    print(f"\n  ── Episode Complete ──────────────────────────────────────")
    print(f"  Steps          : {steps}")
    print(f"  Episode Reward : {total_reward:.3f}")
    print(f"  Resolved       : {final_info.get('incidents_resolved', 0)}")
    print(f"  Coordination   : {final_info.get('coordination_score', 0):.3f}")
    print(f"  Cascades       : {final_info.get('cascade_count', 0)}")
    print(f"\n  ── Grader Result ──────────────────────────────────────────")
    print(f"  Score          : {score:.3f}  ({'PASS ✅' if correct else 'FAIL ❌'})")

    env.close()
    return {
        "agent":       agent.name,
        "task":        task,
        "seed":        seed,
        "steps":       steps,
        "reward":      round(total_reward, 3),
        "grade_score": score,
        "correct":     correct,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DataCenterOps Baseline Runner")
    parser.add_argument("--agent",  choices=["heuristic", "rand"], default="heuristic")
    parser.add_argument("--task",   choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    results = []

    print("\n" + "═" * 66)
    print("  DataCenterOps — Baseline Evaluation  |  OpenEnv Hackathon 2026")
    print("═" * 66)

    for task in tasks:
        task_tier = TaskTier(task)
        tmp_env = DataCenterOpsEnv(task_tier=task_tier)
        agent   = HeuristicAgent(tmp_env) if args.agent == "heuristic" else RandomAgent(tmp_env)
        r       = run_episode(agent, task=task, seed=args.seed, render=args.render)
        results.append(r)
        time.sleep(0.05)

    print("\n" + "═" * 66)
    print("  SUMMARY")
    print("═" * 66)
    print(f"  {'Task':<12} {'Agent':<18} {'Score':>6}  {'Result'}")
    print(f"  {'─'*12} {'─'*18} {'─'*6}  {'─'*8}")
    for r in results:
        status = "PASS ✅" if r["correct"] else "FAIL ❌"
        print(f"  {r['task']:<12} {r['agent']:<18} {r['grade_score']:>6.3f}  {status}")
    avg = sum(r["grade_score"] for r in results) / len(results)
    print(f"\n  Average Score: {avg:.3f}")
    print("═" * 66 + "\n")


if __name__ == "__main__":
    main()
