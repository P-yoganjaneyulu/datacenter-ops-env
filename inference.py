#!/usr/bin/env python3
"""
DataCenterOps — Inference Script
=================================
OpenAI Client-based inference script for OpenEnv Hackathon evaluation.
Uses LLM to control agents in the multi-agent data center environment.

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL - The API endpoint for the LLM (e.g., https://router.huggingface.co/v1)
    MODEL_NAME   - The model identifier to use for inference
    HF_TOKEN     - Your Hugging Face API key (or API_KEY)
    LOCAL_IMAGE_NAME - (Optional) Local Docker image name

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
    export HF_TOKEN="your_token_here"
    python inference.py

STDOUT FORMAT (Required by Hackathon):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import re
import sys
from typing import Optional, Dict, List

from openai import OpenAI

# Environment imports
from environment import DataCenterOpsEnv
from models import TaskTier, ActionType, AgentRole, DataCenterAction
from grader import Grader

# ---------------------------------------------------------------------------
# Configuration (from environment variables as per hackathon requirements)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")

# Task configuration
MAX_STEPS = {"easy": 24, "medium": 42, "hard": 60}
TEMPERATURE = 0.2
MAX_TOKENS = 150
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI operations manager controlling a team of three agents in a data center incident response environment.

AGENTS (take turns in round-robin):
1. WATCHER (actions 0-2): Monitors sensors, detects anomalies, broadcasts alerts
   - 0: watcher_monitor (passive scan)
   - 1: watcher_alert (broadcast anomaly)
   - 2: watcher_investigate (deep analysis)

2. RESPONDER (actions 3-5): Diagnoses root causes, attempts fixes, requests help
   - 3: responder_diagnose (root cause analysis)
   - 4: responder_fix (automated remediation)
   - 5: responder_request_help (ask for technician)

3. COORDINATOR (actions 6-9): Dispatches technicians, resolves incidents
   - 6: coordinator_dispatch (send technician)
   - 7: coordinator_escalate (notify management)
   - 8: coordinator_resolve (close incident)
   - 9: coordinate_message (sync / disagreement resolution)

CORRECT PIPELINE:
watcher_alert → watcher_investigate → responder_diagnose → responder_request_help → coordinator_dispatch → [wait for repair] → coordinator_resolve

RULES:
- You can only use actions belonging to the CURRENT AGENT (shown in observation)
- Follow the pipeline order - skipping steps causes penalties
- Resolve incidents before running out of steps
- Prevent cascade failures (medium/hard tasks)

Respond with ONLY the action number (0-9). No explanation needed."""


class LLMAgent:
    """LLM-based agent for DataCenterOps environment."""

    def __init__(self):
        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
        self.conversation_history: List[Dict] = []

    def _build_observation_prompt(self, obs, state, info: dict) -> str:
        """Build a human-readable observation string for the LLM."""
        current_agent = obs.current_agent.value
        valid_actions = [a.value for a in obs.valid_actions]

        lines = [
            f"=== CURRENT STATE ===",
            f"Step: {info.get('step', 0)}",
            f"Current Agent: {current_agent.upper()}",
            f"Valid Actions: {[a.value for a in obs.valid_actions]}",
            f"Active Incidents: {info.get('incidents_active', 0)}",
            f"Resolved: {info.get('incidents_resolved', 0)}",
            f"Coordination Score: {info.get('coordination_score', 0):.2f}",
            f"Cascades: {info.get('cascade_count', 0)}",
        ]

        # Add agent states
        w = state.agent_states.get("watcher")
        r = state.agent_states.get("responder")
        c = state.agent_states.get("coordinator")

        lines.append(f"\n--- Agent States ---")
        lines.append(f"Watcher: alert={'✓' if w and w.alert_sent else '✗'} investigate={'✓' if w and w.investigation_complete else '✗'}")
        lines.append(f"Responder: diagnose={'✓' if r and r.diagnosis_complete else '✗'} help={'✓' if r and r.help_requested else '✗'}")
        lines.append(f"Coordinator: dispatch={'✓' if c and c.dispatch_complete else '✗'}")

        # Add incident details
        if state.incidents:
            lines.append(f"\n--- Incidents ---")
            for inc in state.incidents[:3]:
                status = "RESOLVED" if inc.resolved else "ACTIVE"
                lines.append(f"#{inc.id} {inc.incident_type.value} [{inc.severity.value}] {status}")

        lines.append(f"\nChoose action for {current_agent} (valid: {valid_actions}):")

        return "\n".join(lines)

    def act(self, obs, state, info: dict) -> int:
        """Get action from LLM based on current observation."""
        # Build the user message
        user_message = self._build_observation_prompt(obs, state, info)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            content = response.choices[0].message.content.strip()

            if DEBUG:
                print(f"[LLM Response] {content}", file=sys.stderr)

            # Extract action number from response
            match = re.search(r'\b(\d)\b', content)
            if match:
                action = int(match.group(1))
            else:
                numbers = re.findall(r'\d+', content)
                if numbers:
                    action = int(numbers[0]) % 10
                else:
                    valid = [a.value for a in obs.valid_actions]
                    action = valid[0] if valid else 0

            # Ensure action is valid for current agent
            valid_actions = [a.value for a in obs.valid_actions]
            if action not in valid_actions:
                if DEBUG:
                    print(f"[WARN] LLM chose invalid action {action}, using {valid_actions[0]}", file=sys.stderr)
                action = valid_actions[0] if valid_actions else 0

            return action

        except Exception as e:
            if DEBUG:
                print(f"[ERROR] LLM call failed: {e}", file=sys.stderr)
            valid = [a.value for a in obs.valid_actions]
            return valid[0] if valid else 0


# ---------------------------------------------------------------------------
# Episode Runner with Required STDOUT Format
# ---------------------------------------------------------------------------

def run_episode(task: str, seed: int = 42) -> dict:
    """
    Run a single episode with LLM agent.
    Outputs in the required hackathon format.
    """
    task_tier = TaskTier(task)
    env = DataCenterOpsEnv(task_tier=task_tier, seed=seed)
    obs = env.reset(seed=seed)

    agent = LLMAgent()
    max_steps = MAX_STEPS[task]

    # Output [START] line
    print(f"[START] task={task} env=datacenter-ops-env model={MODEL_NAME}")

    total_reward = 0.0
    step_count = 0
    rewards_list = []
    info = {}
    last_error = "null"
    success = False
    final_score = 0.0

    while step_count < max_steps:
        state = env.state()

        try:
            action_int = agent.act(obs, state, info)
            action = DataCenterAction(action_type=ActionType(action_int))
        except Exception as e:
            valid = obs.valid_actions
            action = DataCenterAction(action_type=valid[0] if valid else ActionType.WATCHER_MONITOR)
            last_error = str(e)

        try:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            rewards_list.append(reward)
            step_count += 1
            
            # Output [STEP] line
            done = terminated or truncated
            print(f"[STEP] step={step_count} action={action.action_type.value} reward={reward:.2f} done={'true' if done else 'false'} error={last_error}")
            
            last_error = "null"  # Reset error after successful step
            
            if terminated or truncated:
                success = terminated  # terminated = resolved, truncated = timeout
                break
                
        except Exception as e:
            last_error = str(e)
            print(f"[STEP] step={step_count + 1} action={action.action_type.value} reward=0.00 done=false error={last_error}")
            step_count += 1

    # Get final score
    replay = env.get_replay()
    if replay.result:
        final_score = replay.result.score
        success = replay.result.solved
    
    # Format rewards list
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
    
    # Output [END] line
    print(f"[END] success={'true' if success else 'false'} steps={step_count} score={final_score:.2f} rewards={rewards_str}")
    
    env.close()

    return {
        "task": task,
        "seed": seed,
        "steps": step_count,
        "total_reward": round(total_reward, 3),
        "score": final_score,
        "success": success,
        "resolved": info.get("incidents_resolved", 0),
        "cascades": info.get("cascade_count", 0),
        "coordination": round(info.get("coordination_score", 0), 3),
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    """Run inference on all tasks and output results."""
    # Print config to stderr (not stdout, which is reserved for structured output)
    print(f"API Base: {API_BASE_URL}", file=sys.stderr)
    print(f"Model: {MODEL_NAME}", file=sys.stderr)
    print(f"Debug: {DEBUG}", file=sys.stderr)

    results = []
    seeds = [42, 123, 456]  # Multiple seeds for robustness

    for task in ["easy", "medium", "hard"]:
        print(f"\n--- Evaluating {task.upper()} task ---", file=sys.stderr)

        task_results = []
        for seed in seeds:
            try:
                result = run_episode(task, seed)
                task_results.append(result)
                print(f"  Seed {seed}: score={result['score']:.3f}, "
                      f"resolved={result['resolved']}, reward={result['total_reward']:.1f}", file=sys.stderr)
            except Exception as e:
                print(f"  Seed {seed}: FAILED - {e}", file=sys.stderr)
                task_results.append({
                    "task": task,
                    "seed": seed,
                    "score": 0.0,
                    "success": False,
                    "error": str(e),
                })

        # Average scores for this task
        scores = [r.get("score", 0) for r in task_results]
        avg_score = sum(scores) / len(scores) if scores else 0
        pass_rate = sum(1 for r in task_results if r.get("success", False)) / len(task_results)

        results.append({
            "task": task,
            "mean_score": round(avg_score, 3),
            "pass_rate": round(pass_rate, 3),
            "runs": task_results,
        })

        print(f"  → Average: {avg_score:.3f} | Pass rate: {pass_rate:.1%}", file=sys.stderr)

    # Summary to stderr
    print("\n" + "=" * 60, file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"{'Task':<10} {'Mean Score':>12} {'Pass Rate':>12}", file=sys.stderr)
    print("-" * 34, file=sys.stderr)
    for r in results:
        print(f"{r['task']:<10} {r['mean_score']:>12.3f} {r['pass_rate']:>11.0%}", file=sys.stderr)

    overall_mean = sum(r["mean_score"] for r in results) / len(results)
    print("-" * 34, file=sys.stderr)
    print(f"{'OVERALL':<10} {overall_mean:>12.3f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    return results


if __name__ == "__main__":
    # Validate required environment variables
    missing = []
    for var in ["API_BASE_URL", "MODEL_NAME"]:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        print(f"ERROR: Missing required environment variables: {missing}", file=sys.stderr)
        print("Set them before running:", file=sys.stderr)
        print("  export API_BASE_URL='https://router.huggingface.co/v1'", file=sys.stderr)
        print("  export MODEL_NAME='meta-llama/Llama-3.3-70B-Instruct'", file=sys.stderr)
        print("  export HF_TOKEN='your_token_here'", file=sys.stderr)
        sys.exit(1)

    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY must be set", file=sys.stderr)
        sys.exit(1)

    main()
