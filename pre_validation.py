#!/usr/bin/env python3
"""Pre-validation checks for OpenEnv hackathon submission."""

from __future__ import annotations

import importlib
import json
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "openenv.yaml",
    "inference.py",
    "server/app.py",
    "environment.py",
    "models.py",
    "Dockerfile",
    "server/Dockerfile",
    "pre_validation.py",
]


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def check_files() -> None:
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).exists():
            fail(f"Missing required file: {rel}")
    print("[OK] required files present")


def check_imports() -> None:
    modules = ["server.app", "environment", "models", "inference"]
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as e:
            fail(f"Import failed for {m}: {e}")
    print("[OK] imports")


def check_env_contract() -> None:
    from environment import DataCenterOpsEnv
    from models import TaskTier, DataCenterAction

    env = DataCenterOpsEnv(task_tier=TaskTier.EASY, seed=123)
    obs = env.reset(seed=123)
    if not hasattr(env, "state"):
        fail("Environment missing state()")
    if not obs.valid_actions:
        fail("reset() returned no valid actions")

    action = DataCenterAction(action_type=obs.valid_actions[0])
    step_result = env.step(action)
    if not (isinstance(step_result, tuple) and len(step_result) == 5):
        fail("step() must return (observation, reward, terminated, truncated, info)")
    env.close()
    print("[OK] env contract")


def _wait_for_port(host: str, port: int, timeout_s: float = 15.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def check_api_contract() -> None:
    """Boot server and verify /health /reset /step /state + one full episode."""
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "7860",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not _wait_for_port("127.0.0.1", 7860):
            fail("API server did not start on port 7860")

        base = "http://127.0.0.1:7860"
        health = json.loads(urllib.request.urlopen(f"{base}/health", timeout=5).read())
        if health.get("status") != "healthy":
            fail("/health did not return healthy status")

        reset_url = f"{base}/reset?" + urllib.parse.urlencode({"task": "easy", "seed": 123})
        reset = json.loads(urllib.request.urlopen(reset_url, data=b"", timeout=10).read())
        episode_id = reset.get("episode_id")
        obs = reset.get("observation")
        if not episode_id or not isinstance(obs, dict):
            fail("/reset must return episode_id + observation")

        required_step_keys = {"observation", "reward", "terminated", "truncated", "info"}
        for _ in range(24):
            valid_actions = obs.get("valid_actions") or []
            if not valid_actions:
                fail("Observation missing valid_actions")
            action = valid_actions[0]
            req = urllib.request.Request(
                f"{base}/step?" + urllib.parse.urlencode({"episode_id": episode_id}),
                data=json.dumps({"action_type": action}).encode(),
                headers={"Content-Type": "application/json"},
            )
            step = json.loads(urllib.request.urlopen(req, timeout=10).read())
            if set(step.keys()) != required_step_keys:
                fail(f"/step payload keys mismatch: {sorted(step.keys())}")
            obs = step["observation"]
            if step["terminated"] or step["truncated"]:
                break

        state = json.loads(
            urllib.request.urlopen(
                f"{base}/state?" + urllib.parse.urlencode({"episode_id": episode_id}),
                timeout=10,
            ).read()
        )
        if "step_number" not in state or "incidents" not in state:
            fail("/state missing required fields")

        print("[OK] api contract")
    except Exception as e:
        fail(f"API contract check failed: {e}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    check_files()
    check_imports()
    check_env_contract()
    check_api_contract()
    print("[PASS] pre-validation complete")


if __name__ == "__main__":
    main()
