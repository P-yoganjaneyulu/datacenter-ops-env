"""
DataCenterOps — FastAPI Server
==============================

Production-ready FastAPI server implementing the full OpenEnv specification
with winner-level features.

Endpoints:
- POST /reset        - Reset environment
- POST /step         - Execute action
- GET  /state        - Get current state
- GET  /tasks        - List available tasks
- GET  /grader       - Grade current agent
- POST /grader/run   - Run grading for specific agent
- GET  /baseline     - Run baseline agent
- GET  /replay       - Get episode replay
- GET  /health       - Health check
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models import (
    ActionType,
    AgentRole,
    BenchmarkResult,
    DataCenterAction,
    DataCenterObservation,
    DataCenterState,
    GradingResult,
    ReplayRecord,
    TaskDefinition,
    TaskTier,
)
from environment import DataCenterOpsEnv
from grader import Grader, HeuristicAgent, RandomAgent, TASK_DEFINITIONS, get_tasks, get_task
from multi_agent import MultiAgentTeam, create_team


# =============================================================================
# Global State
# =============================================================================

class EnvManager:
    """Manages environment instances."""
    
    def __init__(self):
        self.env: Optional[DataCenterOpsEnv] = None
        self.current_task: TaskTier = TaskTier.EASY
        self.last_replay: Optional[ReplayRecord] = None
    
    def get_or_create(self, task_tier: TaskTier = TaskTier.EASY, seed: Optional[int] = None) -> DataCenterOpsEnv:
        if self.env is None or self.current_task != task_tier:
            self.env = DataCenterOpsEnv(task_tier=task_tier, seed=seed)
            self.current_task = task_tier
        return self.env
    
    def reset(self, task_tier: Optional[TaskTier] = None) -> DataCenterOpsEnv:
        if task_tier:
            self.current_task = task_tier
        self.env = DataCenterOpsEnv(task_tier=self.current_task)
        return self.env


env_manager = EnvManager()


# =============================================================================
# Request/Response Models
# =============================================================================

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_tier: Optional[str] = "easy"


class StepRequest(BaseModel):
    action_type: str
    incident_id: Optional[int] = None
    technician_id: Optional[str] = None
    message: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class GradeRequest(BaseModel):
    agent_type: str = "heuristic"
    task_tier: str = "easy"
    n_episodes: int = 3


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"[{datetime.now().isoformat()}] DataCenterOps server starting...")
    yield
    # Shutdown
    print(f"[{datetime.now().isoformat()}] DataCenterOps server shutting down...")


# =============================================================================
# App
# =============================================================================

app = FastAPI(
    title="DataCenterOps Environment",
    description="Winner-level multi-agent RL environment for data center operations",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Core Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "DataCenterOps Environment",
        "version": "2.0.0",
        "description": "Winner-level multi-agent RL environment for data center operations",
        "endpoints": {
            "/reset": "Reset environment",
            "/step": "Execute action",
            "/state": "Get current state",
            "/tasks": "List available tasks",
            "/grader": "Get grading results",
            "/grader/run": "Run grading",
            "/baseline": "Run baseline agent",
            "/replay": "Get episode replay",
            "/health": "Health check",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    """Reset environment to initial state."""
    try:
        task_tier = TaskTier(request.task_tier.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid task_tier: {request.task_tier}")
    
    env = env_manager.get_or_create(task_tier)
    observation = env.reset(seed=request.seed, episode_id=request.episode_id)
    
    return JSONResponse(content={
        "observation": observation.model_dump(),
        "info": {
            "episode_id": env.episode_id,
            "task_tier": env.task_tier.value,
            "max_steps": env.max_steps,
        }
    })


@app.post("/step")
async def step(request: StepRequest):
    """Execute action in environment."""
    env = env_manager.env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    # Parse action
    try:
        action_type = ActionType(request.action_type.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid action_type: {request.action_type}")
    
    action = DataCenterAction(
        action_type=action_type,
        incident_id=request.incident_id,
        technician_id=request.technician_id,
        message=request.message,
        confidence=request.confidence,
        reasoning=request.reasoning,
    )
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Store replay if episode ended
    if terminated or truncated:
        env_manager.last_replay = env.get_replay()
    
    return JSONResponse(content={
        "observation": observation.model_dump(),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    })


@app.get("/state")
async def state():
    """Get current environment state."""
    env = env_manager.env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    state = env.state()
    return JSONResponse(content=state.model_dump())


# =============================================================================
# Task Endpoints
# =============================================================================

@app.get("/tasks")
async def tasks():
    """List all available tasks."""
    return JSONResponse(content={
        "tasks": [t.model_dump() for t in get_tasks()],
    })


@app.get("/tasks/{task_id}")
async def get_task_detail(task_id: str):
    """Get details for a specific task."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return JSONResponse(content=task.model_dump())


# =============================================================================
# Grading Endpoints
# =============================================================================

@app.get("/grader")
async def grader_status():
    """Get grading system status and last results."""
    return JSONResponse(content={
        "status": "ready",
        "available_tasks": ["easy", "medium", "hard"],
        "agent_types": ["random", "heuristic", "llm"],
    })


@app.post("/grader/run")
async def grader_run(request: GradeRequest):
    """Run grading for an agent on a task."""
    try:
        task_tier = TaskTier(request.task_tier.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid task_tier: {request.task_tier}")
    
    # Create agent
    if request.agent_type == "random":
        agent = RandomAgent(seed=42)
    elif request.agent_type == "heuristic":
        agent = HeuristicAgent()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported agent type: {request.agent_type}")
    
    # Run grading
    grader = Grader(n_episodes=request.n_episodes)
    result = grader.grade_agent(agent, request.task_tier)
    
    return JSONResponse(content=result.model_dump())


@app.get("/grader/benchmark")
async def run_benchmark(agent_type: str = "heuristic"):
    """Run full benchmark across all tasks."""
    if agent_type == "random":
        agent = RandomAgent(seed=42)
    elif agent_type == "heuristic":
        agent = HeuristicAgent()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported agent type: {agent_type}")
    
    grader = Grader(n_episodes=3)
    result = grader.run_benchmark(agent, agent_type=agent_type)
    
    return JSONResponse(content=result.model_dump())


# =============================================================================
# Baseline Endpoints
# =============================================================================

@app.get("/baseline")
async def run_baseline(
    task_tier: str = "easy",
    agent_type: str = "heuristic",
    verbose: bool = False,
):
    """Run a baseline agent and return results."""
    try:
        tier = TaskTier(task_tier.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid task_tier: {task_tier}")
    
    # Create agent
    if agent_type == "random":
        agent = RandomAgent(seed=42)
    elif agent_type == "heuristic":
        agent = HeuristicAgent()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported agent type: {agent_type}")
    
    # Run episode
    env = DataCenterOpsEnv(task_tier=tier)
    obs = env.reset()
    
    if hasattr(agent, 'reset'):
        agent.reset()
    
    total_reward = 0.0
    actions_taken = []
    
    while not obs.done and obs.step_number < env.max_steps:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        actions_taken.append({
            "step": obs.step_number,
            "action": action.action_type.value,
            "reward": reward,
        })
        
        if verbose and obs.step_number % 10 == 0:
            print(f"Step {obs.step_number}: reward={reward:.2f}")
    
    # Get replay
    replay = env.get_replay()
    env_manager.last_replay = replay
    
    return JSONResponse(content={
        "episode_id": env.episode_id,
        "task_tier": task_tier,
        "agent_type": agent_type,
        "total_reward": round(total_reward, 3),
        "steps": obs.step_number,
        "resolved": len(env.resolved_incidents),
        "total_incidents": len(env.incidents),
        "coordination_score": env.coordination_score,
        "actions": actions_taken[-20:],  # Last 20 actions
        "result": replay.result.model_dump() if replay.result else None,
    })


# =============================================================================
# Replay Endpoints
# =============================================================================

@app.get("/replay")
async def get_replay():
    """Get replay of last completed episode."""
    if env_manager.last_replay is None:
        raise HTTPException(status_code=404, detail="No replay available. Run an episode first.")
    
    return JSONResponse(content=env_manager.last_replay.model_dump())


@app.get("/replay/steps")
async def get_replay_steps(limit: int = Query(default=50, le=200)):
    """Get detailed replay steps."""
    if env_manager.last_replay is None:
        raise HTTPException(status_code=404, detail="No replay available. Run an episode first.")
    
    steps = env_manager.last_replay.replay_steps[:limit]
    
    return JSONResponse(content={
        "episode_id": env_manager.last_replay.episode_id,
        "total_steps": len(env_manager.last_replay.replay_steps),
        "steps": [
            {
                "step": s.step_number,
                "agent": s.agent.value,
                "action": s.action.action_type.value,
                "reward": s.reward.total,
                "reasoning": s.reasoning,
            }
            for s in steps
        ]
    })


# =============================================================================
# Multi-Agent Endpoints
# =============================================================================

@app.post("/multi-agent/run")
async def run_multi_agent(
    task_tier: str = "easy",
    agent_type: str = "heuristic",
):
    """Run multi-agent team."""
    try:
        tier = TaskTier(task_tier.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid task_tier: {task_tier}")
    
    # Create team
    team = create_team(agent_type)
    
    # Run episode
    env = DataCenterOpsEnv(task_tier=tier)
    obs = env.reset()
    team.reset()
    
    total_reward = 0.0
    
    while not obs.done and obs.step_number < env.max_steps:
        action = team.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        team.update_all(obs, action, reward)
        total_reward += reward
    
    # Get metrics
    metrics = team.get_team_metrics()
    replay = env.get_replay()
    
    return JSONResponse(content={
        "episode_id": env.episode_id,
        "task_tier": task_tier,
        "agent_type": agent_type,
        "total_reward": round(total_reward, 3),
        "resolved": len(env.resolved_incidents),
        "team_metrics": metrics,
        "result": replay.result.model_dump() if replay.result else None,
    })


# =============================================================================
# Evidence Endpoints
# =============================================================================

@app.get("/evidence")
async def get_evidence():
    """Get all evidence gathered in current episode."""
    env = env_manager.env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    
    return JSONResponse(content={
        "episode_id": env.episode_id,
        "total_evidence": len(env.evidence_gathered),
        "evidence": [
            {
                "id": e.id,
                "source": e.source,
                "content": e.content,
                "relevance_score": e.relevance_score,
                "agent_role": e.agent_role.value,
            }
            for e in env.evidence_gathered
        ]
    })


@app.get("/unknowns")
async def get_unknowns():
    """Get what the agent doesn't know yet."""
    env = env_manager.env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    
    return JSONResponse(content={
        "episode_id": env.episode_id,
        "unknowns": [
            {
                "category": u.category,
                "description": u.description,
                "importance": u.importance,
                "discoverable": u.discoverable,
            }
            for u in env.unknowns
        ]
    })


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
