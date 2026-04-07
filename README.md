---
title: DataCenterOps Environment
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv-0.2.3
  - openenv
  - rl-environment
  - multi-agent
---

# DataCenterOps — Winner-Level Multi-Agent RL Environment

A production-grade multi-agent reinforcement learning environment for data center operations, designed to match or exceed the quality of OpenEnv hackathon winners.

## 🏆 Winner-Level Features

### 1. Pydantic Models (OpenEnv-Compliant)
- Type-safe `Action`, `Observation`, `State` models
- Automatic validation and serialization
- Self-documenting API

### 2. Rubric-Based Reward System
- Detailed reward breakdown for debugging
- Component-level scoring (detection, investigation, resolution)
- Explanation generation for each reward

### 3. Evidence Tracking
- `evidence_gathered`: What the agent has learned
- `unknowns`: What the agent doesn't know yet
- `reasoning_trace`: History of agent decisions

### 4. Grading System
- `/grader` endpoint for 0-1 scoring
- Multi-episode benchmarking
- Baseline agent comparison

### 5. Replay System
- Full episode recording
- Step-by-step analysis
- Performance metrics

### 6. Real Multi-Agent Architecture
- Independent agents with own state
- Structured message passing
- Emergent coordination

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
├─────────────────────────────────────────────────────────────┤
│  /reset  │ /step │ /state │ /grader │ /tasks │ /replay     │
├──────────┴───────┴────────┴─────────┴────────┴─────────────┤
│                    DataCenterOpsEnv                         │
├───────────────────┬────────────────┬───────────────────────┤
│  Watcher Agent    │ Responder      │ Coordinator           │
│  - Monitor        │ - Diagnose     │ - Dispatch            │
│  - Alert          │ - Fix          │ - Escalate            │
│  - Investigate    │ - Request Help │ - Resolve             │
└───────────────────┴────────────────┴───────────────────────┘
```

## Quick Start

### Docker

```bash
docker build -t datacenter-ops .
docker run -p 7860:7860 datacenter-ops
```

### Python Client

```python
from client import DataCenterClient
from models import DataCenterAction, ActionType

async with DataCenterClient("http://localhost:7860") as env:
    # Reset
    obs = await env.reset(task_tier="easy")
    
    # Run episode
    while not obs.done:
        action = DataCenterAction(
            action_type=obs.valid_actions[0],
            reasoning="Agent decision",
            confidence=0.8,
        )
        obs, reward, terminated, truncated, info = await env.step(action)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment |
| `/step` | POST | Execute action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List available tasks |
| `/grader` | GET | Get grading status |
| `/grader/run` | POST | Run grading |
| `/baseline` | GET | Run baseline agent |
| `/replay` | GET | Get episode replay |
| `/evidence` | GET | Get gathered evidence |
| `/unknowns` | GET | Get unknowns list |

## Tasks

### Easy (24 steps)
- Single incident
- Full pipeline execution
- High resource availability

### Medium (42 steps)
- Multiple incidents
- Cascade risk (20%)
- Limited resources

### Hard (60 steps)
- Concurrent incidents
- High cascade risk (35%)
- Resource constraints
- Root cause analysis required

## Action Space

| Agent | Actions |
|-------|---------|
| Watcher | `watcher_monitor`, `watcher_alert`, `watcher_investigate` |
| Responder | `responder_diagnose`, `responder_fix`, `responder_request_help` |
| Coordinator | `coordinator_dispatch`, `coordinator_escalate`, `coordinator_resolve`, `coordinator_message` |

## Reward Structure

```python
class RewardBreakdown:
    # Incident handling
    incident_detected: float    # +2.0 for timely detection
    incident_alerted: float     # +2.0 for proper alert
    incident_investigated: float # +1.5 for investigation
    incident_diagnosed: float   # +1.5 for diagnosis
    incident_dispatched: float  # +2.0 for dispatch
    incident_resolved: float    # +12-20 for resolution
    
    # Quality metrics
    correct_ordering: float     # +0.3 per correct step
    coordination_bonus: float   # Variable
    evidence_quality: float     # Based on relevance
    
    # Penalties
    ordering_violation: float   # -0.5 per skipped step
    invalid_action: float       # -1.0 per invalid action
    sla_penalty: float          # Based on incident age
```

## Grading

```bash
# Run grading
curl -X POST http://localhost:7860/grader/run \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "heuristic", "task_tier": "easy", "n_episodes": 3}'
```

## Baseline Agents

### Heuristic Agent
- Follows correct pipeline order
- Achieves ~80% resolution on easy tasks
- Good coordination score

### Random Agent
- Random valid action selection
- ~10% resolution rate
- Demonstrates learning gap

## Project Structure

```
datacenter-ops-env/
├── models.py           # Pydantic models (Action, Observation, State)
├── environment.py      # Core environment implementation
├── rubrics.py          # Reward computation system
├── grader.py           # Grading and benchmarking
├── llm_agent.py        # LLM agent integration
├── multi_agent.py      # Multi-agent architecture
├── app.py              # FastAPI server
├── client.py           # Python client library
├── tests/              # Comprehensive test suite
├── Dockerfile          # Container definition
└── requirements.txt    # Dependencies
```

## Comparison with Winners

| Feature | This Project | Winners |
|---------|--------------|---------|
| Pydantic Models | ✅ | ✅ |
| Rubric System | ✅ | ✅ |
| Evidence Tracking | ✅ | ✅ |
| Grading Endpoint | ✅ | ✅ |
| Replay System | ✅ | ✅ |
| LLM Integration | ✅ | ✅ |
| Multi-Agent | ✅ | ✅ |

## License

MIT
