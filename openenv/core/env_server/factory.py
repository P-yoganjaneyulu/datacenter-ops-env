"""
OpenEnv Server Factory
======================
Creates a FastAPI application from an OpenEnv environment.

The generated server provides:
- REST endpoints: /health, /schema, /reset, /step, /state
- WebSocket endpoint: /ws for persistent sessions
- OpenAPI documentation: /docs
"""
from typing import Callable, Type, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import uuid

from .types import Action, Observation, State
from .interfaces import Environment


def create_app(
    env: Callable[[], Environment],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    state_cls: Optional[Type[State]] = None,
    env_name: str = "openenv-environment",
) -> FastAPI:
    """
    Create a FastAPI application for an OpenEnv environment.
    
    Args:
        env: Factory function that creates a new environment instance
        action_cls: The Action subclass for this environment
        observation_cls: The Observation subclass for this environment
        state_cls: Optional State subclass (for schema documentation)
        env_name: Human-readable environment name
    
    Returns:
        FastAPI application with OpenEnv endpoints
    """
    app = FastAPI(
        title=f"OpenEnv: {env_name}",
        description="OpenEnv-compliant environment server",
        version="1.0.0",
    )
    
    # Enable CORS for web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store active sessions
    sessions: dict[str, Environment] = {}
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "environment": env_name}
    
    @app.get("/schema")
    async def schema():
        """Return action and observation schemas."""
        return {
            "action": action_cls.model_json_schema(),
            "observation": observation_cls.model_json_schema(),
        }
    
    @app.post("/reset")
    async def reset(seed: Optional[int] = None, task: Optional[str] = None):
        """Reset environment and return initial observation."""
        environment = env()
        episode_id = str(uuid.uuid4())
        sessions[episode_id] = environment
        
        kwargs = {}
        if task:
            kwargs["task"] = task
        
        obs = environment.reset(seed=seed, episode_id=episode_id, **kwargs)
        return {"episode_id": episode_id, "observation": obs.model_dump()}
    
    @app.post("/step")
    async def step(episode_id: str, action: dict):
        """Execute action and return observation."""
        if episode_id not in sessions:
            raise HTTPException(status_code=404, detail="Episode not found")
        
        environment = sessions[episode_id]
        action_obj = action_cls(**action)
        obs = environment.step(action_obj)
        
        return {"observation": obs.model_dump()}
    
    @app.get("/state")
    async def state(episode_id: str):
        """Return full environment state."""
        if episode_id not in sessions:
            raise HTTPException(status_code=404, detail="Episode not found")
        
        environment = sessions[episode_id]
        return environment.state.model_dump()
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for persistent sessions."""
        await websocket.accept()
        
        environment = env()
        episode_id = str(uuid.uuid4())
        
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "reset":
                    seed = message.get("seed")
                    kwargs = message.get("options", {})
                    obs = environment.reset(seed=seed, episode_id=episode_id, **kwargs)
                    await websocket.send_json({
                        "type": "observation",
                        "episode_id": episode_id,
                        "observation": obs.model_dump()
                    })
                
                elif message.get("type") == "step":
                    action = action_cls(**message.get("action", {}))
                    obs = environment.step(action)
                    await websocket.send_json({
                        "type": "observation",
                        "observation": obs.model_dump()
                    })
                
                elif message.get("type") == "state":
                    await websocket.send_json({
                        "type": "state",
                        "state": environment.state.model_dump()
                    })
        
        except WebSocketDisconnect:
            pass
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
    
    return app
