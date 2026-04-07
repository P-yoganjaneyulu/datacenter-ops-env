"""
DataCenterOps — Client Library
==============================

Type-safe client for connecting to DataCenterOps environment servers.

Supports:
- Async and sync operations
- Automatic session management
- OpenEnv-compatible interface
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import httpx

from models import (
    ActionType,
    DataCenterAction,
    DataCenterObservation,
    DataCenterState,
    TaskTier,
)


class DataCenterClient:
    """
    Async client for DataCenterOps environment.
    
    Usage:
        async with DataCenterClient(base_url="http://localhost:7860") as client:
            obs = await client.reset()
            obs, reward, done, _, _ = await client.step(DataCenterAction(...))
    """
    
    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self) -> "DataCenterClient":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
    
    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        
        url = f"{self.base_url}{path}"
        response = await self._client.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    
    async def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_tier: str = "easy",
    ) -> DataCenterObservation:
        """Reset environment."""
        data = {
            "seed": seed,
            "episode_id": episode_id,
            "task_tier": task_tier,
        }
        result = await self._request("POST", "/reset", json=data)
        return DataCenterObservation(**result["observation"])
    
    async def step(
        self,
        action: DataCenterAction,
    ) -> Tuple[DataCenterObservation, float, bool, bool, Dict[str, Any]]:
        """Execute action."""
        data = {
            "action_type": action.action_type.value,
            "incident_id": action.incident_id,
            "technician_id": action.technician_id,
            "message": action.message,
            "confidence": action.confidence,
            "reasoning": action.reasoning,
        }
        result = await self._request("POST", "/step", json=data)
        
        obs = DataCenterObservation(**result["observation"])
        reward = result["reward"]
        terminated = result["terminated"]
        truncated = result["truncated"]
        info = result["info"]
        
        return obs, reward, terminated, truncated, info
    
    async def state(self) -> DataCenterState:
        """Get current state."""
        result = await self._request("GET", "/state")
        return DataCenterState(**result)
    
    async def get_tasks(self) -> List[Dict[str, Any]]:
        """Get available tasks."""
        result = await self._request("GET", "/tasks")
        return result["tasks"]
    
    async def run_baseline(
        self,
        task_tier: str = "easy",
        agent_type: str = "heuristic",
    ) -> Dict[str, Any]:
        """Run baseline agent."""
        params = {"task_tier": task_tier, "agent_type": agent_type}
        return await self._request("GET", "/baseline", params=params)
    
    async def get_replay(self) -> Dict[str, Any]:
        """Get episode replay."""
        return await self._request("GET", "/replay")
    
    async def get_evidence(self) -> Dict[str, Any]:
        """Get gathered evidence."""
        return await self._request("GET", "/evidence")
    
    async def get_unknowns(self) -> Dict[str, Any]:
        """Get unknowns."""
        return await self._request("GET", "/unknowns")
    
    def sync(self) -> "SyncDataCenterClient":
        """Get synchronous wrapper."""
        return SyncDataCenterClient(self)


class SyncDataCenterClient:
    """
    Synchronous wrapper for DataCenterClient.
    
    Usage:
        with DataCenterClient(base_url="...").sync() as client:
            obs = client.reset()
            obs, reward, done, _, _ = client.step(DataCenterAction(...))
    """
    
    def __init__(self, async_client: DataCenterClient):
        self._async = async_client
    
    def __enter__(self) -> "SyncDataCenterClient":
        return self
    
    def __exit__(self, *args):
        pass
    
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_tier: str = "easy",
    ) -> DataCenterObservation:
        """Reset environment."""
        import asyncio
        return asyncio.run(self._async.reset(seed, episode_id, task_tier))
    
    def step(
        self,
        action: DataCenterAction,
    ) -> Tuple[DataCenterObservation, float, bool, bool, Dict[str, Any]]:
        """Execute action."""
        import asyncio
        return asyncio.run(self._async.step(action))
    
    def state(self) -> DataCenterState:
        """Get current state."""
        import asyncio
        return asyncio.run(self._async.state())
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        """Get available tasks."""
        import asyncio
        return asyncio.run(self._async.get_tasks())
    
    def run_baseline(self, task_tier: str = "easy", agent_type: str = "heuristic") -> Dict[str, Any]:
        """Run baseline agent."""
        import asyncio
        return asyncio.run(self._async.run_baseline(task_tier, agent_type))
    
    def get_replay(self) -> Dict[str, Any]:
        """Get episode replay."""
        import asyncio
        return asyncio.run(self._async.get_replay())


# Convenience function
def create_client(base_url: str = "http://localhost:7860") -> DataCenterClient:
    """Create environment client."""
    return DataCenterClient(base_url=base_url)


# Example usage
async def example_usage():
    """Example of using the client."""
    async with DataCenterClient(base_url="http://localhost:7860") as client:
        # Reset
        obs = await client.reset(task_tier="easy")
        print(f"Episode started: {obs.episode_id}")
        print(f"Active incidents: {len(obs.active_incidents)}")
        
        # Run episode
        done = False
        total_reward = 0.0
        
        while not done:
            # Select valid action
            action = DataCenterAction(
                action_type=obs.valid_actions[0],
                confidence=0.8,
                reasoning="Following heuristic policy",
            )
            
            obs, reward, terminated, truncated, info = await client.step(action)
            total_reward += reward
            done = terminated or truncated
            
            print(f"Step {obs.step_number}: {action.action_type.value} -> {reward:.2f}")
        
        print(f"\nEpisode complete!")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Resolved: {len(obs.resolved_incidents)}")
        
        # Get replay
        replay = await client.get_replay()
        print(f"Replay steps: {len(replay.get('replay_steps', []))}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
