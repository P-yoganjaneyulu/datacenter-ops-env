"""
OpenEnv Core Types
==================
Pydantic base classes for Action, Observation, and State.
These define the contract between agents and environments.
"""
from typing import Optional, Any
from pydantic import BaseModel, Field


class Action(BaseModel):
    """
    Base class for environment actions.
    
    Environments define their own Action subclass with specific fields.
    Actions are sent by agents to interact with the environment.
    """
    pass


class Observation(BaseModel):
    """
    Base class for environment observations.
    
    Returned by reset() and step(). Contains:
    - reward: scalar reward from last transition
    - done: episode termination flag
    - Environment-specific observation fields
    """
    reward: float = Field(default=0.0, description="Reward from last step")
    done: bool = Field(default=False, description="Episode termination flag")
    
    class Config:
        extra = "allow"


class State(BaseModel):
    """
    Base class for full environment state.
    
    Returned by the state() endpoint. Provides complete readable state
    for debugging, visualization, and human oversight.
    """
    episode_id: str = Field(default="", description="Unique episode identifier")
    step_count: int = Field(default=0, description="Number of steps taken")
    
    class Config:
        extra = "allow"
