"""
OpenEnv Environment Interface
=============================
Abstract base class for OpenEnv-compliant environments.

Every OpenEnv environment must implement this interface to work with:
- FastAPI server (create_app)
- WebSocket sessions
- OpenEnv client libraries
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional

from .types import Action, Observation, State


ActionType = TypeVar("ActionType", bound=Action)
ObservationType = TypeVar("ObservationType", bound=Observation)  
StateType = TypeVar("StateType", bound=State)


class Environment(ABC, Generic[ActionType, ObservationType, StateType]):
    """
    Abstract base class for OpenEnv environments.
    
    Type Parameters:
        ActionType: The Action subclass this environment accepts
        ObservationType: The Observation subclass this environment returns
        StateType: The State subclass for full state access
    
    Required Methods:
        reset(): Initialize a new episode
        step(action): Take an action, return observation
        state: Property returning current full state
    
    Class Attributes:
        SUPPORTS_CONCURRENT_SESSIONS: Set True if environment supports
            multiple concurrent WebSocket sessions (default: False)
    """
    
    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    
    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs
    ) -> ObservationType:
        """
        Reset the environment and return initial observation.
        
        Args:
            seed: Optional random seed for reproducibility
            episode_id: Optional episode identifier (auto-generated if None)
            **kwargs: Environment-specific options (e.g., task difficulty)
        
        Returns:
            Initial observation
        """
        pass
    
    @abstractmethod
    def step(self, action: ActionType) -> ObservationType:
        """
        Execute an action and return the resulting observation.
        
        Args:
            action: The action to execute
        
        Returns:
            Observation with reward, done flag, and environment-specific data
        """
        pass
    
    @property
    @abstractmethod
    def state(self) -> StateType:
        """
        Return the full readable environment state.
        
        This is a superset of the observation, providing complete
        visibility into environment internals for debugging and oversight.
        
        Returns:
            Full environment state
        """
        pass
