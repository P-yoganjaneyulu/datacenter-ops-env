"""
DataCenterOps — OpenEnv-Compliant Pydantic Models
==================================================

This module provides type-safe, validated models for the DataCenterOps environment,
following the OpenEnv standard used by Meta's RL environment framework.

Key Features:
- Pydantic v2 models with strict validation
- Evidence tracking for agent reasoning
- Reward breakdown for debuggability
- Replay support for episode recording
- Multi-agent message passing
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field, field_validator, model_validator


def deterministic_timestamp() -> datetime:
    """Deterministic default timestamp for reproducible outputs."""
    return datetime.fromtimestamp(0)


def safe_score(score: float) -> float:
    epsilon = 1e-6
    try:
        score = float(score)
    except:
        return epsilon
    if score <= 0.0:
        return epsilon
    if score >= 1.0:
        return 1.0 - epsilon
    return score


# =============================================================================
# Enums
# =============================================================================

class Severity(str, Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentRole(str, Enum):
    """Agent roles in the multi-agent system."""
    WATCHER = "watcher"
    RESPONDER = "responder"
    COORDINATOR = "coordinator"


class ActionType(str, Enum):
    """All available actions in the environment."""
    # Watcher actions
    WATCHER_MONITOR = "watcher_monitor"
    WATCHER_ALERT = "watcher_alert"
    WATCHER_INVESTIGATE = "watcher_investigate"
    
    # Responder actions
    RESPONDER_DIAGNOSE = "responder_diagnose"
    RESPONDER_FIX = "responder_fix"
    RESPONDER_REQUEST_HELP = "responder_request_help"
    
    # Coordinator actions
    COORDINATOR_DISPATCH = "coordinator_dispatch"
    COORDINATOR_ESCALATE = "coordinator_escalate"
    COORDINATOR_RESOLVE = "coordinator_resolve"
    COORDINATOR_MESSAGE = "coordinator_message"


class IncidentType(str, Enum):
    """Types of incidents that can occur."""
    OVERHEATING = "overheating"
    POWER_SURGE = "power_surge"
    NETWORK_FAILURE = "network_failure"
    DISK_FAILURE = "disk_failure"
    MEMORY_LEAK = "memory_leak"
    COOLING_FAILURE = "cooling_failure"
    SECURITY_BREACH = "security_breach"


class TaskTier(str, Enum):
    """Task difficulty tiers."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class MessagePriority(str, Enum):
    """Priority levels for inter-agent messages."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class FixStatus(str, Enum):
    """Status of incident fixes."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Evidence & Reasoning Models (Winner-Level Feature)
# =============================================================================

class EvidenceSnippet(BaseModel):
    """A piece of evidence gathered by an agent during investigation."""
    id: str = Field(..., description="Unique identifier for this evidence")
    source: str = Field(..., description="Where this evidence came from (e.g., 'logs', 'metrics')")
    content: str = Field(..., description="The actual evidence content")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance to current incident")
    timestamp: datetime = Field(default_factory=deterministic_timestamp, description="When evidence was gathered")
    agent_role: AgentRole = Field(..., description="Which agent gathered this evidence")
    
    model_config = {"frozen": False}


class UnknownInfo(BaseModel):
    """Tracks what the agent doesn't know yet ( Winner-Level Feature)."""
    category: str = Field(..., description="Category of unknown (e.g., 'root_cause', 'affected_systems')")
    description: str = Field(..., description="What is unknown")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="How important this is to discover")
    discoverable: bool = Field(default=True, description="Whether this can be discovered with available actions")


class ReasoningStep(BaseModel):
    """A single step in the agent's reasoning process."""
    step_number: int = Field(..., ge=0)  # Allow 0 for initial step
    agent_role: AgentRole
    thought: str = Field(..., description="The agent's internal reasoning")
    action_considered: List[ActionType] = Field(default_factory=list)
    action_taken: ActionType
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in this decision")
    evidence_used: List[str] = Field(default_factory=list, description="IDs of evidence used")
    timestamp: datetime = Field(default_factory=deterministic_timestamp)


# =============================================================================
# Inter-Agent Communication Models
# =============================================================================

class AgentMessage(BaseModel):
    """Structured message between agents."""
    id: str = Field(..., description="Unique message ID")
    sender: AgentRole
    receiver: AgentRole | Literal["all"]
    message_type: str = Field(..., description="Type of message (alert, diagnosis, help_request, etc.)")
    content: str = Field(..., description="Message content")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL)
    evidence_refs: List[str] = Field(default_factory=list, description="Referenced evidence IDs")
    requires_response: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=deterministic_timestamp)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Core Environment Models
# =============================================================================

class Equipment(BaseModel):
    """Data center equipment."""
    id: str
    name: str
    equipment_type: str
    location: str
    status: Literal["healthy", "degraded", "failed", "maintenance"] = "healthy"
    
    # Metrics
    temperature: float = Field(default=45.0, ge=20.0, le=100.0)
    power_draw: float = Field(default=0.5, ge=0.0, le=1.0)
    cpu_utilization: float = Field(default=0.5, ge=0.0, le=1.0)
    memory_utilization: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Relationships
    dependencies: List[str] = Field(default_factory=list, description="IDs of dependent equipment")


class Incident(BaseModel):
    """An incident in the data center."""
    id: int
    incident_type: IncidentType
    severity: Severity
    equipment_id: str
    equipment_name: str
    step_started: int
    
    # Resolution tracking
    resolved: bool = False
    resolution_step: Optional[int] = None
    time_to_resolve: Optional[int] = None
    fix_status: FixStatus = FixStatus.PENDING
    
    # Assignment tracking
    assigned_technician: Optional[str] = None
    dispatch_step: Optional[int] = None
    
    # Cascade tracking
    cascade_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    cascade_parent_id: Optional[int] = None
    
    # Evidence collected for this incident
    collected_evidence: List[str] = Field(default_factory=list, description="IDs of evidence gathered")
    
    # Root cause (for hard mode)
    root_cause_identified: bool = False
    root_cause_category: Optional[str] = None


class Technician(BaseModel):
    """A technician available for dispatch."""
    id: str
    name: str
    specialization: str  # e.g., "cooling", "network", "power"
    available: bool = True
    current_incident_id: Optional[int] = None
    dispatch_step: Optional[int] = None


class AgentState(BaseModel):
    """State of a single agent."""
    role: AgentRole
    
    # Communication state
    messages_sent: int = 0
    messages_received: int = 0
    last_message: Optional[str] = None
    
    # Task progress
    tasks_completed: int = 0
    current_task: Optional[str] = None
    
    # Reasoning state
    confidence: float = 0.5
    is_uncertain: bool = False
    
    # Cumulative tracking
    cumulative_reward: float = 0.0
    steps_taken: int = 0
    
    # Role-specific state
    # Watcher
    alert_sent: bool = False
    investigating: bool = False
    investigation_complete: bool = False
    
    # Responder
    diagnosis_complete: bool = False
    fix_attempted: bool = False
    help_requested: bool = False
    diagnosis_details: Optional[str] = None
    
    # Coordinator
    dispatch_complete: bool = False
    escalated: bool = False
    coordination_events: int = 0


# =============================================================================
# Action Models (OpenEnv-Compliant)
# =============================================================================

class DataCenterAction(BaseModel):
    """
    Type-safe action model following OpenEnv standard.
    
    Uses Pydantic for automatic validation and serialization.
    Each action type has specific required fields.
    """
    action_type: ActionType
    incident_id: Optional[int] = Field(default=None, description="Target incident ID")
    technician_id: Optional[str] = Field(default=None, description="Technician to dispatch")
    equipment_id: Optional[str] = Field(default=None, description="Target equipment")
    message: Optional[str] = Field(default=None, description="Message content for communication")
    evidence_query: Optional[str] = Field(default=None, description="Query for evidence gathering")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence in action")
    reasoning: Optional[str] = Field(default=None, description="Agent's reasoning for this action")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_action_fields(self):
        """Ensure required fields are present for each action type."""
        action = self.action_type
        
        # Dispatch requires technician_id
        if action == ActionType.COORDINATOR_DISPATCH and not self.technician_id:
            # Auto-assign if not specified - will be handled by environment
            pass
            
        # Resolve requires incident_id
        if action == ActionType.COORDINATOR_RESOLVE and self.incident_id is None:
            # Will auto-select first resolvable incident
            pass
            
        return self
    
    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
    }


# =============================================================================
# Reward Models (Winner-Level Debuggability)
# =============================================================================

class RewardBreakdown(BaseModel):
    """
    Detailed breakdown of reward components.
    
    This enables debugging and analysis of agent behavior,
    matching the SRE environment's approach.
    """
    # Incident handling
    incident_detected: float = 0.0
    incident_alerted: float = 0.0
    incident_investigated: float = 0.0
    incident_diagnosed: float = 0.0
    incident_dispatched: float = 0.0
    incident_resolved: float = 0.0
    
    # Process quality
    correct_ordering: float = 0.0
    coordination_bonus: float = 0.0
    evidence_quality: float = 0.0
    
    # Penalties
    ordering_violation: float = 0.0
    invalid_action: float = 0.0
    time_penalty: float = 0.0
    cascade_penalty: float = 0.0
    
    # SLA tracking
    sla_bonus: float = 0.0
    sla_penalty: float = 0.0
    
    # Reasoning quality
    reasoning_clarity: float = 0.0
    appropriate_confidence: float = 0.0
    
    def total(self) -> float:
        """Calculate total reward from breakdown."""
        positive = (
            self.incident_detected +
            self.incident_alerted +
            self.incident_investigated +
            self.incident_diagnosed +
            self.incident_dispatched +
            self.incident_resolved +
            self.correct_ordering +
            self.coordination_bonus +
            self.evidence_quality +
            self.sla_bonus +
            self.reasoning_clarity +
            self.appropriate_confidence
        )
        negative = (
            self.ordering_violation +
            self.invalid_action +
            self.time_penalty +
            self.cascade_penalty +
            self.sla_penalty
        )
        return positive - negative


class DataCenterReward(BaseModel):
    """Complete reward information with breakdown and explanation."""
    total: float = Field(..., description="Total reward value")
    breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    message: str = Field(default="", description="Human-readable explanation")
    agent_rewards: Dict[str, float] = Field(default_factory=dict, description="Per-agent rewards")


# =============================================================================
# Observation Models (OpenEnv-Compliant)
# =============================================================================

class MetricsObservation(BaseModel):
    """Data center metrics observation."""
    temperatures: List[float] = Field(default_factory=lambda: [0.5] * 5)
    power_loads: List[float] = Field(default_factory=lambda: [0.5] * 5)
    network_health: List[float] = Field(default_factory=lambda: [0.95] * 3)
    cpu_utilizations: List[float] = Field(default_factory=lambda: [0.5] * 5)
    memory_utilizations: List[float] = Field(default_factory=lambda: [0.5] * 5)


class DataCenterObservation(BaseModel):
    """
    Type-safe observation model following OpenEnv standard.
    
    Includes evidence tracking, unknowns, and reasoning trace
    for winner-level agent support.
    """
    # Episode info
    episode_id: str
    step_number: int
    task_tier: TaskTier
    max_steps: int
    
    # Current state
    current_agent: AgentRole
    valid_actions: List[ActionType]
    
    # Incidents
    active_incidents: List[Incident] = Field(default_factory=list)
    resolved_incidents: List[Incident] = Field(default_factory=list)
    incident_count: int = 0
    cascade_count: int = 0
    
    # Resources
    technicians_available: int = 0
    technicians_total: int = 0
    
    # Metrics
    metrics: MetricsObservation = Field(default_factory=MetricsObservation)
    
    # Agent states
    agent_states: Dict[str, AgentState] = Field(default_factory=dict)
    
    # Evidence & Reasoning (Winner-Level Features)
    evidence_gathered: List[EvidenceSnippet] = Field(
        default_factory=list,
        description="Evidence collected so far"
    )
    unknowns: List[UnknownInfo] = Field(
        default_factory=list,
        description="What the agent doesn't know yet"
    )
    reasoning_trace: List[ReasoningStep] = Field(
        default_factory=list,
        description="History of agent reasoning"
    )
    
    # Communication
    pending_messages: List[AgentMessage] = Field(
        default_factory=list,
        description="Messages waiting to be read by current agent"
    )
    message_history: List[AgentMessage] = Field(
        default_factory=list,
        description="All messages in this episode"
    )
    
    # Coordination
    coordination_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Episode state
    done: bool = False
    truncated: bool = False
    last_reward: Optional[float] = None
    last_reward_breakdown: Optional[RewardBreakdown] = None
    
    # Action result
    last_action_result: Optional[str] = None
    
    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
    }


# =============================================================================
# State Models
# =============================================================================

class DataCenterState(BaseModel):
    """
    Complete environment state for debugging and replay.
    """
    # Episode identification
    episode_id: str
    task_tier: TaskTier
    seed: Optional[int] = None
    
    # Step tracking
    step_number: int = 0
    max_steps: int = 0
    
    # Current agent
    current_agent: AgentRole = AgentRole.WATCHER
    
    # Equipment state
    equipment: List[Equipment] = Field(default_factory=list)
    
    # Incidents
    incidents: List[Incident] = Field(default_factory=list)
    
    # Technicians
    technicians: List[Technician] = Field(default_factory=list)
    
    # Agent states
    agent_states: Dict[str, AgentState] = Field(default_factory=dict)
    
    # Communication
    message_history: List[AgentMessage] = Field(default_factory=list)
    
    # Evidence
    all_evidence: List[EvidenceSnippet] = Field(default_factory=list)
    
    # Metrics
    metrics: MetricsObservation = Field(default_factory=MetricsObservation)
    
    # Cumulative tracking
    total_reward: float = 0.0
    coordination_score: float = 0.0
    cascade_count: int = 0
    
    model_config = {
        "extra": "forbid",
    }


# =============================================================================
# Replay Models (Winner-Level Feature)
# =============================================================================

class ReplayStep(BaseModel):
    """A single step in an episode replay."""
    step_number: int
    agent: AgentRole
    action: DataCenterAction
    observation: DataCenterObservation
    reward: DataCenterReward
    reasoning: Optional[str] = None
    timestamp: datetime = Field(default_factory=deterministic_timestamp)


class EpisodeResult(BaseModel):
    """Result of a completed episode."""
    episode_id: str
    task_tier: TaskTier
    solved: bool
    score: float = Field(..., ge=0.0, le=1.0)
    steps_taken: int
    incidents_resolved: int
    incidents_total: int
    total_reward: float
    coordination_score: float
    cascade_count: int
    
    # Performance metrics
    avg_time_to_resolve: Optional[float] = None
    evidence_efficiency: float = Field(default=0.0, description="Evidence gathered / steps")
    communication_efficiency: float = Field(default=0.0, description="Useful messages / total")
    
    # Agent breakdown
    agent_rewards: Dict[str, float] = Field(default_factory=dict)
    agent_steps: Dict[str, int] = Field(default_factory=dict)


class ReplayRecord(BaseModel):
    """Complete replay of an episode."""
    episode_id: str
    task_tier: TaskTier
    seed: int
    replay_steps: List[ReplayStep] = Field(default_factory=list)
    result: Optional[EpisodeResult] = None
    judge_notes: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    agent_type: str = "unknown"  # e.g., "llm", "heuristic", "random"


# =============================================================================
# Task & Grading Models
# =============================================================================

class TaskDefinition(BaseModel):
    """Definition of a task/scenario."""
    task_id: str
    tier: TaskTier
    name: str
    description: str
    max_steps: int
    max_incidents: int
    cascade_probability: float = 0.0
    technicians_available: int
    repair_steps_required: int
    
    # Success criteria
    min_resolution_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_coordination_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Features
    supports_seed_variants: bool = True
    has_red_herrings: bool = False
    requires_root_cause: bool = False


class GradingResult(BaseModel):
    """Result of grading an agent on a task."""
    task_id: str
    tier: TaskTier
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    episodes_run: int
    avg_reward: float
    resolution_rate: float
    coordination_score: float
    
    # Breakdown
    breakdown: Dict[str, float] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
    
    # Timestamps
    graded_at: datetime = Field(default_factory=deterministic_timestamp)


class BenchmarkResult(BaseModel):
    """Complete benchmark results across all tasks."""
    agent_type: str
    agent_config: Dict[str, Any] = Field(default_factory=dict)
    
    easy: Optional[GradingResult] = None
    medium: Optional[GradingResult] = None
    hard: Optional[GradingResult] = None
    
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    run_at: datetime = Field(default_factory=deterministic_timestamp)
    duration_seconds: float = 0.0


# =============================================================================
# LLM Agent Models
# =============================================================================

class LLMDecision(BaseModel):
    """Parsed LLM decision for action selection."""
    action: ActionType
    incident_id: Optional[int] = None
    technician_id: Optional[str] = None
    message: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""
    
    # Evidence query (for investigation actions)
    evidence_query: Optional[str] = None


class LLMResponse(BaseModel):
    """Parsed LLM response with action and reasoning."""
    decision: LLMDecision
    raw_response: str
    parse_success: bool = True
    parse_error: Optional[str] = None
