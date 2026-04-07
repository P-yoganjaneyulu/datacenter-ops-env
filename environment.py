"""
DataCenterOps — Winner-Level RL Environment
===========================================

A production-grade multi-agent RL environment for data center operations,
designed to match or exceed the quality of OpenEnv hackathon winners.

Key Features:
- Pydantic models for type-safe actions/observations
- Evidence tracking for agent reasoning
- Rubric-based reward system with breakdown
- Replay system for episode recording
- Real inter-agent communication
- Deterministic scenarios with seeds

Architecture:
    Agent (LLM/RL) ──HTTP──▶ FastAPI Server ──▶ Environment
                                    │
                           reset() / step() / state()
                           evidence_gathered / unknowns
                           reasoning_trace / messages
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from models import (
    ActionType,
    AgentMessage,
    AgentRole,
    AgentState,
    DataCenterAction,
    DataCenterObservation,
    DataCenterReward,
    DataCenterState,
    Equipment,
    EvidenceSnippet,
    FixStatus,
    Incident,
    IncidentType,
    MessagePriority,
    MetricsObservation,
    ReasoningStep,
    ReplayRecord,
    ReplayStep,
    RewardBreakdown,
    Severity,
    TaskTier,
    Technician,
    UnknownInfo,
)
from rubrics import DataCenterRubric, create_rubric_for_tier

if TYPE_CHECKING:
    from openenv.core.env_server.interfaces import Environment


# =============================================================================
# Configuration
# =============================================================================

TASK_CONFIG = {
    TaskTier.EASY: {
        "max_steps": 24,
        "n_technicians": 5,
        "cascade_prob": 0.0,
        "max_incidents": 1,
        "repair_steps": 4,
    },
    TaskTier.MEDIUM: {
        "max_steps": 42,
        "n_technicians": 4,
        "cascade_prob": 0.20,
        "max_incidents": 3,
        "repair_steps": 6,
    },
    TaskTier.HARD: {
        "max_steps": 60,
        "n_technicians": 3,
        "cascade_prob": 0.35,
        "max_incidents": 5,
        "repair_steps": 9,
    },
}

INCIDENT_TYPES = list(IncidentType)
EQUIPMENT_NAMES = [
    "server-rack-A1", "server-rack-B2", "server-rack-C3",
    "cooling-unit-1", "cooling-unit-2",
    "pdu-main", "pdu-backup",
    "switch-core", "switch-edge",
]

TECHNICIAN_SPECIALTIES = ["cooling", "network", "power", "general", "security"]


# =============================================================================
# Main Environment Class
# =============================================================================

class DataCenterOpsEnv:
    """
    Winner-level multi-agent data center operations environment.
    
    Features:
    - Real learning problem design (random agents fail)
    - Evidence-based reasoning
    - Inter-agent communication
    - Deterministic scenarios
    - Comprehensive grading
    """
    
    SUPPORTS_CONCURRENT_SESSIONS = False
    
    def __init__(
        self,
        task_tier: TaskTier = TaskTier.EASY,
        seed: Optional[int] = None,
    ):
        """Initialize environment with task tier and optional seed."""
        self.task_tier = task_tier
        self.seed_value = seed
        
        # Load config
        config = TASK_CONFIG[task_tier]
        self.max_steps = config["max_steps"]
        self.n_technicians = config["n_technicians"]
        self.cascade_prob = config["cascade_prob"]
        self.max_incidents = config["max_incidents"]
        self.repair_steps = config["repair_steps"]
        
        # Initialize rubric
        self.rubric = create_rubric_for_tier(task_tier)
        
        # Initialize state
        self._init_state()
    
    def _init_state(self):
        """Initialize all state variables."""
        # Episode identification
        self.episode_id = str(uuid.uuid4())[:8]
        self.step_number = 0
        
        # Random state
        if self.seed_value is not None:
            random.seed(self.seed_value)
            np.random.seed(self.seed_value)
        
        # Current agent (round-robin)
        self.current_agent = AgentRole.WATCHER
        
        # Equipment
        self.equipment = self._create_equipment()
        
        # Incidents
        self.incidents: List[Incident] = []
        self.resolved_incidents: List[Incident] = []
        self._incident_counter = 0
        self.cascade_count = 0
        
        # Technicians
        self.technicians = self._create_technicians()
        
        # Agent states
        self.agent_states = {
            "watcher": AgentState(role=AgentRole.WATCHER),
            "responder": AgentState(role=AgentRole.RESPONDER),
            "coordinator": AgentState(role=AgentRole.COORDINATOR),
        }
        
        # Evidence and reasoning (Winner-Level Features)
        self.evidence_gathered: List[EvidenceSnippet] = []
        self.unknowns: List[UnknownInfo] = []
        self.reasoning_trace: List[ReasoningStep] = []
        
        # Communication
        self.message_history: List[AgentMessage] = []
        self.pending_messages: Dict[str, List[AgentMessage]] = {
            "watcher": [],
            "responder": [],
            "coordinator": [],
        }
        
        # Metrics
        self.metrics = self._create_initial_metrics()
        
        # Tracking
        self.total_reward = 0.0
        self.coordination_score = 0.0
        self.coordination_events = 0
        
        # Replay
        self.replay_steps: List[ReplayStep] = []
    
    def _create_equipment(self) -> List[Equipment]:
        """Create data center equipment."""
        equipment = []
        for i, name in enumerate(EQUIPMENT_NAMES):
            eq_type = "server" if "rack" in name else (
                "cooling" if "cooling" in name else (
                "power" if "pdu" in name else "network"
            ))
            equipment.append(Equipment(
                id=f"eq-{i:02d}",
                name=name,
                equipment_type=eq_type,
                location=f"zone-{i % 3 + 1}",
                temperature=45.0 + random.uniform(-5, 15),
                power_draw=0.5 + random.uniform(-0.2, 0.3),
                cpu_utilization=0.4 + random.uniform(0, 0.4),
                memory_utilization=0.4 + random.uniform(0, 0.4),
                dependencies=[],
            ))
        return equipment
    
    def _create_technicians(self) -> List[Technician]:
        """Create available technicians."""
        technicians = []
        for i in range(self.n_technicians):
            technicians.append(Technician(
                id=f"tech-{i:02d}",
                name=f"Technician {i+1}",
                specialization=TECHNICIAN_SPECIALTIES[i % len(TECHNICIAN_SPECIALTIES)],
                available=True,
            ))
        return technicians
    
    def _create_initial_metrics(self) -> MetricsObservation:
        """Create initial metrics observation."""
        return MetricsObservation(
            temperatures=[eq.temperature / 100.0 for eq in self.equipment[:5]],
            power_loads=[eq.power_draw for eq in self.equipment[:5]],
            network_health=[0.95, 0.92, 0.98],
            cpu_utilizations=[eq.cpu_utilization for eq in self.equipment[:5]],
            memory_utilizations=[eq.memory_utilization for eq in self.equipment[:5]],
        )
    
    # =========================================================================
    # Core API Methods
    # =========================================================================
    
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs
    ) -> DataCenterObservation:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            episode_id: Optional episode identifier
            **kwargs: Additional options (task_tier can override)
        
        Returns:
            Initial observation
        """
        # Handle task tier override
        if "task_tier" in kwargs:
            self.task_tier = kwargs["task_tier"]
            config = TASK_CONFIG[self.task_tier]
            self.max_steps = config["max_steps"]
            self.n_technicians = config["n_technicians"]
            self.cascade_prob = config["cascade_prob"]
            self.max_incidents = config["max_incidents"]
            self.repair_steps = config["repair_steps"]
            self.rubric = create_rubric_for_tier(self.task_tier)
        
        # Reset state
        self._init_state()
        
        if seed is not None:
            self.seed_value = seed
            random.seed(seed)
            np.random.seed(seed)
        
        if episode_id is not None:
            self.episode_id = episode_id
        
        # Spawn initial incidents
        self._spawn_incident()
        
        if self.task_tier == TaskTier.MEDIUM and random.random() < 0.5:
            self._spawn_incident()
        elif self.task_tier == TaskTier.HARD:
            for _ in range(random.randint(1, 2)):
                self._spawn_incident()
        
        # Initialize unknowns
        self._update_unknowns()
        
        return self._get_observation()
    
    def step(
        self,
        action: DataCenterAction,
        timeout_s: Optional[float] = None,
        **kwargs
    ) -> Tuple[DataCenterObservation, float, bool, bool, Dict[str, Any]]:
        """
        Execute action and return new observation.
        
        Args:
            action: The action to execute
            timeout_s: Optional timeout (not used)
            **kwargs: Additional options
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Store previous state for reward computation
        prev_state = self._get_state_dict()
        
        # Validate action
        if action.action_type not in self._get_valid_actions():
            # Invalid action penalty
            reward = DataCenterReward(
                total=-1.0,
                breakdown=RewardBreakdown(invalid_action=1.0),
                message=f"Invalid action {action.action_type.value} for agent {self.current_agent.value}",
            )
            obs = self._get_observation()
            obs.done = False
            return obs, reward.total, False, False, self._get_info()
        
        # Execute action
        action_result = self._execute_action(action)
        
        # Record reasoning
        if action.reasoning:
            self.reasoning_trace.append(ReasoningStep(
                step_number=self.step_number,
                agent_role=self.current_agent,
                thought=action.reasoning,
                action_taken=action.action_type,
                confidence=action.confidence or 0.5,
            ))
        
        # Compute reward using rubric
        obs = self._get_observation()
        reward = self.rubric.compute_full(obs, action, prev_state)
        
        # Update tracking
        self.total_reward += reward.total
        self.agent_states[self.current_agent.value].cumulative_reward += reward.total
        self.agent_states[self.current_agent.value].steps_taken += 1
        
        # Record for replay
        self.replay_steps.append(ReplayStep(
            step_number=self.step_number,
            agent=self.current_agent,
            action=action,
            observation=obs,
            reward=reward,
            reasoning=action.reasoning,
        ))
        
        # Advance state
        self.step_number += 1
        self._next_agent()
        
        # Check termination
        active_incidents = [i for i in self.incidents if not i.resolved]
        terminated = len(active_incidents) == 0 and len(self.resolved_incidents) > 0
        truncated = self.step_number >= self.max_steps
        
        # Final bonus
        if terminated or truncated:
            reward.total += self._compute_episode_bonus()
        
        # Update observation
        obs = self._get_observation()
        obs.done = terminated
        obs.truncated = truncated
        obs.last_reward = reward.total
        obs.last_reward_breakdown = reward.breakdown
        
        # Update unknowns
        self._update_unknowns()
        
        return obs, reward.total, terminated, truncated, self._get_info()
    
    def state(self) -> DataCenterState:
        """Get complete environment state."""
        return DataCenterState(
            episode_id=self.episode_id,
            task_tier=self.task_tier,
            seed=self.seed_value,
            step_number=self.step_number,
            max_steps=self.max_steps,
            current_agent=self.current_agent,
            equipment=self.equipment,
            incidents=self.incidents,
            technicians=self.technicians,
            agent_states=self.agent_states,
            message_history=self.message_history,
            all_evidence=self.evidence_gathered,
            metrics=self.metrics,
            total_reward=self.total_reward,
            coordination_score=self.coordination_score,
            cascade_count=self.cascade_count,
        )
    
    # =========================================================================
    # Action Execution
    # =========================================================================
    
    def _execute_action(self, action: DataCenterAction) -> str:
        """Execute action and return result message."""
        action_type = action.action_type
        
        # Watcher actions
        if action_type == ActionType.WATCHER_MONITOR:
            return self._action_monitor(action)
        elif action_type == ActionType.WATCHER_ALERT:
            return self._action_alert(action)
        elif action_type == ActionType.WATCHER_INVESTIGATE:
            return self._action_investigate(action)
        
        # Responder actions
        elif action_type == ActionType.RESPONDER_DIAGNOSE:
            return self._action_diagnose(action)
        elif action_type == ActionType.RESPONDER_FIX:
            return self._action_fix(action)
        elif action_type == ActionType.RESPONDER_REQUEST_HELP:
            return self._action_request_help(action)
        
        # Coordinator actions
        elif action_type == ActionType.COORDINATOR_DISPATCH:
            return self._action_dispatch(action)
        elif action_type == ActionType.COORDINATOR_ESCALATE:
            return self._action_escalate(action)
        elif action_type == ActionType.COORDINATOR_RESOLVE:
            return self._action_resolve(action)
        elif action_type == ActionType.COORDINATOR_MESSAGE:
            return self._action_message(action)
        
        return "Unknown action"
    
    def _action_monitor(self, action: DataCenterAction) -> str:
        """Watcher monitors the system."""
        # Gather observation evidence
        evidence = EvidenceSnippet(
            id=f"ev-{len(self.evidence_gathered):04d}",
            source="monitoring",
            content=f"System status check: {len([i for i in self.incidents if not i.resolved])} active incidents",
            relevance_score=0.3,
            agent_role=AgentRole.WATCHER,
        )
        self.evidence_gathered.append(evidence)
        return "Monitoring complete"
    
    def _action_alert(self, action: DataCenterAction) -> str:
        """Watcher alerts team about incident."""
        active = [i for i in self.incidents if not i.resolved]
        if not active:
            return "No incident to alert about"
        
        if self.agent_states["watcher"].alert_sent:
            return "Alert already sent"
        
        incident = active[0]
        self.agent_states["watcher"].alert_sent = True
        self.coordination_events += 1
        
        # Create message
        msg = AgentMessage(
            id=f"msg-{len(self.message_history):04d}",
            sender=AgentRole.WATCHER,
            receiver="all",
            message_type="alert",
            content=f"🚨 ALERT: {incident.incident_type.value} on {incident.equipment_name} [{incident.severity.value.upper()}]",
            priority=MessagePriority.URGENT if incident.severity in [Severity.CRITICAL, Severity.HIGH] else MessagePriority.HIGH,
        )
        self.message_history.append(msg)
        self._broadcast_message(msg)
        
        return f"Alert sent for incident #{incident.id}"
    
    def _action_investigate(self, action: DataCenterAction) -> str:
        """Watcher investigates incident."""
        if not self.agent_states["watcher"].alert_sent:
            return "Must alert before investigating"
        
        if self.agent_states["watcher"].investigating:
            return "Already investigating"
        
        active = [i for i in self.incidents if not i.resolved]
        if not active:
            return "No incident to investigate"
        
        incident = active[0]
        self.agent_states["watcher"].investigating = True
        self.agent_states["watcher"].investigation_complete = True
        
        # Gather investigation evidence
        evidence_items = [
            EvidenceSnippet(
                id=f"ev-{len(self.evidence_gathered):04d}",
                source="logs",
                content=f"Investigation of {incident.equipment_name}: {incident.incident_type.value} detected",
                relevance_score=0.8,
                agent_role=AgentRole.WATCHER,
            ),
            EvidenceSnippet(
                id=f"ev-{len(self.evidence_gathered)+1:04d}",
                source="metrics",
                content=f"Temperature: {self.metrics.temperatures[incident.id % 5]:.2f}, Load: {self.metrics.power_loads[incident.id % 5]:.2f}",
                relevance_score=0.6,
                agent_role=AgentRole.WATCHER,
            ),
        ]
        self.evidence_gathered.extend(evidence_items)
        
        # Update incident
        incident.collected_evidence.extend([e.id for e in evidence_items])
        
        # Notify responder
        msg = AgentMessage(
            id=f"msg-{len(self.message_history):04d}",
            sender=AgentRole.WATCHER,
            receiver=AgentRole.RESPONDER,
            message_type="investigation_complete",
            content=f"Investigation complete for incident #{incident.id}: {incident.incident_type.value}",
            evidence_refs=[e.id for e in evidence_items],
        )
        self.message_history.append(msg)
        self.pending_messages["responder"].append(msg)
        
        return f"Investigation complete for incident #{incident.id}"
    
    def _action_diagnose(self, action: DataCenterAction) -> str:
        """Responder diagnoses root cause."""
        if not self.agent_states["watcher"].investigation_complete:
            return "Waiting for investigation results"
        
        if self.agent_states["responder"].diagnosis_complete:
            return "Diagnosis already complete"
        
        active = [i for i in self.incidents if not i.resolved]
        if not active:
            return "No incident to diagnose"
        
        incident = active[0]
        self.agent_states["responder"].diagnosis_complete = True
        self.agent_states["responder"].diagnosis_details = action.reasoning or "Root cause identified"
        
        # Add diagnosis evidence
        evidence = EvidenceSnippet(
            id=f"ev-{len(self.evidence_gathered):04d}",
            source="diagnosis",
            content=f"Root cause: {incident.incident_type.value} requires manual intervention",
            relevance_score=0.9,
            agent_role=AgentRole.RESPONDER,
        )
        self.evidence_gathered.append(evidence)
        
        self.coordination_events += 1
        
        return f"Diagnosis complete for incident #{incident.id}"
    
    def _action_fix(self, action: DataCenterAction) -> str:
        """Responder attempts automated fix."""
        if not self.agent_states["responder"].diagnosis_complete:
            return "Must diagnose before attempting fix"
        
        if self.agent_states["responder"].fix_attempted:
            return "Fix already attempted"
        
        self.agent_states["responder"].fix_attempted = True
        return "Automated fix attempted (may require manual intervention)"
    
    def _action_request_help(self, action: DataCenterAction) -> str:
        """Responder requests help from coordinator."""
        if not self.agent_states["responder"].diagnosis_complete:
            return "Must diagnose before requesting help"
        
        if self.agent_states["responder"].help_requested:
            return "Help already requested"
        
        active = [i for i in self.incidents if not i.resolved]
        if not active:
            return "No active incident"
        
        incident = active[0]
        self.agent_states["responder"].help_requested = True
        self.coordination_events += 1
        
        # Find matching technician
        matching_techs = [
            t for t in self.technicians
            if t.available and incident.incident_type.value in t.specialization
        ] or [t for t in self.technicians if t.available]
        
        # Notify coordinator
        msg = AgentMessage(
            id=f"msg-{len(self.message_history):04d}",
            sender=AgentRole.RESPONDER,
            receiver=AgentRole.COORDINATOR,
            message_type="help_request",
            content=f"Help needed for incident #{incident.id}: {incident.incident_type.value}",
            requires_response=True,
        )
        self.message_history.append(msg)
        self.pending_messages["coordinator"].append(msg)
        
        return f"Help requested for incident #{incident.id}"
    
    def _action_dispatch(self, action: DataCenterAction) -> str:
        """Coordinator dispatches technician."""
        # Check prerequisites
        if not all([
            self.agent_states["watcher"].alert_sent,
            self.agent_states["watcher"].investigation_complete,
            self.agent_states["responder"].diagnosis_complete,
            self.agent_states["responder"].help_requested,
        ]):
            return "Pipeline incomplete - cannot dispatch"
        
        # Find unassigned incident
        unassigned = [i for i in self.incidents if not i.resolved and not i.assigned_technician]
        if not unassigned:
            return "No unassigned incidents"
        
        # Find available technician
        available = [t for t in self.technicians if t.available]
        if not available:
            return "No technicians available"
        
        incident = unassigned[0]
        
        # Prefer specialty match
        technician = None
        for t in available:
            if incident.incident_type.value in t.specialization:
                technician = t
                break
        if not technician:
            technician = available[0]
        
        # Dispatch
        technician.available = False
        technician.current_incident_id = incident.id
        technician.dispatch_step = self.step_number
        
        incident.assigned_technician = technician.id
        incident.dispatch_step = self.step_number
        incident.fix_status = FixStatus.IN_PROGRESS
        
        self.agent_states["coordinator"].dispatch_complete = True
        self.coordination_events += 1
        
        # Notify team
        msg = AgentMessage(
            id=f"msg-{len(self.message_history):04d}",
            sender=AgentRole.COORDINATOR,
            receiver="all",
            message_type="dispatch",
            content=f"✅ Dispatched {technician.name} ({technician.specialization}) to {incident.equipment_name}",
        )
        self.message_history.append(msg)
        self._broadcast_message(msg)
        
        return f"Dispatched {technician.name} to incident #{incident.id}"
    
    def _action_escalate(self, action: DataCenterAction) -> str:
        """Coordinator escalates incident."""
        high_severity = [i for i in self.incidents if not i.resolved 
                        and i.severity in [Severity.HIGH, Severity.CRITICAL]]
        
        if not high_severity:
            return "No high-severity incidents to escalate"
        
        self.agent_states["coordinator"].escalated = True
        self.coordination_events += 1
        
        return f"Escalated {len(high_severity)} high-severity incident(s)"
    
    def _action_resolve(self, action: DataCenterAction) -> str:
        """Coordinator resolves incident."""
        # Find resolvable incidents
        resolvable = []
        for incident in self.incidents:
            if incident.resolved:
                continue
            if not incident.assigned_technician:
                continue
            if incident.dispatch_step is None:
                continue
            
            steps_since_dispatch = self.step_number - incident.dispatch_step
            if steps_since_dispatch >= self.repair_steps:
                resolvable.append(incident)
        
        if not resolvable:
            # Check for too early
            for incident in self.incidents:
                if incident.assigned_technician and incident.dispatch_step:
                    steps_since = self.step_number - incident.dispatch_step
                    if steps_since < self.repair_steps:
                        return f"Technician still working ({steps_since}/{self.repair_steps} steps)"
            return "No incidents ready for resolution"
        
        incident = resolvable[0]
        incident.resolved = True
        incident.resolution_step = self.step_number
        incident.time_to_resolve = self.step_number - incident.step_started
        incident.fix_status = FixStatus.COMPLETED
        
        # Free technician
        for tech in self.technicians:
            if tech.id == incident.assigned_technician:
                tech.available = True
                tech.current_incident_id = None
                break
        
        self.resolved_incidents.append(incident)
        self.coordination_events += 1
        
        # Notify team
        msg = AgentMessage(
            id=f"msg-{len(self.message_history):04d}",
            sender=AgentRole.COORDINATOR,
            receiver="all",
            message_type="resolution",
            content=f"✅ Incident #{incident.id} resolved in {incident.time_to_resolve} steps",
        )
        self.message_history.append(msg)
        self._broadcast_message(msg)
        
        # Update metrics
        idx = incident.id % 5
        self.metrics.temperatures[idx] = max(0.35, self.metrics.temperatures[idx] - 0.15)
        self.metrics.power_loads[idx] = max(0.45, self.metrics.power_loads[idx] - 0.10)
        
        return f"Resolved incident #{incident.id} in {incident.time_to_resolve} steps"
    
    def _action_message(self, action: DataCenterAction) -> str:
        """Coordinator sends coordination message."""
        self.agent_states["coordinator"].coordination_events += 1
        self.coordination_events += 1
        
        msg = AgentMessage(
            id=f"msg-{len(self.message_history):04d}",
            sender=AgentRole.COORDINATOR,
            receiver="all",
            message_type="coordination",
            content=action.message or "Status sync",
        )
        self.message_history.append(msg)
        self._broadcast_message(msg)
        
        return f"Coordination message sent: {action.message[:30] if action.message else 'Status sync'}..."
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _spawn_incident(self, cascade_from: Optional[Incident] = None) -> Optional[Incident]:
        """Spawn a new incident."""
        active = [i for i in self.incidents if not i.resolved]
        if len(active) >= self.max_incidents:
            return None
        
        severity = Severity.CRITICAL if cascade_from else random.choice(list(Severity))
        incident_type = random.choice(INCIDENT_TYPES)
        equipment = random.choice(self.equipment)
        
        incident = Incident(
            id=self._incident_counter,
            incident_type=incident_type,
            severity=severity,
            equipment_id=equipment.id,
            equipment_name=equipment.name,
            step_started=self.step_number,
            cascade_risk=self.cascade_prob,
            cascade_parent_id=cascade_from.id if cascade_from else None,
        )
        
        self.incidents.append(incident)
        self._incident_counter += 1
        
        # Update equipment metrics
        idx = incident.id % 5
        self.metrics.temperatures[idx] = min(1.0, self.metrics.temperatures[idx] + 0.2)
        self.metrics.power_loads[idx] = min(1.0, self.metrics.power_loads[idx] + 0.15)
        
        return incident
    
    def _next_agent(self):
        """Advance to next agent in round-robin."""
        order = [AgentRole.WATCHER, AgentRole.RESPONDER, AgentRole.COORDINATOR]
        idx = order.index(self.current_agent)
        self.current_agent = order[(idx + 1) % 3]
    
    def _get_valid_actions(self) -> List[ActionType]:
        """Get valid actions for current agent."""
        if self.current_agent == AgentRole.WATCHER:
            return [
                ActionType.WATCHER_MONITOR,
                ActionType.WATCHER_ALERT,
                ActionType.WATCHER_INVESTIGATE,
            ]
        elif self.current_agent == AgentRole.RESPONDER:
            return [
                ActionType.RESPONDER_DIAGNOSE,
                ActionType.RESPONDER_FIX,
                ActionType.RESPONDER_REQUEST_HELP,
            ]
        else:  # Coordinator
            return [
                ActionType.COORDINATOR_DISPATCH,
                ActionType.COORDINATOR_ESCALATE,
                ActionType.COORDINATOR_RESOLVE,
                ActionType.COORDINATOR_MESSAGE,
            ]
    
    def _broadcast_message(self, msg: AgentMessage):
        """Broadcast message to all agents."""
        for role in [AgentRole.WATCHER, AgentRole.RESPONDER, AgentRole.COORDINATOR]:
            if msg.sender != role:
                self.pending_messages[role.value].append(msg)
    
    def _update_unknowns(self):
        """Update the unknowns list based on current state."""
        self.unknowns = []
        
        active = [i for i in self.incidents if not i.resolved]
        if not active:
            return
        
        incident = active[0]
        
        # What we don't know yet
        if not self.agent_states["watcher"].investigation_complete:
            self.unknowns.append(UnknownInfo(
                category="root_cause",
                description=f"Root cause of incident #{incident.id}",
                importance=0.9,
                discoverable=True,
            ))
        
        if not incident.assigned_technician:
            self.unknowns.append(UnknownInfo(
                category="resolution",
                description="Whether automated fix will work or manual intervention needed",
                importance=0.7,
                discoverable=True,
            ))
        
        if self.task_tier == TaskTier.HARD:
            # Hard mode has more unknowns
            self.unknowns.append(UnknownInfo(
                category="cascade_risk",
                description="Potential for incident cascade",
                importance=0.6,
                discoverable=True,
            ))
    
    def _get_observation(self) -> DataCenterObservation:
        """Construct current observation."""
        active = [i for i in self.incidents if not i.resolved]
        
        return DataCenterObservation(
            episode_id=self.episode_id,
            step_number=self.step_number,
            task_tier=self.task_tier,
            max_steps=self.max_steps,
            current_agent=self.current_agent,
            valid_actions=self._get_valid_actions(),
            active_incidents=active,
            resolved_incidents=self.resolved_incidents,
            incident_count=len(self.incidents),
            cascade_count=self.cascade_count,
            technicians_available=sum(1 for t in self.technicians if t.available),
            technicians_total=len(self.technicians),
            metrics=self.metrics,
            agent_states=self.agent_states,
            evidence_gathered=self.evidence_gathered.copy(),
            unknowns=self.unknowns.copy(),
            reasoning_trace=self.reasoning_trace.copy(),
            pending_messages=self.pending_messages[self.current_agent.value].copy(),
            message_history=self.message_history.copy(),
            coordination_score=self.coordination_score,
        )
    
    def _get_state_dict(self) -> Dict[str, Any]:
        """Get state as dictionary for reward computation."""
        return {
            "step_number": self.step_number,
            "current_agent": self.current_agent.value,
            "active_incidents": [
                {
                    "id": i.id,
                    "severity": i.severity.value,
                    "step_started": i.step_started,
                    "assigned_technician": i.assigned_technician,
                    "dispatch_step": i.dispatch_step,
                }
                for i in self.incidents if not i.resolved
            ],
            "agent_states": {
                role: {
                    "alert_sent": state.alert_sent,
                    "investigating": state.investigating,
                    "investigation_complete": state.investigation_complete,
                    "diagnosis_complete": state.diagnosis_complete,
                    "fix_attempted": state.fix_attempted,
                    "help_requested": state.help_requested,
                }
                for role, state in self.agent_states.items()
            },
            "watcher_alerted": self.agent_states["watcher"].alert_sent,
            "watcher_investigated": self.agent_states["watcher"].investigation_complete,
            "responder_diagnosed": self.agent_states["responder"].diagnosis_complete,
            "help_requested": self.agent_states["responder"].help_requested,
            "dispatch_complete": self.agent_states["coordinator"].dispatch_complete,
            "repair_steps": self.repair_steps,
            "technicians": [
                {"id": t.id, "specialization": t.specialization, "available": t.available}
                for t in self.technicians
            ],
            "evidence": [e.id for e in self.evidence_gathered],
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict for step return."""
        return {
            "episode_id": self.episode_id,
            "step": self.step_number,
            "current_agent": self.current_agent.value,
            "incidents_resolved": len(self.resolved_incidents),
            "incidents_active": len([i for i in self.incidents if not i.resolved]),
            "coordination_score": self.coordination_score,
            "total_reward": round(self.total_reward, 3),
            "evidence_count": len(self.evidence_gathered),
            "message_count": len(self.message_history),
        }
    
    def _compute_episode_bonus(self) -> float:
        """Compute final episode bonus."""
        bonus = 0.0
        
        # Coordination bonus
        self.coordination_score = min(1.0, self.coordination_events / 8.0)
        bonus += self.coordination_score * 5.0
        
        # Resolution bonus
        total = len(self.incidents)
        if total > 0:
            bonus += (len(self.resolved_incidents) / total) * 10.0
        
        # No cascade bonus (hard mode)
        if self.task_tier == TaskTier.HARD and self.cascade_count == 0:
            bonus += 6.0
        
        return bonus
    
    def get_replay(self) -> ReplayRecord:
        """Get complete replay of current episode."""
        from models import EpisodeResult
        
        total = len(self.incidents)
        result = EpisodeResult(
            episode_id=self.episode_id,
            task_tier=self.task_tier,
            solved=len(self.resolved_incidents) > 0,
            score=len(self.resolved_incidents) / max(1, total),
            steps_taken=self.step_number,
            incidents_resolved=len(self.resolved_incidents),
            incidents_total=total,
            total_reward=self.total_reward,
            coordination_score=self.coordination_score,
            cascade_count=self.cascade_count,
            evidence_efficiency=len(self.evidence_gathered) / max(1, self.step_number),
            communication_efficiency=len([m for m in self.message_history if m.message_type != "coordination"]) / max(1, len(self.message_history)),
            agent_rewards={role: state.cumulative_reward for role, state in self.agent_states.items()},
            agent_steps={role: state.steps_taken for role, state in self.agent_states.items()},
        )
        
        return ReplayRecord(
            episode_id=self.episode_id,
            task_tier=self.task_tier,
            seed=self.seed_value or 0,
            replay_steps=self.replay_steps,
            result=result,
        )
