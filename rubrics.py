"""
DataCenterOps — Rubric System for Debuggable Rewards
=====================================================

This module implements a comprehensive rubric system that provides
detailed reward breakdowns for debugging and analysis.

Inspired by the SRE Incident Environment's approach to grading.

Key Features:
- Component-level reward tracking
- SLA compliance scoring
- Evidence quality evaluation
- Coordination assessment
- Explanation generation
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from models import (
    ActionType,
    AgentRole,
    AgentState,
    DataCenterAction,
    DataCenterObservation,
    DataCenterReward,
    EvidenceSnippet,
    Incident,
    RewardBreakdown,
    Severity,
    TaskTier,
)

if TYPE_CHECKING:
    from environment import DataCenterOpsEnv


# =============================================================================
# Rubric Base Classes
# =============================================================================

class Rubric(ABC):
    """
    Abstract base class for rubrics.
    
    A rubric evaluates some aspect of agent performance and returns
    a component of the reward with explanation.
    """
    
    name: str = "base_rubric"
    description: str = "Base rubric class"
    
    @abstractmethod
    def compute(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute reward component."""
        pass
    
    @abstractmethod
    def explain(self, score: float) -> str:
        """Generate human-readable explanation."""
        pass


class CompositeRubric(Rubric):
    """
    A rubric composed of multiple sub-rubrics.
    
    Each sub-rubric evaluates a different aspect of performance.
    """
    
    def __init__(self, rubrics: List[Rubric], weights: Optional[List[float]] = None):
        self.rubrics = rubrics
        if weights is None:
            self.weights = [1.0] * len(rubrics)
        else:
            assert len(weights) == len(rubrics)
            self.weights = weights
    
    def compute(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> float:
        total = 0.0
        for rubric, weight in zip(self.rubrics, self.weights):
            total += weight * rubric.compute(observation, action, prev_state)
        return total
    
    def explain(self, score: float) -> str:
        return f"Composite rubric score: {score:.3f}"
    
    def get_breakdown(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Get breakdown by sub-rubric."""
        return {
            rubric.name: rubric.compute(observation, action, prev_state)
            for rubric in self.rubrics
        }


# =============================================================================
# Incident Handling Rubrics
# =============================================================================

class IncidentDetectionRubric(Rubric):
    """Evaluates incident detection performance."""
    
    name = "incident_detection"
    description = "Rewards timely incident detection"
    
    def __init__(self, urgency_decay: float = 0.1):
        self.urgency_decay = urgency_decay
    
    def compute(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> float:
        if action.action_type != ActionType.WATCHER_ALERT:
            return 0.0
        
        if prev_state is None:
            return 0.0
        
        # Check if there are active incidents that haven't been alerted
        watcher_state = prev_state.get("agent_states", {}).get("watcher", {})
        if watcher_state.get("alert_sent", False):
            return -0.5  # Duplicate alert penalty
        
        # Check if there are active incidents
        active_incidents = prev_state.get("active_incidents", [])
        if not active_incidents:
            return -0.3  # False alert
        
        # Calculate timeliness bonus
        incident = active_incidents[0]
        steps_since_start = observation.step_number - incident.get("step_started", observation.step_number)
        timeliness = max(0.0, 1.0 - self.urgency_decay * steps_since_start)
        
        # Severity bonus
        severity_map = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        severity_bonus = 0.5 * (severity_map.get(incident.get("severity", "medium").lower(), 1) / 3)
        
        return 2.0 + timeliness + severity_bonus
    
    def explain(self, score: float) -> str:
        if score > 2.0:
            return f"Excellent incident detection (+{score:.2f})"
        elif score > 0:
            return f"Good detection, but could be faster (+{score:.2f})"
        elif score < 0:
            return f"Detection issue ({score:.2f})"
        return "No detection activity (0.00)"


class InvestigationRubric(Rubric):
    """Evaluates investigation thoroughness."""
    
    name = "investigation"
    description = "Rewards thorough investigation with evidence gathering"
    
    def __init__(self, min_evidence: int = 2):
        self.min_evidence = min_evidence
    
    def compute(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> float:
        if action.action_type != ActionType.WATCHER_INVESTIGATE:
            return 0.0
        
        if prev_state is None:
            return 0.0
        
        watcher_state = prev_state.get("agent_states", {}).get("watcher", {})
        
        # Must alert first
        if not watcher_state.get("alert_sent", False):
            return -1.0  # Investigating without alerting
        
        # Already investigated?
        if watcher_state.get("investigating", False):
            return -0.5  # Duplicate investigation
        
        # Bonus for evidence gathered
        evidence_count = len([e for e in observation.evidence_gathered 
                             if e.agent_role == AgentRole.WATCHER])
        evidence_bonus = min(1.0, evidence_count * 0.2)
        
        # Check if investigation was thorough
        evidence_quality = 0.0
        for evidence in observation.evidence_gathered:
            if evidence.agent_role == AgentRole.WATCHER:
                evidence_quality += evidence.relevance_score * 0.3
        
        return 1.5 + evidence_bonus + evidence_quality
    
    def explain(self, score: float) -> str:
        if score > 2.0:
            return f"Thorough investigation with quality evidence (+{score:.2f})"
        elif score > 1.0:
            return f"Good investigation (+{score:.2f})"
        elif score < 0:
            return f"Investigation issue ({score:.2f})"
        return "No investigation (0.00)"


class DiagnosisRubric(Rubric):
    """Evaluates diagnosis accuracy and depth."""
    
    name = "diagnosis"
    description = "Rewards accurate and detailed diagnosis"
    
    def compute(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> float:
        if action.action_type != ActionType.RESPONDER_DIAGNOSE:
            return 0.0
        
        if prev_state is None:
            return 0.0
        
        responder_state = prev_state.get("agent_states", {}).get("responder", {})
        watcher_state = prev_state.get("agent_states", {}).get("watcher", {})
        
        # Need prior investigation
        if not watcher_state.get("investigation_complete", False):
            return -0.8  # Diagnosing without investigation
        
        # Already diagnosed?
        if responder_state.get("diagnosis_complete", False):
            return -0.4  # Duplicate diagnosis
        
        # Evidence used in diagnosis
        evidence_used = len(observation.reasoning_trace[-1].evidence_used) if observation.reasoning_trace else 0
        evidence_bonus = min(0.5, evidence_used * 0.15)
        
        # Reasoning quality
        reasoning_bonus = 0.0
        if action.reasoning and len(action.reasoning) > 20:
            reasoning_bonus = 0.3
        
        return 1.5 + evidence_bonus + reasoning_bonus
    
    def explain(self, score: float) -> str:
        if score > 2.0:
            return f"Excellent diagnosis with reasoning (+{score:.2f})"
        elif score > 1.0:
            return f"Good diagnosis (+{score:.2f})"
        elif score < 0:
            return f"Diagnosis issue ({score:.2f})"
        return "No diagnosis (0.00)"


class DispatchRubric(Rubric):
    """Evaluates dispatch decisions."""
    
    name = "dispatch"
    description = "Rewards appropriate technician dispatch"
    
    def compute(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> float:
        if action.action_type != ActionType.COORDINATOR_DISPATCH:
            return 0.0
        
        if prev_state is None:
            return 0.0
        
        coordinator_state = prev_state.get("agent_states", {}).get("coordinator", {})
        responder_state = prev_state.get("agent_states", {}).get("responder", {})
        
        # Need complete pipeline before dispatch
        required_complete = (
            prev_state.get("watcher_alerted", False) and
            prev_state.get("watcher_investigated", False) and
            responder_state.get("diagnosis_complete", False) and
            responder_state.get("help_requested", False)
        )
        
        if not required_complete:
            missing = sum([
                not prev_state.get("watcher_alerted", False),
                not prev_state.get("watcher_investigated", False),
                not responder_state.get("diagnosis_complete", False),
                not responder_state.get("help_requested", False),
            ])
            return -0.4 * missing
        
        # Check technician availability
        if observation.technicians_available <= 0:
            return -0.3  # No technicians available
        
        # Check for unassigned incidents
        unassigned = [i for i in observation.active_incidents 
                     if not i.assigned_technician]
        if not unassigned:
            return -0.2  # No incidents to dispatch to
        
        # Specialist matching bonus
        incident = unassigned[0]
        if action.technician_id:
            # Check if technician specialty matches incident type
            technicians = prev_state.get("technicians", [])
            tech = next((t for t in technicians if t["id"] == action.technician_id), None)
            if tech and incident.incident_type.value in tech.get("specialization", ""):
                return 2.5  # Specialist match bonus
        
        return 2.0
    
    def explain(self, score: float) -> str:
        if score > 2.0:
            return f"Excellent dispatch with specialist match (+{score:.2f})"
        elif score > 1.5:
            return f"Good dispatch (+{score:.2f})"
        elif score < 0:
            return f"Dispatch issue ({score:.2f})"
        return "No dispatch (0.00)"


class ResolutionRubric(Rubric):
    """Evaluates incident resolution."""
    
    name = "resolution"
    description = "Rewards successful incident resolution"
    
    def __init__(self, max_steps: int = 60):
        self.max_steps = max_steps
    
    def compute(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> float:
        if action.action_type != ActionType.COORDINATOR_RESOLVE:
            return 0.0
        
        if prev_state is None:
            return 0.0
        
        # Find resolvable incidents
        resolvable = []
        too_early = []
        
        for incident in observation.active_incidents:
            if incident.assigned_technician and incident.dispatch_step is not None:
                steps_since_dispatch = observation.step_number - incident.dispatch_step
                repair_steps = prev_state.get("repair_steps", 4)
                
                if steps_since_dispatch >= repair_steps:
                    resolvable.append(incident)
                else:
                    too_early.append(incident)
        
        if resolvable:
            incident = resolvable[0]
            
            # Base resolution reward
            base = 12.0
            
            # Speed bonus
            time_to_resolve = observation.step_number - incident.step_started
            speed_bonus = max(0.0, 1.0 - time_to_resolve / self.max_steps) * 6.0
            
            # Severity bonus
            severity_map = {"critical": 3, "high": 2, "medium": 1, "low": 0}
            severity_bonus = (severity_map.get(incident.severity.value, 1) + 1) * 2.5
            
            return base + speed_bonus + severity_bonus
        
        elif too_early:
            return -0.3  # Technician not done yet
        
        return -0.6  # No incident to resolve
    
    def explain(self, score: float) -> str:
        if score > 15.0:
            return f"Excellent resolution (+{score:.2f})"
        elif score > 10.0:
            return f"Good resolution (+{score:.2f})"
        elif score < 0:
            return f"Resolution issue ({score:.2f})"
        return "No resolution attempt (0.00)"


# =============================================================================
# Quality Rubrics
# =============================================================================

class OrderingRubric(Rubric):
    """Evaluates correct action ordering."""
    
    name = "ordering"
    description = "Rewards correct sequential ordering of actions"
    
    # Define correct ordering
    CORRECT_ORDER = [
        ActionType.WATCHER_ALERT,
        ActionType.WATCHER_INVESTIGATE,
        ActionType.RESPONDER_DIAGNOSE,
        ActionType.RESPONDER_REQUEST_HELP,
        ActionType.COORDINATOR_DISPATCH,
        ActionType.COORDINATOR_RESOLVE,
    ]
    
    def compute(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> float:
        if prev_state is None:
            return 0.0
        
        # Check if action is in correct order
        action_idx = None
        for i, a in enumerate(self.CORRECT_ORDER):
            if action.action_type == a:
                action_idx = i
                break
        
        if action_idx is None:
            return 0.0  # Not a sequenced action
        
        # Check prerequisites
        required_complete = True
        for i in range(action_idx):
            required_action = self.CORRECT_ORDER[i]
            
            if required_action == ActionType.WATCHER_ALERT:
                required_complete = prev_state.get("watcher_alerted", False)
            elif required_action == ActionType.WATCHER_INVESTIGATE:
                required_complete = prev_state.get("watcher_investigated", False)
            elif required_action == ActionType.RESPONDER_DIAGNOSE:
                required_complete = prev_state.get("responder_diagnosed", False)
            elif required_action == ActionType.RESPONDER_REQUEST_HELP:
                required_complete = prev_state.get("help_requested", False)
            elif required_action == ActionType.COORDINATOR_DISPATCH:
                required_complete = prev_state.get("dispatch_complete", False)
            
            if not required_complete:
                return -0.5 * (action_idx - i)  # Penalty proportional to skip
        
        return 0.3  # Small bonus for correct ordering
    
    def explain(self, score: float) -> str:
        if score > 0:
            return f"Correct action ordering (+{score:.2f})"
        elif score < 0:
            return f"Skipped prerequisite actions ({score:.2f})"
        return "Not a sequenced action (0.00)"


class CoordinationRubric(Rubric):
    """Evaluates inter-agent coordination."""
    
    name = "coordination"
    description = "Rewards effective communication and coordination"
    
    def compute(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> float:
        score = 0.0
        
        # Message actions get coordination points
        if action.action_type in [
            ActionType.WATCHER_ALERT,
            ActionType.RESPONDER_REQUEST_HELP,
            ActionType.COORDINATOR_MESSAGE,
        ]:
            score += 0.5
        
        # Check message quality
        if action.action_type == ActionType.COORDINATOR_MESSAGE:
            if action.message and len(action.message) > 10:
                score += 0.3  # Substantive message
        
        # Coordination efficiency (not too many messages)
        total_messages = len(observation.message_history)
        if total_messages > 10:
            score -= 0.1 * (total_messages - 10)  # Spam penalty

        # Penalize repetitive coordination chatter while incidents remain unresolved
        if action.action_type == ActionType.COORDINATOR_MESSAGE and observation.active_incidents:
            if len(observation.message_history) >= 2:
                last_two = [m.message_type for m in observation.message_history[-2:]]
                if last_two == ["coordination", "coordination"]:
                    score -= 0.4
        
        return score
    
    def explain(self, score: float) -> str:
        if score > 0.5:
            return f"Good coordination (+{score:.2f})"
        elif score > 0:
            return f"Some coordination (+{score:.2f})"
        elif score < 0:
            return f"Communication spam ({score:.2f})"
        return "No coordination (0.00)"


class EvidenceQualityRubric(Rubric):
    """Evaluates quality of evidence gathered."""
    
    name = "evidence_quality"
    description = "Rewards gathering relevant, high-quality evidence"
    
    def compute(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> float:
        if action.action_type not in [
            ActionType.WATCHER_INVESTIGATE,
            ActionType.RESPONDER_DIAGNOSE,
        ]:
            return 0.0
        
        # Get evidence from this action
        new_evidence = [e for e in observation.evidence_gathered 
                       if e.timestamp and 
                       (prev_state is None or 
                        len(prev_state.get("evidence", [])) < len(observation.evidence_gathered))]
        
        if not new_evidence:
            return 0.0
        
        # Score based on relevance and coverage
        relevance = sum(e.relevance_score for e in new_evidence) / len(new_evidence)
        
        # Diversity bonus (different sources)
        sources = set(e.source for e in new_evidence)
        diversity_bonus = min(0.5, len(sources) * 0.15)
        
        return relevance * 0.5 + diversity_bonus
    
    def explain(self, score: float) -> str:
        if score > 0.5:
            return f"High-quality evidence (+{score:.2f})"
        elif score > 0:
            return f"Some useful evidence (+{score:.2f})"
        return "No evidence gathered (0.00)"


class StagnationRubric(Rubric):
    """Penalizes passive/no-progress actions when urgent incidents exist."""

    name = "stagnation"
    description = "Penalizes no-op behavior under active incident pressure"

    def compute(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> float:
        if not observation.active_incidents:
            return 0.0

        severity_weight = {"critical": 1.4, "high": 1.0, "medium": 0.6, "low": 0.3}
        max_pressure = 0.0
        for incident in observation.active_incidents:
            age = max(0, observation.step_number - incident.step_started)
            sev = severity_weight.get(incident.severity.value, 0.6)
            pressure = sev + min(1.0, age / 20.0)
            max_pressure = max(max_pressure, pressure)

        passive_actions = {
            ActionType.WATCHER_MONITOR,
            ActionType.RESPONDER_FIX,
            ActionType.COORDINATOR_MESSAGE,
        }
        if action.action_type in passive_actions:
            return -0.4 * max_pressure

        return 0.0

    def explain(self, score: float) -> str:
        if score < 0:
            return f"Passive action under incident pressure ({score:.2f})"
        return "No stagnation penalty (0.00)"


# =============================================================================
# SLA Rubric
# =============================================================================

class SLARubric(Rubric):
    """Evaluates SLA compliance."""
    
    name = "sla"
    description = "Penalizes SLA violations based on incident age"
    
    # SLA thresholds in steps
    SLA_THRESHOLDS = {
        Severity.CRITICAL: 10,
        Severity.HIGH: 20,
        Severity.MEDIUM: 30,
        Severity.LOW: 40,
    }
    
    def compute(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> float:
        if not observation.active_incidents:
            return 0.0
        
        penalty = 0.0
        
        for incident in observation.active_incidents:
            age = observation.step_number - incident.step_started
            threshold = self.SLA_THRESHOLDS.get(incident.severity, 30)
            
            if age > threshold:
                # Progressive penalty
                violation_ratio = (age - threshold) / threshold
                severity_multiplier = {
                    Severity.CRITICAL: 3.0,
                    Severity.HIGH: 2.0,
                    Severity.MEDIUM: 1.0,
                    Severity.LOW: 0.5,
                }.get(incident.severity, 1.0)
                
                penalty -= 0.3 * violation_ratio * severity_multiplier
        
        return penalty
    
    def explain(self, score: float) -> str:
        if score < -1.0:
            return f"Severe SLA violations ({score:.2f})"
        elif score < 0:
            return f"Minor SLA violations ({score:.2f})"
        return "SLA compliant (0.00)"


# =============================================================================
# Composite DataCenter Rubric
# =============================================================================

class DataCenterRubric(CompositeRubric):
    """
    Main rubric for DataCenterOps environment.
    
    Combines all sub-rubrics with appropriate weights.
    """
    
    name = "datacenter_main"
    description = "Composite rubric for data center operations"
    
    def __init__(self, max_steps: int = 60):
        rubrics = [
            IncidentDetectionRubric(),
            InvestigationRubric(),
            DiagnosisRubric(),
            DispatchRubric(),
            ResolutionRubric(max_steps=max_steps),
            OrderingRubric(),
            CoordinationRubric(),
            EvidenceQualityRubric(),
            StagnationRubric(),
            SLARubric(),
        ]
        
        weights = [
            1.0,  # detection
            1.0,  # investigation
            1.0,  # diagnosis
            1.0,  # dispatch
            2.0,  # resolution (higher weight)
            0.5,  # ordering
            0.5,  # coordination
            0.3,  # evidence quality
            0.8,  # stagnation
            1.0,  # SLA
        ]
        
        super().__init__(rubrics, weights)
    
    def compute_full(
        self,
        observation: DataCenterObservation,
        action: DataCenterAction,
        prev_state: Optional[Dict[str, Any]] = None
    ) -> DataCenterReward:
        """
        Compute full reward with breakdown.
        
        Returns a DataCenterReward with detailed breakdown and explanation.
        """
        breakdown = RewardBreakdown()
        
        # Compute each component
        for rubric in self.rubrics:
            score = rubric.compute(observation, action, prev_state)
            
            # Map to breakdown fields
            if rubric.name == "incident_detection":
                breakdown.incident_alerted = max(0, score)
                if score < 0:
                    breakdown.invalid_action += abs(score)
            elif rubric.name == "investigation":
                breakdown.incident_investigated = max(0, score)
            elif rubric.name == "diagnosis":
                breakdown.incident_diagnosed = max(0, score)
            elif rubric.name == "dispatch":
                breakdown.incident_dispatched = max(0, score)
            elif rubric.name == "resolution":
                breakdown.incident_resolved = max(0, score)
            elif rubric.name == "ordering":
                if score > 0:
                    breakdown.correct_ordering = score
                else:
                    breakdown.ordering_violation = abs(score)
            elif rubric.name == "coordination":
                breakdown.coordination_bonus = max(0, score)
            elif rubric.name == "evidence_quality":
                breakdown.evidence_quality = max(0, score)
            elif rubric.name == "stagnation":
                if score < 0:
                    breakdown.time_penalty += abs(score)
            elif rubric.name == "sla":
                if score < 0:
                    breakdown.sla_penalty = abs(score)
        
        total = breakdown.total()
        
        # Generate explanation
        explanations = []
        for rubric, weight in zip(self.rubrics, self.weights):
            score = rubric.compute(observation, action, prev_state)
            if abs(score) > 0.01:
                explanations.append(rubric.explain(score))
        
        return DataCenterReward(
            total=total,
            breakdown=breakdown,
            message=" | ".join(explanations) if explanations else "No significant reward",
        )


# =============================================================================
# Utility Functions
# =============================================================================

def create_rubric_for_tier(tier: TaskTier) -> DataCenterRubric:
    """Create appropriate rubric for task tier."""
    max_steps_map = {
        TaskTier.EASY: 24,
        TaskTier.MEDIUM: 42,
        TaskTier.HARD: 60,
    }
    return DataCenterRubric(max_steps=max_steps_map[tier])
