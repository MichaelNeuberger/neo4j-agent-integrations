"""Data models for PSS Neo4j persistence layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class SessionStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    ERROR = "error"


class PhaseName(str, Enum):
    INITIALIZATION = "initialization"
    EXPLORATION = "exploration"
    CONVERGENCE = "convergence"
    RESONANCE = "resonance"
    STABILITY = "stability"
    INSTABILITY = "instability"


class DriftSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftPhase(str, Enum):
    STABLE = "stable"
    SHIFTING = "shifting"
    DRIFTED = "drifted"


class MemoryTier(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class AggregationStrategy(str, Enum):
    WEIGHTED_AVG = "weighted_avg"
    ATTENTION = "attention"


class ConsensusOutcome(str, Enum):
    AGREED = "agreed"
    DISAGREED = "disagreed"
    TIMEOUT = "timeout"


class AnomalyType(str, Enum):
    CROSS_CLUSTER_CONVERGENCE = "cross_cluster_convergence"
    SYSTEMIC_DRIFT = "systemic_drift"
    CLUSTER_DIVERGENCE = "cluster_divergence"


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Agent:
    agent_id: str
    name: str
    framework: str = ""
    version: str = ""


@dataclass
class AgentSession:
    session_id: str = field(default_factory=_new_id)
    agent_id: str = ""
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    status: SessionStatus = SessionStatus.ACTIVE


@dataclass
class SemanticState:
    state_id: str = field(default_factory=_new_id)
    vector: list[float] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    step: int = 0
    beta: float = 0.0
    mean_similarity: float = 0.0
    variance: float = 0.0


@dataclass
class DriftEvent:
    event_id: str = field(default_factory=_new_id)
    timestamp: Optional[datetime] = None
    drift_score: float = 0.0
    drift_phase: DriftPhase = DriftPhase.STABLE
    topic_switch: bool = False
    cosine_drop: float = 0.0
    mean_sim: float = 0.0
    variance: float = 0.0
    severity: DriftSeverity = DriftSeverity.LOW


@dataclass
class Phase:
    phase_id: str = field(default_factory=_new_id)
    name: PhaseName = PhaseName.INITIALIZATION
    entered_at: Optional[datetime] = None
    exited_at: Optional[datetime] = None
    srs_score: float = 0.0
    tc_score: float = 0.0
    fsm_state: str = ""
    be_score: float = 0.0
    markov_probability: float = 0.0
    rule_score: float = 0.0


@dataclass
class Memory:
    memory_id: str = field(default_factory=_new_id)
    tier: MemoryTier = MemoryTier.SHORT
    content_vector: list[float] = field(default_factory=list)
    importance: float = 0.0
    recency: float = 1.0
    access_count: int = 0
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    text_summary: str = ""


@dataclass
class Cluster:
    cluster_id: str = field(default_factory=_new_id)
    name: str = ""
    strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_AVG
    coupling_strength: float = 0.0
    created_at: Optional[datetime] = None


@dataclass
class Region:
    region_id: str = field(default_factory=_new_id)
    name: str = ""
    consensus_threshold: float = 0.6
    voting_weight_scheme: str = "influence"


@dataclass
class ConsensusEvent:
    event_id: str = field(default_factory=_new_id)
    timestamp: Optional[datetime] = None
    outcome: ConsensusOutcome = ConsensusOutcome.TIMEOUT
    vote_count: int = 0
    threshold_used: float = 0.6
    drift_score_consensus: float = 0.0


@dataclass
class AnomalyEvent:
    event_id: str = field(default_factory=_new_id)
    timestamp: Optional[datetime] = None
    anomaly_type: AnomalyType = AnomalyType.SYSTEMIC_DRIFT
    severity: float = 0.0
    affected_clusters: list[str] = field(default_factory=list)


@dataclass
class GlobalObserver:
    observer_id: str = field(default_factory=_new_id)
    pattern_vector: list[float] = field(default_factory=list)
    last_scan: Optional[datetime] = None
