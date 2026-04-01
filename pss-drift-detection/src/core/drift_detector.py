"""Drift detection bridge: PSS API → Neo4j Graph.

PSS (the patented product) does ALL computation:
  - Semantic state evolution (384-dim vector, 9-step beta)
  - 3-layer drift detection (topic switch, phase, API score)
  - Multi-resolution memory (short/medium/long with K-means++)
  - Phase detection (6×6 Markov + Rules hybrid)

Neo4j mirrors the results for:
  - Queryable drift history (temporal graph)
  - Phase transition paths (traversable chain)
  - Cross-session analytics (vector index, PageRank)
  - Multi-agent topology (clusters, regions, observer)

Drift detection strategy:
  PSS returns top_similarity (cosine between current message and accumulated
  context).  This is the most reliable signal: on-topic messages score 0.4-0.7,
  off-topic messages drop to 0.05-0.20.  We track a rolling average and flag
  drift when the current similarity drops significantly below the session's
  running mean.  No composite scores, no warmup heuristics — just the raw
  PSS similarity signal with a simple rolling baseline.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from src.core.pss_client import PSSClient
from src.persistence.models import (
    DriftEvent,
    DriftPhase,
    DriftSeverity,
    Phase,
    PhaseName,
    SemanticState,
)
from src.persistence.neo4j_state_store import Neo4jStateStore
from src.persistence.neo4j_phase_store import Neo4jPhaseStore
from src.persistence.neo4j_drift_event_store import Neo4jDriftEventStore


# How many steps of similarity history to track for the rolling average
ROLLING_WINDOW = 4

# Drift fires when current similarity is this far below the rolling average.
# Example: rolling avg 0.50, current 0.15 → drop = 0.35 → above 0.25 → DRIFT
SIMILARITY_DROP_THRESHOLD = 0.25

# Minimum steps before drift detection activates (PSS needs context first)
MIN_STEPS = 4

# Severity is based on how far similarity dropped below the rolling average
SEVERITY_LOW = 0.25
SEVERITY_MEDIUM = 0.35
SEVERITY_HIGH = 0.50

# Drift phase name mapping from PSS API string
_DRIFT_PHASE_MAP = {
    "stable": DriftPhase.STABLE,
    "shifting": DriftPhase.SHIFTING,
    "drifted": DriftPhase.DRIFTED,
}


def _classify_severity(drop: float) -> DriftSeverity:
    if drop < SEVERITY_LOW:
        return DriftSeverity.LOW
    elif drop < SEVERITY_MEDIUM:
        return DriftSeverity.MEDIUM
    elif drop < SEVERITY_HIGH:
        return DriftSeverity.HIGH
    return DriftSeverity.CRITICAL


class DriftDetector:
    """Bridge between PSS API and Neo4j persistence.

    Calls the real PSS API for all drift computation,
    then mirrors results into Neo4j for graph queries.

    Drift detection uses PSS top_similarity with a rolling baseline:
    when the current similarity drops far below the recent average,
    a drift event is created.
    """

    def __init__(
        self,
        pss_client: PSSClient,
        state_store: Neo4jStateStore,
        phase_store: Neo4jPhaseStore,
        drift_event_store: Neo4jDriftEventStore,
    ):
        self._pss = pss_client
        self._state_store = state_store
        self._phase_store = phase_store
        self._drift_event_store = drift_event_store

        # Map neo4j session_id → PSS session_id
        self._pss_sessions: dict[str, str] = {}
        # Per-session: step count, rolling similarity window
        self._step_counts: dict[str, int] = {}
        self._sim_history: dict[str, deque] = {}
        # Inline-store: buffer previous LLM response per session
        self._prev_responses: dict[str, str | None] = {}

    def process_input(self, session_id: str, message: str) -> dict:
        """Send message to PSS API, mirror results to Neo4j."""
        pss_session_id = self._pss_sessions.get(session_id)

        # Inline-store: send previous LLM response with this /run call
        prev_response = self._prev_responses.pop(session_id, None)

        # === Call the real PSS API ===
        pss_result = self._pss.run(
            message=message,
            session_id=pss_session_id,
            response=prev_response,
        )

        if session_id not in self._pss_sessions:
            self._pss_sessions[session_id] = pss_result["session_id"]
        pss_session_id = pss_result["session_id"]

        drift_score = pss_result.get("drift_score", 0.0)
        drift_detected_pss = pss_result.get("drift_detected", False)
        drift_phase_str = pss_result.get("drift_phase", "stable")
        context = pss_result.get("context", "")
        top_similarity = pss_result.get("top_similarity", 0.0)
        short_circuit = pss_result.get("short_circuit", False)

        drift_phase = _DRIFT_PHASE_MAP.get(drift_phase_str, DriftPhase.STABLE)

        # --- Rolling similarity baseline ---
        step_num = self._step_counts.get(session_id, 0)
        self._step_counts[session_id] = step_num + 1

        if session_id not in self._sim_history:
            self._sim_history[session_id] = deque(maxlen=ROLLING_WINDOW)

        sim_window = self._sim_history[session_id]
        rolling_avg = sum(sim_window) / len(sim_window) if sim_window else 0.0
        sim_drop = max(0.0, rolling_avg - top_similarity)

        # Update window (only add non-zero similarities — step 0 is always 0)
        if top_similarity > 0:
            sim_window.append(top_similarity)

        # Drift decision: similarity dropped significantly below rolling average
        topic_switched = (
            step_num >= MIN_STEPS
            and len(sim_window) >= 2
            and sim_drop >= SIMILARITY_DROP_THRESHOLD
        )
        severity = _classify_severity(sim_drop)

        # === Mirror to Neo4j: Semantic State ===
        prev_state = self._state_store.get_current_state(session_id)
        step = (prev_state.step + 1) if prev_state else 0

        state = SemanticState(
            step=step,
            beta=sim_drop,  # similarity drop as the drift metric
            mean_similarity=top_similarity,
            variance=drift_score,  # raw PSS drift_score preserved
            vector=[],
        )
        self._state_store.append_state(
            session_id, state, cosine_similarity=top_similarity
        )

        # === Mirror to Neo4j: Phase ===
        self._update_phase_from_pss(session_id, drift_phase, sim_drop, top_similarity)

        # === Mirror to Neo4j: Drift Event ===
        if topic_switched:
            event = DriftEvent(
                drift_score=sim_drop,
                drift_phase=drift_phase,
                topic_switch=True,
                cosine_drop=sim_drop,
                mean_sim=top_similarity,
                variance=drift_score,
                severity=severity,
            )
            self._drift_event_store.create_drift_event(
                session_id, state.state_id, event
            )

        return {
            "context": context,
            "drift_score": sim_drop,
            "pss_drift_score": drift_score,
            "drift_detected": topic_switched,
            "pss_drift_detected": drift_detected_pss,
            "drift_phase": drift_phase,
            "top_similarity": top_similarity,
            "similarity_drop": sim_drop,
            "rolling_avg": rolling_avg,
            "short_circuit": short_circuit,
            "severity": severity,
            "context_reset": False,
            "state_id": state.state_id,
            "step": step,
            "pss_session_id": pss_session_id,
        }

    def store_response(self, session_id: str, response: str) -> dict:
        """Buffer LLM response for inline-store on next /run call."""
        self._prev_responses[session_id] = response
        pss_session_id = self._pss_sessions.get(session_id)
        if not pss_session_id:
            return {"error": "No PSS session — call process_input first"}
        return self._pss.store(pss_session_id, response)

    def get_pss_session_id(self, session_id: str) -> Optional[str]:
        return self._pss_sessions.get(session_id)

    def _update_phase_from_pss(
        self, session_id: str, drift_phase: DriftPhase,
        sim_drop: float, similarity: float,
    ) -> None:
        current = self._phase_store.get_current_phase(session_id)
        new_phase = self._infer_phase(drift_phase, sim_drop, similarity, current)
        if current is None or new_phase != current.name:
            self._phase_store.set_phase(session_id, Phase(
                name=new_phase,
                srs_score=similarity,
                tc_score=similarity,
                be_score=sim_drop,
            ))

    @staticmethod
    def _infer_phase(
        drift_phase: DriftPhase, sim_drop: float,
        similarity: float, current: Optional[Phase],
    ) -> PhaseName:
        if current is None:
            return PhaseName.INITIALIZATION
        if drift_phase == DriftPhase.DRIFTED:
            return PhaseName.INSTABILITY
        if sim_drop < 0.1 and similarity > 0.6:
            return PhaseName.STABILITY
        if sim_drop < 0.15 and similarity > 0.4:
            return PhaseName.RESONANCE
        if similarity > 0.3 and sim_drop < 0.25:
            return PhaseName.CONVERGENCE
        return PhaseName.EXPLORATION
