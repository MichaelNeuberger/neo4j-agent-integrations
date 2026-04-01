"""Client for the real PSS (Persistent Semantic State) API.

PSS is the patented product — this client calls the production API
and returns the results. Neo4j is only a persistence/query layer on top.

API Docs: see /docs/API.md in the PSS repository.
"""

from __future__ import annotations

import os
from typing import Optional

import requests

PSS_BASE_URL = os.environ.get("PSS_BASE_URL", "https://pss.versino.de/api/v1")
PSS_API_KEY = os.environ.get("PSS_API_KEY", "")


class PSSClient:
    """HTTP client for the PSS REST API (Layers 1–5)."""

    def __init__(
        self,
        base_url: str = PSS_BASE_URL,
        api_key: str = PSS_API_KEY,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        })

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = self._session.get(self._url(path), params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: dict | None = None) -> dict:
        r = self._session.post(self._url(path), json=data or {})
        r.raise_for_status()
        return r.json()

    def _put(self, path: str, data: dict | None = None) -> dict:
        r = self._session.put(self._url(path), json=data or {})
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> dict:
        r = self._session.delete(self._url(path))
        r.raise_for_status()
        return r.json()

    # === Health ===

    def health(self) -> dict:
        return self._get("/health")

    # === Layer 1: Core Session ===

    def run(
        self,
        message: str,
        session_id: Optional[str] = None,
        response: Optional[str] = None,
        short_circuit_threshold: float = 0.85,
        reset_context: bool = False,
    ) -> dict:
        """Send message to PSS, get compressed context + drift analysis.

        Returns: session_id, context, top_similarity, short_circuit,
                 drift_score, drift_detected, drift_phase
        """
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        if response:
            payload["response"] = response
        payload["short_circuit_threshold"] = short_circuit_threshold
        payload["reset_context"] = reset_context
        return self._post("/run", payload)

    def store(self, session_id: str, response: str) -> dict:
        """Store LLM response so PSS learns."""
        return self._post("/store", {
            "session_id": session_id,
            "response": response,
        })

    # === Layer 1b: Session Control ===

    def create_session(
        self,
        dimension: int = 384,
        model_name: str = "all-MiniLM-L6-v2",
        use_meta_pss: bool = False,
        enable_topic_switch: bool = True,
    ) -> dict:
        """Create a PSS session."""
        return self._post("/session/create", {
            "dimension": dimension,
            "model_name": model_name,
            "use_meta_pss": use_meta_pss,
            "enable_topic_switch": enable_topic_switch,
        })

    def export_session(self, session_id: str) -> dict:
        """Export full PSS state with checksum."""
        return self._get(f"/session/{session_id}/export")

    def import_session(self, session_id: str, state_dict: dict) -> dict:
        """Restore session from exported state."""
        return self._post(f"/session/{session_id}/import", {
            "state_dict": state_dict,
        })

    def add_anchor(self, session_id: str, embedding: list[float]) -> dict:
        """Add semantic drift anchor."""
        return self._post(f"/session/{session_id}/anchor", {
            "embedding": embedding,
        })

    def get_anchor_score(self, session_id: str) -> dict:
        """Get current anchor score and drift status."""
        return self._get(f"/session/{session_id}/anchor_score")

    def inject_memory(
        self,
        session_id: str,
        embedding: list[float],
        text: str,
        tier: str = "short_term",
        importance: float = 1.0,
    ) -> dict:
        """Inject memory entry into a tier."""
        return self._post(f"/session/{session_id}/memory", {
            "embedding": embedding,
            "text": text,
            "tier": tier,
            "importance": importance,
        })

    def set_isolation(
        self,
        session_id: str,
        level: str = "OPEN",
        similarity_threshold: float = 0.8,
    ) -> dict:
        """Set input isolation level."""
        return self._put(f"/session/{session_id}/isolation", {
            "level": level,
            "similarity_threshold": similarity_threshold,
        })

    def add_trigger(self, session_id: str, keyword: str) -> dict:
        """Add resonance trigger."""
        return self._post(f"/session/{session_id}/trigger", {
            "keyword": keyword,
        })

    # === Layer 2: Clusters ===

    def create_cluster(
        self,
        name: str,
        aggregation_mode: str = "weighted_average",
        coupling_factor: float = 0.0,
    ) -> dict:
        return self._post("/cluster/", {
            "name": name,
            "aggregation_mode": aggregation_mode,
            "coupling_factor": coupling_factor,
        })

    def list_clusters(self) -> list[dict]:
        return self._get("/cluster/")

    def get_cluster(self, cluster_id: str) -> dict:
        return self._get(f"/cluster/{cluster_id}")

    def delete_cluster(self, cluster_id: str) -> dict:
        return self._delete(f"/cluster/{cluster_id}")

    def cluster_run(
        self, cluster_id: str, message: str,
        response: Optional[str] = None,
        short_circuit_threshold: Optional[float] = None,
    ) -> dict:
        payload: dict = {"message": message}
        if response:
            payload["response"] = response
        if short_circuit_threshold is not None:
            payload["short_circuit_threshold"] = short_circuit_threshold
        return self._post(f"/cluster/{cluster_id}/run", payload)

    def cluster_store(self, cluster_id: str, message: str, response: str) -> dict:
        return self._post(f"/cluster/{cluster_id}/store", {
            "message": message,
            "response": response,
        })

    def cluster_feedback(self, cluster_id: str) -> dict:
        """Apply G4 coupling feedback."""
        return self._post(f"/cluster/{cluster_id}/feedback")

    def add_cluster_member(self, cluster_id: str, session_id: str) -> dict:
        return self._post(f"/cluster/{cluster_id}/members", {
            "session_id": session_id,
        })

    def remove_cluster_member(self, cluster_id: str, session_id: str) -> dict:
        return self._delete(f"/cluster/{cluster_id}/members/{session_id}")

    # === Layer 3: Regions ===

    def create_region(
        self,
        name: str,
        consensus_threshold: float = 0.5,
        vote_window_seconds: float = 60.0,
    ) -> dict:
        return self._post("/region/", {
            "name": name,
            "consensus_threshold": consensus_threshold,
            "vote_window_seconds": vote_window_seconds,
        })

    def list_regions(self) -> list[dict]:
        return self._get("/region/")

    def get_region(self, region_id: str) -> dict:
        return self._get(f"/region/{region_id}")

    def delete_region(self, region_id: str) -> dict:
        return self._delete(f"/region/{region_id}")

    def add_region_cluster(self, region_id: str, cluster_id: str) -> dict:
        return self._post(f"/region/{region_id}/clusters", {
            "cluster_id": cluster_id,
        })

    def remove_region_cluster(self, region_id: str, cluster_id: str) -> dict:
        return self._delete(f"/region/{region_id}/clusters/{cluster_id}")

    def get_region_events(self, region_id: str, limit: int = 20) -> list[dict]:
        return self._get(f"/region/{region_id}/events", {"limit": limit})

    # === Layer 4: Global Observer ===

    def create_observer(
        self,
        sample_interval_seconds: float = 30.0,
        region_ids: list[str] | None = None,
    ) -> dict:
        return self._post("/observer/", {
            "sample_interval_seconds": sample_interval_seconds,
            "region_ids": region_ids or [],
        })

    def get_observer_summary(self) -> dict:
        return self._get("/observer/summary")

    def observer_sample(self) -> dict:
        """Trigger manual sample."""
        return self._post("/observer/sample")

    def get_anomalies(self, limit: int = 20) -> list[dict]:
        return self._get("/observer/anomalies", {"limit": limit})

    def clear_anomalies(self) -> dict:
        return self._delete("/observer/anomalies")

    # === Layer 5: Network ===

    def transfer_delta(
        self,
        source_session_id: str,
        target_session_id: str,
        max_weight: float = 0.15,
    ) -> dict:
        return self._post("/network/transfer", {
            "source_session_id": source_session_id,
            "target_session_id": target_session_id,
            "max_weight": max_weight,
        })

    def switch_user(self, user_id: str) -> dict:
        return self._post("/network/users/switch", {"user_id": user_id})

    def get_active_user(self) -> dict:
        return self._get("/network/users/active")

    def propose_consensus(
        self,
        proposer_session_id: str,
        target_embedding: list[float],
    ) -> dict:
        return self._post("/network/consensus", {
            "proposer_session_id": proposer_session_id,
            "target_embedding": target_embedding,
        })

    def get_trust_scores(self) -> dict:
        return self._get("/network/consensus/trust")
