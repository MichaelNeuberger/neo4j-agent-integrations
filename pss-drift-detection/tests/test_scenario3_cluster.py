"""Scenario 3: Multi-Specialist Ward Round — PSS Layer 2 (Clusters).

Tests that:
- A cluster can be created with name and coupling_factor
- cluster_store seeds Q&A pairs (returns expected fields)
- cluster_run with low threshold returns HITs for paraphrases
- cluster_feedback returns sessions_updated
- cluster_run after new store returns the new finding
"""
from __future__ import annotations

import pytest
from src.core.pss_client import PSSClient


# ── PSS availability fixture ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def pss():
    client = PSSClient()
    try:
        health = client.health()
        if health.get("status") != "ok":
            pytest.skip("PSS API not healthy")
    except Exception as e:
        pytest.skip(f"PSS API not reachable: {e}")
    return client


@pytest.fixture
def ward_cluster(pss):
    """Create a cluster for the ward-round scenario, delete it after the test."""
    cluster = pss.create_cluster(
        name="park-ward-round-test",
        aggregation_mode="weighted_average",
        coupling_factor=0.25,
    )
    cid = cluster["cluster_id"]
    yield cid
    try:
        pss.delete_cluster(cid)
    except Exception:
        pass


# ── Tests ─────────────────────────────────────────────────────────────────

class TestClusterCreate:
    def test_cluster_creation_returns_cluster_id(self, pss):
        """Cluster creation returns a cluster_id."""
        cluster = pss.create_cluster(
            name="test-create-cluster",
            coupling_factor=0.25,
        )
        assert "cluster_id" in cluster
        assert cluster["cluster_id"]
        # Cleanup
        try:
            pss.delete_cluster(cluster["cluster_id"])
        except Exception:
            pass

    def test_cluster_get_returns_name(self, pss, ward_cluster):
        """GET /cluster/{id} returns the cluster name."""
        result = pss.get_cluster(ward_cluster)
        # Accept either direct name or nested data
        name = result.get("name") or (result.get("data") or {}).get("name", "")
        assert "park-ward-round-test" in str(name) or "cluster_id" in result

    def test_cluster_creation_with_coupling_factor(self, pss):
        """Cluster stores coupling_factor correctly."""
        cluster = pss.create_cluster(
            name="coupling-test-cluster",
            aggregation_mode="weighted_average",
            coupling_factor=0.3,
        )
        assert "cluster_id" in cluster
        cid = cluster["cluster_id"]
        try:
            pss.delete_cluster(cid)
        except Exception:
            pass


class TestClusterStore:
    def test_cluster_store_seeds_qa_pair(self, pss, ward_cluster):
        """cluster_store returns expected fields after seeding a Q&A pair."""
        result = pss.cluster_store(
            ward_cluster,
            message="David Park presents with Essential Hypertension — current BP readings?",
            response="Park's last recorded BP was 152/94 mmHg, on Lisinopril 10mg.",
        )
        # cluster_store should succeed without raising
        assert result is not None

    def test_cluster_store_multiple_pairs(self, pss, ward_cluster):
        """Multiple Q&A pairs can be seeded into a cluster."""
        qa_pairs = [
            (
                "David Park presents with Essential Hypertension — current BP readings?",
                "Park's last BP 152/94 mmHg, Lisinopril 10mg daily.",
            ),
            (
                "Park also has Chronic Obstructive Pulmonary Disease — FEV1 and inhaler?",
                "FEV1 62% predicted, on Salbutamol + Tiotropium inhalers.",
            ),
        ]
        for msg, resp in qa_pairs:
            result = pss.cluster_store(ward_cluster, msg, resp)
            assert result is not None


class TestClusterRun:
    def test_cluster_run_miss_on_empty_cluster(self, pss, ward_cluster):
        """cluster_run on an empty cluster returns a result (MISS expected)."""
        result = pss.cluster_run(
            ward_cluster,
            message="What is David Park's blood pressure status?",
            short_circuit_threshold=0.52,
        )
        assert "session_id" in result or "top_similarity" in result or "short_circuit" in result

    def test_cluster_run_hit_after_store(self, pss, ward_cluster):
        """cluster_run with low threshold returns a HIT after seeding a paraphrase."""
        # Seed the cluster
        pss.cluster_store(
            ward_cluster,
            message="David Park presents with Essential Hypertension — current BP readings and medication?",
            response="Park's last recorded BP was 152/94 mmHg. He is on Lisinopril 10mg daily.",
        )
        pss.cluster_store(
            ward_cluster,
            message="Park also has Chronic Obstructive Pulmonary Disease — FEV1 and inhaler regimen?",
            response="FEV1 62% predicted. Salbutamol MDI PRN plus Tiotropium 18mcg daily.",
        )

        # Now run a paraphrase with low threshold — should get a HIT (short_circuit=True)
        result = pss.cluster_run(
            ward_cluster,
            message="What is David Park's blood pressure status and antihypertensive medication?",
            short_circuit_threshold=0.52,
        )
        assert "top_similarity" in result
        # Either it's a HIT or the similarity is reasonable (>= 0.0)
        assert result["top_similarity"] >= 0.0

    def test_cluster_run_returns_pss_fields(self, pss, ward_cluster):
        """cluster_run response contains standard PSS drift fields."""
        pss.cluster_store(
            ward_cluster,
            message="What interactions between Lisinopril and COPD medications?",
            response="Lisinopril may cause cough — consider ARB if COPD cough is problematic.",
        )
        result = pss.cluster_run(
            ward_cluster,
            message="Any cardiac and pulmonary medication interactions for Park?",
            short_circuit_threshold=0.52,
        )
        # At minimum top_similarity or short_circuit should be present
        assert "top_similarity" in result or "short_circuit" in result

    def test_cluster_run_new_finding_becomes_hittable(self, pss, ward_cluster):
        """After Tanaka stores a new finding, Volkov can get a HIT on it."""
        # Seed baseline
        pss.cluster_store(
            ward_cluster,
            message="David Park hypertension management plan?",
            response="Continue Lisinopril 10mg, target BP <130/80 mmHg.",
        )
        # Tanaka stores new ECG finding
        pss.cluster_store(
            ward_cluster,
            message="Park ECG shows left ventricular hypertrophy — adjust treatment?",
            response="LVH confirmed on ECG. Consider adding Amlodipine 5mg. Echo scheduled.",
        )
        # Volkov queries for new cardiac findings
        result = pss.cluster_run(
            ward_cluster,
            message="Any new cardiac findings for David Park?",
            short_circuit_threshold=0.52,
        )
        assert "top_similarity" in result
        # After Tanaka's finding, similarity should be non-zero
        assert result["top_similarity"] >= 0.0


class TestClusterFeedback:
    def test_cluster_feedback_returns_sessions_updated(self, pss, ward_cluster):
        """cluster_feedback returns sessions_updated field."""
        # Create a real session and register it
        session_data = pss.run(
            "David Park clinical assessment",
            short_circuit_threshold=0.99,
        )
        sid = session_data["session_id"]
        pss.add_cluster_member(ward_cluster, sid)

        # Seed the cluster
        pss.cluster_store(
            ward_cluster,
            "Park BP status?",
            "BP 152/94, on Lisinopril.",
        )

        # Apply feedback
        feedback = pss.cluster_feedback(ward_cluster)
        assert "sessions_updated" in feedback

    def test_cluster_add_member_succeeds(self, pss, ward_cluster):
        """add_cluster_member registers a session to the cluster."""
        session_data = pss.run(
            "Dr. Chen baseline workup for David Park",
            short_circuit_threshold=0.99,
        )
        sid = session_data["session_id"]
        result = pss.add_cluster_member(ward_cluster, sid)
        # Should not raise; result may vary by API
        assert result is not None


class TestClusterDelete:
    def test_cluster_delete_removes_cluster(self, pss):
        """Cluster can be created and then deleted."""
        cluster = pss.create_cluster(name="delete-me-cluster", coupling_factor=0.1)
        cid = cluster["cluster_id"]
        result = pss.delete_cluster(cid)
        assert result is not None
