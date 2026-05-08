"""Contract tests for the local SemvecClient.

Asserts that the wrapper around the bundled ``semvec`` package returns
the dictionary shapes the rest of the codebase depends on.

A 16-dimensional :class:`HashEmbedder` keeps the tests offline and fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.embedder import HashEmbedder
from src.core.semvec_client import SemvecClient


@pytest.fixture(autouse=True)
def clean_test_data():
    """Override the conftest autouse fixture — these tests need no Neo4j."""
    yield


@pytest.fixture
def client():
    return SemvecClient(embedder=HashEmbedder(dimension=16))


# ---------------------------------------------------------------------------
# Health


class TestHealth:
    def test_health_returns_ok(self, client):
        h = client.health()
        assert h["status"] == "ok"

    def test_health_includes_active_sessions(self, client):
        h = client.health()
        assert "active_sessions" in h
        assert isinstance(h["active_sessions"], int)


# ---------------------------------------------------------------------------
# Layer 1 — Sessions / Run / Store


class TestLayer1Run:
    def test_run_creates_session_when_none_given(self, client):
        r = client.run("Patient with chest pain")
        assert "session_id" in r
        assert isinstance(r["session_id"], str)

    def test_run_returns_required_keys(self, client):
        r = client.run("Heart failure follow-up")
        for key in (
            "session_id",
            "context",
            "top_similarity",
            "short_circuit",
            "drift_score",
            "drift_detected",
            "drift_phase",
        ):
            assert key in r, f"missing key {key!r} in run response"
        assert isinstance(r["context"], str)
        assert isinstance(r["top_similarity"], float)
        assert isinstance(r["drift_score"], float)
        assert isinstance(r["drift_detected"], bool)
        assert r["drift_phase"] in {"stable", "shifting", "drifted"}

    def test_run_reuses_session(self, client):
        first = client.run("Initial visit")
        sid = first["session_id"]
        second = client.run("Follow-up visit", session_id=sid)
        assert second["session_id"] == sid

    def test_run_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.run("Hi", session_id="does-not-exist")

    def test_run_reset_context_creates_session_with_id(self, client):
        r = client.run("first", session_id="custom-sid", reset_context=True)
        assert r["session_id"] == "custom-sid"

    def test_run_inline_store_response(self, client):
        first = client.run("Question one")
        sid = first["session_id"]
        # Pass response so it gets stored before the next turn
        second = client.run("Question two", session_id=sid, response="Answer one")
        assert second["session_id"] == sid

    def test_store_persists_response(self, client):
        first = client.run("Patient labs")
        sid = first["session_id"]
        result = client.store(sid, "Labs are within normal range.")
        assert result is not None


class TestLayer1bSession:
    def test_create_session_returns_session_id(self, client):
        result = client.create_session()
        assert "session_id" in result
        assert isinstance(result["session_id"], str)

    def test_export_session_round_trip(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        export = client.export_session(sid)
        assert "session_id" in export
        assert "state_dict" in export
        assert "checksum" in export

    def test_import_session_returns_dict(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        export = client.export_session(sid)
        # Create a target session and import into it
        target = client.create_session()
        result = client.import_session(target["session_id"], export["state_dict"])
        assert isinstance(result, dict)
        assert "session_id" in result

    def test_add_anchor_returns_dict(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        emb = list(HashEmbedder(dimension=16).get_embedding("oncology").astype(float))
        result = client.add_anchor(sid, emb)
        assert isinstance(result, dict)
        assert "anchor_count" in result or "anchor_index" in result

    def test_get_anchor_score_returns_float(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        emb = list(HashEmbedder(dimension=16).get_embedding("oncology").astype(float))
        client.add_anchor(sid, emb)
        score = client.get_anchor_score(sid)
        assert "anchor_score" in score
        assert isinstance(score["anchor_score"], float)

    def test_inject_memory(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        emb = list(HashEmbedder(dimension=16).get_embedding("interaction").astype(float))
        result = client.inject_memory(
            sid, embedding=emb, text="Metformin contraindicated with renal failure",
            tier="short_term", importance=0.9,
        )
        assert isinstance(result, dict)

    def test_set_isolation(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        result = client.set_isolation(sid, level="QUARANTINE", similarity_threshold=0.5)
        assert isinstance(result, dict)
        assert result.get("level") == "QUARANTINE"

    def test_add_trigger(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        result = client.add_trigger(sid, "CRITICAL")
        assert isinstance(result, dict)


class TestSessionLifecycle:
    """delete_session, reset_session — wraps SessionManager.delete_session/reset_session."""

    def test_delete_session_removes_it(self, client):
        first = client.run("init turn")
        sid = first["session_id"]
        result = client.delete_session(sid)
        assert isinstance(result, dict)
        assert result.get("deleted") is True
        # Subsequent run on a deleted session must surface as a clear error.
        with pytest.raises(ValueError):
            client.run("follow-up", session_id=sid)

    def test_delete_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.delete_session("does-not-exist")

    def test_delete_cluster_backed_session_is_refused(self, client):
        """Deleting a cluster's backing session would orphan the cluster."""
        cluster = client.create_cluster(name="ward-test")
        cid = cluster["cluster_id"]
        with pytest.raises(ValueError, match="cluster"):
            client.delete_session(cid)
        # Cluster still exists and is usable
        assert client.get_cluster(cid)["cluster_id"] == cid

    def test_reset_session_keeps_session_alive(self, client):
        # Build up state across several turns
        first = client.run("Patient one — diabetes")
        sid = first["session_id"]
        for msg in (
            "Patient one — Metformin",
            "Patient one — Lisinopril",
            "Patient one — Atorvastatin",
        ):
            client.run(msg, session_id=sid)
        before_metrics = client.get_metrics(sid) if hasattr(client, "get_metrics") else None
        result = client.reset_session(sid)
        assert isinstance(result, dict)
        assert result.get("reset") is True
        # Session still works and starts a fresh similarity history
        post = client.run("Patient one — fresh start", session_id=sid)
        assert post["session_id"] == sid
        # First turn after reset has top_similarity 0.0 because semantic_state is reset
        assert post["top_similarity"] == 0.0

    def test_reset_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.reset_session("does-not-exist")


class TestTriggerLifecycle:
    """clear_triggers, release_quarantine — inverse of add_trigger / set_isolation."""

    def test_clear_triggers_drops_all(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        client.add_trigger(sid, "CRITICAL")
        client.add_trigger(sid, "contraindication")

        result = client.clear_triggers(sid)
        assert isinstance(result, dict)
        assert result.get("trigger_count") == 0

    def test_clear_triggers_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.clear_triggers("does-not-exist")

    def test_release_quarantine_returns_count(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        client.set_isolation(sid, level="QUARANTINE", similarity_threshold=0.5)
        result = client.release_quarantine(sid)
        assert isinstance(result, dict)
        assert "released_count" in result
        assert isinstance(result["released_count"], int)

    def test_release_quarantine_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.release_quarantine("does-not-exist")


# ---------------------------------------------------------------------------
# Layer 2 — Cluster


class TestLayer2Cluster:
    def test_create_cluster(self, client):
        c = client.create_cluster(name="ward-A")
        assert "cluster_id" in c
        assert c["name"] == "ward-A"

    def test_list_clusters_includes_created(self, client):
        c = client.create_cluster(name="ward-B")
        all_clusters = client.list_clusters()
        ids = [x["cluster_id"] for x in all_clusters]
        assert c["cluster_id"] in ids

    def test_get_cluster(self, client):
        c = client.create_cluster(name="ward-C")
        got = client.get_cluster(c["cluster_id"])
        assert got["name"] == "ward-C"

    def test_delete_cluster(self, client):
        c = client.create_cluster(name="ephemeral")
        ok = client.delete_cluster(c["cluster_id"])
        assert ok.get("deleted") is True or ok.get("status") in {"deleted", "ok"}

    def test_cluster_run_returns_run_shape(self, client):
        c = client.create_cluster(name="ward-D")
        r = client.cluster_run(c["cluster_id"], "Ward round assessment")
        for key in ("session_id", "context", "top_similarity", "drift_score", "drift_detected", "drift_phase"):
            assert key in r

    def test_cluster_store(self, client):
        c = client.create_cluster(name="ward-E")
        r = client.cluster_store(c["cluster_id"], "Q?", "A!")
        assert isinstance(r, dict)

    def test_cluster_feedback(self, client):
        c = client.create_cluster(name="ward-F", coupling_factor=0.2)
        r = client.cluster_feedback(c["cluster_id"])
        assert "sessions_updated" in r

    def test_add_remove_cluster_member(self, client):
        c = client.create_cluster(name="ward-G")
        first = client.run("Hello")
        sid = first["session_id"]
        added = client.add_cluster_member(c["cluster_id"], sid)
        assert added.get("added") is True or added.get("status") in {"ok", "added"}
        removed = client.remove_cluster_member(c["cluster_id"], sid)
        assert removed.get("removed") is True or removed.get("status") in {"ok", "removed"}


# ---------------------------------------------------------------------------
# Layer 3 — Region


class TestLayer3Region:
    def test_create_region(self, client):
        r = client.create_region(name="hospital")
        assert "region_id" in r
        assert r["name"] == "hospital"

    def test_list_regions(self, client):
        client.create_region(name="hospital-x")
        regions = client.list_regions()
        names = [x["name"] for x in regions]
        assert "hospital-x" in names

    def test_add_region_cluster_then_events(self, client):
        c = client.create_cluster(name="ward-H")
        r = client.create_region(name="hospital-y")
        added = client.add_region_cluster(r["region_id"], c["cluster_id"])
        assert added.get("added") is True or added.get("status") in {"ok", "added"}

        # Cluster activity should not error and should give an event list
        client.cluster_run(c["cluster_id"], "Initial discussion")
        events = client.get_region_events(r["region_id"], limit=10)
        assert isinstance(events, list)


# ---------------------------------------------------------------------------
# Layer 4 — Observer


class TestLayer4Observer:
    def test_create_observer_then_summary(self, client):
        obs = client.create_observer(sample_interval_seconds=5.0)
        assert isinstance(obs, dict)
        summary = client.get_observer_summary()
        assert isinstance(summary, dict)

    def test_observer_sample(self, client):
        client.create_observer(sample_interval_seconds=5.0)
        sample = client.observer_sample()
        assert isinstance(sample, dict)

    def test_get_anomalies_returns_list(self, client):
        client.create_observer(sample_interval_seconds=5.0)
        anomalies = client.get_anomalies(limit=5)
        assert isinstance(anomalies, list)

    def test_clear_anomalies_returns_dict(self, client):
        client.create_observer(sample_interval_seconds=5.0)
        result = client.clear_anomalies()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Layer 5 — Network


class TestLayer5Network:
    def test_switch_user(self, client):
        result = client.switch_user("alice")
        assert isinstance(result, dict)

    def test_get_active_user(self, client):
        client.switch_user("bob")
        active = client.get_active_user()
        assert isinstance(active, dict)
        assert active.get("active_user") == "bob"

    def test_transfer_delta(self, client):
        a = client.run("source")
        b = client.run("target")
        result = client.transfer_delta(a["session_id"], b["session_id"], max_weight=0.1)
        assert isinstance(result, dict)

    def test_get_trust_scores(self, client):
        scores = client.get_trust_scores()
        assert isinstance(scores, dict)
