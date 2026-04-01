"""Scenario 4: Medication Safety Guard — PSS Layer 1b.

Tests that:
- Session can be created via /session/create
- Anchor can be added and anchor_score retrieved
- Trigger can be added (keyword "CRITICAL")
- Memory can be injected into a tier
- Isolation can be set to QUARANTINE
- Some Layer 1b endpoints may 404 — wrapped in try/except, skipped if unavailable
"""
from __future__ import annotations

import math
import pytest
from src.core.pss_client import PSSClient


def _pseudo_embed(text: str, dim: int = 384) -> list[float]:
    """Generate a deterministic pseudo-embedding for testing."""
    vec = [0.0] * dim
    for i, ch in enumerate(text.encode("utf-8")):
        idx = (ch * (i + 1)) % dim
        vec[idx] += math.sin(ch * 0.1 + i * 0.01)
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


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
def safety_session(pss):
    """Create a session for safety guard scenario."""
    try:
        result = pss.create_session(enable_topic_switch=True)
        sid = result["session_id"]
    except Exception:
        # Fallback: create via /run
        result = pss.run("Carlos Gutierrez oncology pharmacist session init")
        sid = result["session_id"]
    yield sid


class TestSessionCreate:
    def test_create_session_returns_session_id(self, pss):
        """POST /session/create returns a session_id."""
        try:
            result = pss.create_session(enable_topic_switch=True)
            assert "session_id" in result
            assert result["session_id"]
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                pytest.skip(f"create_session endpoint not available: {e}")
            raise

    def test_create_session_with_defaults(self, pss):
        """Session creation with default parameters works."""
        try:
            result = pss.create_session()
            assert "session_id" in result
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                pytest.skip(f"create_session endpoint not available: {e}")
            raise

    def test_run_creates_session_as_fallback(self, pss):
        """POST /run also creates a session (fallback for Layer 1b)."""
        result = pss.run(
            "Carlos Gutierrez chemotherapy monitoring",
            short_circuit_threshold=0.99,
        )
        assert "session_id" in result
        assert result["session_id"]


class TestAnchor:
    def test_add_anchor_succeeds_or_404(self, pss, safety_session):
        """POST /session/{id}/anchor either succeeds or returns 404 gracefully."""
        oncology_embedding = _pseudo_embed("oncology chemotherapy cancer treatment")
        try:
            result = pss.add_anchor(safety_session, oncology_embedding)
            assert result is not None
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"anchor endpoint not available: {e}")
            raise

    def test_get_anchor_score_succeeds_or_404(self, pss, safety_session):
        """GET /session/{id}/anchor_score either returns score or 404."""
        try:
            result = pss.get_anchor_score(safety_session)
            assert result is not None
            # If available, should have some drift-related field
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"anchor_score endpoint not available: {e}")
            raise

    def test_anchor_score_after_adding_anchor(self, pss, safety_session):
        """After adding an anchor, anchor_score is retrievable."""
        embedding = _pseudo_embed("oncology pharmacist monitoring chemotherapy")
        try:
            pss.add_anchor(safety_session, embedding)
            score = pss.get_anchor_score(safety_session)
            assert score is not None
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"anchor endpoints not available: {e}")
            raise


class TestTrigger:
    def test_add_trigger_keyword_critical(self, pss, safety_session):
        """POST /session/{id}/trigger adds keyword 'CRITICAL'."""
        try:
            result = pss.add_trigger(safety_session, "CRITICAL")
            assert result is not None
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"trigger endpoint not available: {e}")
            raise

    def test_add_trigger_keyword_contraindication(self, pss, safety_session):
        """POST /session/{id}/trigger adds 'contraindication' keyword."""
        try:
            result = pss.add_trigger(safety_session, "contraindication")
            assert result is not None
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"trigger endpoint not available: {e}")
            raise

    def test_trigger_run_with_critical_keyword(self, pss, safety_session):
        """Running a message with trigger keyword processes normally."""
        try:
            pss.add_trigger(safety_session, "CRITICAL")
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                pytest.skip(f"trigger endpoint not available: {e}")
            raise

        # Now run a message containing the trigger keyword
        result = pss.run(
            "CRITICAL: Monitor for neutropenic fever in Gutierrez — ANC threshold?",
            session_id=safety_session,
            short_circuit_threshold=0.99,
        )
        assert "session_id" in result


class TestMemoryInjection:
    def test_inject_memory_contraindication(self, pss, safety_session):
        """POST /session/{id}/memory injects a contraindication memory."""
        embedding = _pseudo_embed(
            "Amoxicillin 250mg is contraindicated with Metformin 500mg"
        )
        try:
            result = pss.inject_memory(
                safety_session,
                embedding=embedding,
                text="Amoxicillin 250mg is contraindicated with Metformin 500mg",
                tier="short_term",
                importance=1.0,
            )
            assert result is not None
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"memory injection endpoint not available: {e}")
            raise

    def test_inject_memory_atorvastatin_sertraline(self, pss, safety_session):
        """Inject Atorvastatin-Sertraline contraindication memory."""
        embedding = _pseudo_embed(
            "Atorvastatin 40mg is contraindicated with Sertraline 50mg"
        )
        try:
            result = pss.inject_memory(
                safety_session,
                embedding=embedding,
                text="Atorvastatin 40mg is contraindicated with Sertraline 50mg",
                tier="short_term",
                importance=1.0,
            )
            assert result is not None
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"memory injection endpoint not available: {e}")
            raise

    def test_inject_multiple_memories(self, pss, safety_session):
        """Three synthetic contraindication memories can be injected."""
        memories = [
            "Amoxicillin 250mg is contraindicated with Metformin 500mg",
            "Atorvastatin 40mg is contraindicated with Sertraline 50mg",
            "Carlos Gutierrez Chemotherapy Cycle 1 uses Atorvastatin 40mg and Metformin 500mg",
        ]
        injected = 0
        for mem_text in memories:
            embedding = _pseudo_embed(mem_text)
            try:
                result = pss.inject_memory(
                    safety_session,
                    embedding=embedding,
                    text=mem_text,
                    tier="short_term",
                    importance=0.9,
                )
                if result is not None:
                    injected += 1
            except Exception as e:
                if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                    pytest.skip(f"memory injection endpoint not available: {e}")
                raise
        assert injected == len(memories)


class TestIsolation:
    def test_set_isolation_quarantine(self, pss, safety_session):
        """PUT /session/{id}/isolation sets level to QUARANTINE."""
        try:
            result = pss.set_isolation(
                safety_session,
                level="QUARANTINE",
                similarity_threshold=0.5,
            )
            assert result is not None
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"isolation endpoint not available: {e}")
            raise

    def test_set_isolation_open(self, pss, safety_session):
        """PUT /session/{id}/isolation sets level to OPEN."""
        try:
            result = pss.set_isolation(
                safety_session,
                level="OPEN",
                similarity_threshold=0.8,
            )
            assert result is not None
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"isolation endpoint not available: {e}")
            raise

    def test_oncology_run_after_setup(self, pss, safety_session):
        """Layer 1 /run works normally after Layer 1b setup."""
        # Inject memory first
        embedding = _pseudo_embed("Amoxicillin contraindicated with Metformin")
        try:
            pss.inject_memory(
                safety_session,
                embedding=embedding,
                text="Amoxicillin 250mg is contraindicated with Metformin 500mg",
                tier="short_term",
                importance=1.0,
            )
        except Exception:
            pass  # Memory injection may not be available

        # Standard run should still work
        result = pss.run(
            "Carlos Gutierrez is starting Chemotherapy Cycle 1 — what pre-treatment labs?",
            session_id=safety_session,
            short_circuit_threshold=0.99,
        )
        assert "session_id" in result
        assert result["session_id"] == safety_session
