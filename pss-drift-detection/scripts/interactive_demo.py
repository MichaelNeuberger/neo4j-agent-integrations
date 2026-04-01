#!/usr/bin/env python3
"""Interactive PSS + Neo4j Demo — shows the interplay between both systems.

Scenarios:
  1. Live Drift Detection    — Type messages, watch drift score + Neo4j state chain grow
  2. Topic Switch Detection  — Guided scenario: stay on topic, then switch, observe drift
  3. Multi-Agent Comparison  — Run parallel agents, compare trajectories in Neo4j
  4. Memory & Recall         — Store memories, query by similarity, consolidate tiers
  5. Cross-Session Analytics  — Influence scoring, similarity matrix, drift points
  6. Session Export/Transfer  — PSS Layer 5: transfer knowledge between agents
  7. Neo4j Graph Explorer     — Run Cypher queries directly, see the graph grow

Usage:
    python3 scripts/interactive_demo.py              # fake responses (fast)
    python3 scripts/interactive_demo.py --llm        # real LLM responses
"""

from __future__ import annotations

import json
import math
import os
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from src.core.pss_client import PSSClient
from src.core.drift_detector import DriftDetector
from src.mcp.pss_mcp_server import PSSMCPServer
from src.persistence.adapter import Neo4jPSSAdapter
from src.persistence.models import Cluster, AggregationStrategy, MemoryTier
from src.persistence.neo4j_cluster_store import Neo4jClusterStore
from src.persistence.neo4j_state_store import Neo4jStateStore
from src.persistence.neo4j_phase_store import Neo4jPhaseStore
from src.persistence.neo4j_drift_event_store import Neo4jDriftEventStore
from src.persistence.neo4j_memory_store import Neo4jMemoryStore
from src.analytics.similarity import SimilarityAnalyzer
from src.analytics.influence import InfluenceAnalyzer
from src.analytics.trajectories import TrajectoryAnalyzer

# ── Config ──────────────────────────────────────────────────────────────

NEO4J_URI = os.environ.get("NEO4J_TEST_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_TEST_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_TEST_PASSWORD", "testpassword")
NEO4J_DATABASE = os.environ.get("NEO4J_TEST_DATABASE", "neo4j")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "schema", "pss_schema.cypher")

USE_LLM = "--llm" in sys.argv

# ── LLM Client ──────────────────────────────────────────────────────────

class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat endpoint."""

    def __init__(self):
        import openai
        self._client = openai.OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", ""),
            api_key=os.environ.get("OPENAI_API_KEY", "not-set"),
        )
        self._model = os.environ.get("OPENAI_MODEL", "openai/gpt-oss-120b")

    def respond(
        self,
        user_message: str,
        system_prompt: str = "",
        context: str = "",
        max_tokens: int = 400,
    ) -> str:
        """Generate a response.  Returns content (or reasoning_content as fallback)."""
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context:
            messages.append({
                "role": "system",
                "content": f"Conversation context so far:\n{context}",
            })
        messages.append({"role": "user", "content": user_message})

        try:
            r = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
            )
            msg = r.choices[0].message
            return (
                msg.content
                or getattr(msg, "reasoning_content", None)
                or "(empty response)"
            )
        except Exception as e:
            return f"(LLM error: {e})"


_llm: LLMClient | None = None

def get_llm() -> LLMClient:
    """Lazy-init singleton."""
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def generate_response(
    user_message: str,
    agent_role: str = "a helpful medical research assistant",
    pss_context: str = "",
) -> str:
    """Generate an agent response — real LLM or placeholder."""
    if not USE_LLM:
        return f"Acknowledged: {user_message}"
    llm = get_llm()
    system = (
        f"You are {agent_role}. "
        "Answer concisely in 2-3 sentences. Stay factual."
    )
    return llm.respond(user_message, system_prompt=system, context=pss_context)

# ── ANSI Colors ─────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
WHITE = "\033[97m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"

def color_drift(score: float) -> str:
    if score < 0.1:
        return f"{GREEN}{score:.3f}{RESET}"
    elif score < 0.3:
        return f"{YELLOW}{score:.3f}{RESET}"
    elif score < 0.5:
        return f"{RED}{score:.3f}{RESET}"
    return f"{BG_RED}{WHITE}{score:.3f}{RESET}"

def color_phase(phase: str) -> str:
    colors = {
        "initialization": DIM,
        "exploration": BLUE,
        "convergence": CYAN,
        "resonance": GREEN,
        "stability": f"{BOLD}{GREEN}",
        "instability": f"{BOLD}{RED}",
    }
    return f"{colors.get(phase, '')}{phase}{RESET}"

def bar(value: float, width: int = 30, fill="█", empty="░") -> str:
    filled = int(value * width)
    return f"{fill * filled}{empty * (width - filled)}"

def header(text: str):
    w = 70
    print(f"\n{CYAN}{'═' * w}{RESET}")
    print(f"{CYAN}  {BOLD}{text}{RESET}")
    print(f"{CYAN}{'═' * w}{RESET}\n")

def subheader(text: str):
    print(f"\n{YELLOW}── {text} {'─' * max(0, 60 - len(text))}{RESET}\n")

def info(label: str, value: str):
    print(f"  {DIM}{label:>22}{RESET}  {value}")

def _pseudo_embed(text: str, dim: int = 384) -> list[float]:
    vec = [0.0] * dim
    for i, ch in enumerate(text.encode("utf-8")):
        idx = (ch * (i + 1)) % dim
        vec[idx] += math.sin(ch * 0.1 + i * 0.01)
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


# ── Scenario 1: Live Drift Detection ────────────────────────────────────

def scenario_live_drift(mcp: PSSMCPServer, driver):
    header("Scenario 1: Live Drift Detection")
    print("  Type messages and watch PSS compute drift + Neo4j persist the state chain.")
    print(f"  Type {BOLD}'quit'{RESET} to return to menu.\n")

    session = mcp.create_pss_session(agent_id=f"interactive-{int(time.time())}")
    sid = session["session_id"]
    info("Session", sid)
    info("Agent", session["agent_id"])
    info("LLM", f"{GREEN}enabled{RESET} ({os.environ.get('OPENAI_MODEL', 'N/A')})" if USE_LLM else f"{DIM}off (use --llm){RESET}")
    print()

    step = 0
    while True:
        try:
            msg = input(f"  {CYAN}[step {step}]{RESET} You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg or msg.lower() == "quit":
            break

        t0 = time.time()
        result = mcp.detect_drift(sid, msg)
        pss_latency = (time.time() - t0) * 1000

        sim = result["top_similarity"]
        drift_score = result["drift_score"]
        phase = result["drift_phase"]
        detected = result.get("drift_detected", False)

        # Generate agent response (real LLM or placeholder)
        t1 = time.time()
        response = generate_response(msg, pss_context=result.get("context", ""))
        llm_latency = (time.time() - t1) * 1000

        # Teach PSS the response
        mcp.store_response(sid, response)

        print()
        if USE_LLM:
            print(f"  {GREEN}Agent:{RESET} {textwrap.fill(response, 72, subsequent_indent='         ')}")
            print()
        info("Similarity", f"{sim:.3f}  {bar(sim)}")
        info("Drift Score", f"{drift_score:.3f}")
        info("Phase", color_phase(phase))
        info("Drift Detected", f"{RED}YES{RESET}" if detected else f"{GREEN}no{RESET}")
        info("Context (PSS)", textwrap.shorten(result["context"], 80) if result["context"] else "(empty)")
        info("Neo4j State", f"#{result['step']}  (id: {result['state_id'][:12]}...)")
        info("Latency", f"PSS {pss_latency:.0f}ms" + (f" + LLM {llm_latency:.0f}ms" if USE_LLM else ""))

        # Show Neo4j state chain length
        trajectory = mcp.get_state_trajectory(sid, steps=50)
        current_phase = mcp.get_phase(sid)
        info("Chain Length", f"{len(trajectory)} states in Neo4j")
        info("Neo4j Phase", color_phase(current_phase.get("phase", "N/A")))
        print()
        step += 1

    # End session summary
    summary = mcp.end_pss_session(sid)
    subheader("Session Summary")
    info("Final Phase", summary.get("final_phase", "N/A"))
    info("Total States", str(summary.get("total_states", 0)))
    info("Drift Events", str(summary.get("total_drift_events", 0)))


# ── Scenario 2: Topic Switch Detection ──────────────────────────────────

def scenario_topic_switch(mcp: PSSMCPServer, driver):
    header("Scenario 2: Drift Detection + Short-Circuit")
    print("  Four phases demonstrating drift AND short-circuit in one narrative:")
    print(f"    {GREEN}Phase 1{RESET}  Morrison diabetes workup — deep clinical context")
    print(f"    {RED}Phase 2{RESET}  Patel psychiatry (MDD/Sertraline/CBT) — different specialty")
    print(f"    {CYAN}Phase 3{RESET}  Rodriguez cardiology — return to clinical domain")
    print(f"    {MAGENTA}Phase 4{RESET}  Paraphrased Phase 1 queries — short-circuit, LLM skipped")
    print(f"  {BOLD}30 steps total.{RESET}\n")

    pss = PSSClient()

    # We use the raw PSS client so we can leverage the inline-store pattern
    # (response= on the next /run call) and read short_circuit directly.
    # Neo4j mirroring happens via MCP in parallel.
    neo4j_session = mcp.create_pss_session(agent_id="drift-shortcircuit-demo")
    neo4j_sid = neo4j_session["session_id"]

    # Phase 1: 17 clinical investigation queries — deep diabetes patient workup
    # (References: Patient James Morrison, Type 2 Diabetes + COPD + HTN,
    #  treated by Dr. Volkov and Dr. Tanaka, Metformin/Lisinopril/Atorvastatin)
    on_topic_1 = [
        "James Morrison presents with Type 2 Diabetes Mellitus — what is his current treatment protocol?",
        "He is on Metformin 500mg — are there contraindications given his COPD diagnosis?",
        "Metformin is contraindicated with renal impairment GFR below 30 — check his lab results",
        "Dr. Elena Volkov and Dr. Yuki Tanaka both treat Morrison — who should lead the diabetes management?",
        "Lisinopril 10mg is also indicated for his Essential Hypertension — is the combination safe?",
        "What Atorvastatin 40mg interactions should we monitor alongside Metformin?",
        "His encounter at Emergency Room Visit showed chest pain — should we adjust his diabetes meds?",
        "Review the Cardiac Catheterization treatment — does it use Metformin or Atorvastatin?",
        "What is the recommended insulin regimen if Metformin fails to achieve target A1C?",
        "Are there drug interactions between Metformin and the contrast dye used in catheterization?",
        "What dietary recommendations should Morrison follow alongside his diabetes medications?",
        "How does continuous glucose monitoring improve glycemic control in Type 2 Diabetes?",
        "What role does bariatric surgery play in diabetes remission for patients like Morrison?",
        "How do we manage diabetic nephropathy progression in stage 3 CKD?",
        "What are the cardiovascular risk factors we should monitor given his comorbidities?",
        "Should we consider adding an SGLT2 inhibitor like Empagliflozin to Morrisons regimen?",
        "What is the long-term prognosis for Morrison with optimal diabetes management?",
    ]
    # Phase 2: Different patient, different specialty — psychiatry
    # (References: Aisha Patel, Major Depressive Disorder, Sertraline 50mg, CBT)
    # Realistic: same clinician sees next patient in a different domain.
    off_topic = [
        "Aisha Patel presents with Major Depressive Disorder — what is her current treatment plan?",
        "She is on Sertraline 50mg — what are the common side effects and monitoring requirements?",
        "Is Cognitive Behavioral Therapy recommended alongside Sertraline for treatment-resistant depression?",
        "What PHQ-9 score threshold should trigger a medication adjustment for Patel?",
        "Are there contraindications between Sertraline and any medications Patel might be taking?",
    ]
    # Phase 3: Return to clinical — different patient, cardiology focus
    # (References: Maria Rodriguez, AMI + Hypertension, Dr. O'Brien → Dr. Okonkwo)
    on_topic_2 = [
        "Maria Rodriguez was diagnosed with Acute Myocardial Infarction — what is her treatment plan?",
        "Dr. O'Brien referred her to Dr. Okonkwo — is the referral for cardiology follow-up?",
        "Her encounter at Emergency Room Visit resulted in which diagnoses?",
    ]
    # Phase 4: Paraphrased Phase 1 queries — PSS should short-circuit (context from Phase 1)
    short_circuit_queries = [
        ("What treatment protocol does James Morrison follow for his diabetes?",           "≈ Phase 1 Q1"),
        ("Are there Metformin contraindications for a patient with COPD?",                 "≈ Phase 1 Q2"),
        ("Check renal function thresholds for Metformin prescribing",                      "≈ Phase 1 Q3"),
        ("Which providers are responsible for Morrison's diabetes care?",                   "≈ Phase 1 Q4"),
        ("Is Lisinopril safe to combine with Metformin for hypertension and diabetes?",    "≈ Phase 1 Q5"),
    ]

    all_msgs = on_topic_1 + off_topic + on_topic_2
    all_labels = (
        ["ON-TOPIC"] * len(on_topic_1)
        + ["OFF-TOPIC"] * len(off_topic)
        + ["NEW-TOPIC"] * len(on_topic_2)
    )
    all_roles = (
        ["a clinical pharmacist reviewing patient James Morrison's diabetes treatment"] * len(on_topic_1)
        + ["a psychiatrist managing Aisha Patel's depression treatment"] * len(off_topic)
        + ["a cardiologist reviewing Maria Rodriguez's post-MI care"] * len(on_topic_2)
    )

    # Legend
    subheader("How to read the output")
    print(f"  Each step shows the {BOLD}PSS similarity{RESET} — how well the query matches accumulated context.")
    print(f"  All signals come directly from the PSS API, no client-side heuristics.\n")
    print(f"  {GREEN}{'█' * 5}{RESET} sim > 0.4 (on-topic)     "
          f"{YELLOW}{'█' * 5}{RESET} sim 0.2–0.4 (shifting)     "
          f"{RED}{'█' * 5}{RESET} sim < 0.2 (off-topic)")
    print(f"  {BOLD}DRIFT{RESET} = PSS drift_detected=True  (drift_score shown)")
    print(f"  {DIM}phase{RESET} = PSS drift_phase: stable / shifting / drifted")
    print()

    # Map queries to healthcare entities they reference (for INVESTIGATED relationships)
    _entity_refs: list[list[tuple[str, str]]] = [
        # Phase 1: Morrison diabetes workup (17 queries)
        [("Patient", "James Morrison"), ("Diagnosis", "Type 2 Diabetes Mellitus")],
        [("Medication", "Metformin 500mg"), ("Diagnosis", "Chronic Obstructive Pulmonary Disease")],
        [("Medication", "Metformin 500mg")],
        [("Provider", "Dr. Elena Volkov"), ("Provider", "Dr. Yuki Tanaka")],
        [("Medication", "Lisinopril 10mg"), ("Diagnosis", "Essential Hypertension")],
        [("Medication", "Atorvastatin 40mg"), ("Medication", "Metformin 500mg")],
        [("Encounter", "Emergency Room Visit"), ("Patient", "James Morrison")],
        [("Treatment", "Cardiac Catheterization"), ("Medication", "Metformin 500mg")],
        [("Medication", "Metformin 500mg")],          # insulin regimen
        [("Medication", "Metformin 500mg"), ("Treatment", "Cardiac Catheterization")],  # contrast
        [("Patient", "James Morrison")],               # dietary
        [("Patient", "James Morrison")],               # CGM
        [("Patient", "James Morrison")],               # bariatric
        [("Patient", "James Morrison"), ("Medication", "Lisinopril 10mg")],  # nephropathy
        [("Patient", "James Morrison"), ("Medication", "Atorvastatin 40mg")],  # CV risk
        [("Patient", "James Morrison")],               # SGLT2i
        [("Patient", "James Morrison")],               # prognosis
        # Phase 2: Psychiatry — Aisha Patel
        [("Patient", "Aisha Patel"), ("Diagnosis", "Major Depressive Disorder")],
        [("Medication", "Sertraline 50mg")],
        [("Treatment", "Cognitive Behavioral Therapy")],
        [("Patient", "Aisha Patel")],
        [("Medication", "Sertraline 50mg")],
        # Phase 3: Rodriguez cardiology
        [("Patient", "Maria Rodriguez"), ("Diagnosis", "Acute Myocardial Infarction")],
        [("Provider", "Dr. Michael O'Brien"), ("Provider", "Dr. Rachel Okonkwo")],
        [("Encounter", "Emergency Room Visit")],
    ]

    pss_sid = None
    prev_response = None
    sc_threshold = 0.65
    drift_events_count = 0
    BAR_W = 20  # bar width

    def _sim_bar(sim: float) -> str:
        """Similarity bar — green=on-topic, yellow=shifting, red=drifted."""
        filled = int(min(sim, 1.0) * BAR_W)
        if sim > 0.4:
            return f"{GREEN}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"
        elif sim > 0.2:
            return f"{YELLOW}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"
        else:
            return f"{RED}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"

    prev_label = None
    for i, (msg, label, role) in enumerate(zip(all_msgs, all_labels, all_roles)):
        # Phase separator
        if label != prev_label:
            phase_names = {"ON-TOPIC": "Phase 1: Morrison Diabetes", "OFF-TOPIC": "Phase 2: Patel Psychiatry", "NEW-TOPIC": "Phase 3: Rodriguez Cardiology"}
            phase_colors = {"ON-TOPIC": GREEN, "OFF-TOPIC": RED, "NEW-TOPIC": CYAN}
            c = phase_colors.get(label, "")
            print(f"\n  {c}{'─' * 60}")
            print(f"  {BOLD}{phase_names.get(label, label)}{RESET}")
            print(f"  {c}{'─' * 60}{RESET}\n")
            prev_label = label

        # Mirror to Neo4j via MCP (drift detection + persistence)
        result = mcp.detect_drift(neo4j_sid, msg)

        # Also run on raw PSS session for short-circuit tracking (inline-store)
        pss_data = pss.run(msg, session_id=pss_sid, response=prev_response,
                           short_circuit_threshold=sc_threshold)
        pss_sid = pss_data["session_id"]

        response = generate_response(msg, agent_role=role, pss_context=result.get("context", ""))
        mcp.store_response(neo4j_sid, response)
        prev_response = response

        # Create INVESTIGATED relationships: Agent → Healthcare entities
        if i < len(_entity_refs):
            for entity_label, entity_name in _entity_refs[i]:
                with driver.session(database=NEO4J_DATABASE) as db:
                    db.run(f"""
                        MATCH (s:AgentSession {{session_id: $sid}})
                        MATCH (e:{entity_label} {{name: $name}})
                        MERGE (s)-[:INVESTIGATED {{step: $step, drift_score: $drift,
                               phase: $label}}]->(e)
                    """, sid=neo4j_sid, name=entity_name, step=i,
                    drift=result["drift_score"], label=label).consume()

        sim = result["top_similarity"]
        drift_score = result["drift_score"]
        phase = result.get("drift_phase", "stable")
        detected = result.get("drift_detected", False)
        if detected:
            drift_events_count += 1

        # Phase color
        phase_c = {
            "stable": GREEN, "shifting": YELLOW, "drifted": RED,
        }.get(phase, DIM)

        # Drift flag
        drift_flag = f"  {RED}{BOLD}DRIFT{RESET}" if detected else ""

        # Main output: step, bar, sim, drift_score, phase, message
        print(f"  {i:>2}  {_sim_bar(sim)} sim={sim:.2f}  drift={drift_score:.2f}  "
              f"{phase_c}{phase:>8}{RESET}  {textwrap.shorten(msg, 40)}{drift_flag}")
        if USE_LLM:
            print(f"      {DIM}A: {textwrap.shorten(response, 68)}{RESET}")

    # ── Phase 4: Short-circuit ──
    n_phases123 = len(all_msgs)
    print(f"\n  {MAGENTA}{'─' * 60}")
    print(f"  {BOLD}Phase 4: Short-Circuit — paraphrased Phase 1 queries{RESET}")
    print(f"  {MAGENTA}{'─' * 60}{RESET}")
    print(f"\n  PSS remembers the 8 diabetes Q&A pairs from Phase 1.")
    print(f"  Paraphrases with sim >= {sc_threshold} → {GREEN}HIT{RESET} (LLM skipped, cached context returned)\n")

    sc_hits = 0
    for j, (q, label) in enumerate(short_circuit_queries):
        step = n_phases123 + j
        pss_data = pss.run(q, session_id=pss_sid, response=prev_response,
                           short_circuit_threshold=sc_threshold)
        sim = pss_data["top_similarity"]
        sc = pss_data["short_circuit"]
        mcp.detect_drift(neo4j_sid, q)

        if sc:
            sc_hits += 1
            sim_bar = f"{GREEN}{'█' * int(sim * BAR_W)}{'░' * (BAR_W - int(sim * BAR_W))}{RESET}"
            print(f"  {step:>2}  {sim_bar} sim={sim:.2f}  {GREEN}HIT{RESET}  LLM skipped  {DIM}{label}{RESET}")
            prev_response = None
        else:
            sim_bar = f"{RED}{'█' * int(sim * BAR_W)}{'░' * (BAR_W - int(sim * BAR_W))}{RESET}"
            response = generate_response(q, agent_role="a clinical pharmacist reviewing diabetes treatment",
                                         pss_context=pss_data.get("context", ""))
            mcp.store_response(neo4j_sid, response)
            prev_response = response
            print(f"  {step:>2}  {sim_bar} sim={sim:.2f}  {RED}MISS{RESET} LLM called   {DIM}{label}{RESET}")

    # ── Summary ──
    subheader("Summary")
    total_steps = n_phases123 + len(short_circuit_queries)
    n1, n2, n3 = len(on_topic_1), len(off_topic), len(on_topic_2)

    print(f"  Steps:  {total_steps} total  ({n1} diabetes + {n2} psychiatry + {n3} cardiology + {len(short_circuit_queries)} paraphrase)")
    print(f"  Drift:  {drift_events_count} events stored in Neo4j")
    print(f"  HITs:   {sc_hits}/{len(short_circuit_queries)} paraphrases recognized  (threshold: sim >= {sc_threshold})")
    print(f"  LLM:    {total_steps - sc_hits} calls made, {sc_hits} skipped\n")

    print(f"  {BOLD}All signals from PSS API:{RESET}")
    print(f"    drift_detected   PSS flags topic switch (drift_score >= threshold)")
    print(f"    drift_score      semantic drift intensity (0.0–1.0)")
    print(f"    top_similarity   cosine match between query and accumulated context")
    print(f"    short_circuit    paraphrase recognized, LLM skippable")

    phase_info = mcp.get_phase(neo4j_sid)
    print(f"\n  Final phase: {color_phase(phase_info.get('phase', 'N/A') or 'N/A')}")

    mcp.end_pss_session(neo4j_sid)


# ── Scenario 3: Multi-Specialist Ward Round (PSS Layer 2 — Clusters) ─────

def scenario_ward_round(mcp: PSSMCPServer, driver):
    header("Scenario 3: Multi-Specialist Ward Round (PSS Layer 2 — Clusters)")
    print("  Three specialists examine David Park (Essential Hypertension + COPD).")
    print("  They share a PSS cluster so findings propagate between agents.")
    print(f"  {GREEN}Dr. Chen{RESET} (internist) seeds baseline findings.")
    print(f"  {CYAN}Dr. Volkov{RESET} (pulmonologist) queries — gets HITs from Chen's work.")
    print(f"  {MAGENTA}Dr. Tanaka{RESET} (cardiologist) adds ECG finding — Volkov re-queries.\n")

    pss = PSSClient()

    BAR_W = 20

    def _sim_bar(sim: float) -> str:
        filled = int(min(sim, 1.0) * BAR_W)
        if sim > 0.4:
            return f"{GREEN}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"
        elif sim > 0.2:
            return f"{YELLOW}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"
        else:
            return f"{RED}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"

    # ── Step 1: Create cluster ──
    subheader("Step 1: Create PSS cluster 'park-ward-round'")
    try:
        cluster = pss.create_cluster(
            name="park-ward-round",
            aggregation_mode="weighted_average",
            coupling_factor=0.25,
        )
        cid = cluster["cluster_id"]
        info("Cluster ID", cid[:20] + "...")
        info("Coupling factor", "0.25")
    except Exception as e:
        print(f"  {RED}Cluster creation failed: {e}{RESET}")
        return

    # ── Step 2: Dr. Chen seeds baseline (threshold=0.99 → always MISS) ──
    subheader("Step 2: Dr. Chen seeds baseline — 4 Q&A pairs about David Park")
    info("Role", "Dr. Sarah Chen — internist performing baseline workup for David Park")
    print()

    chen_role = "an internist performing baseline workup for David Park"
    chen_queries = [
        "David Park presents with Essential Hypertension — current BP readings and medication?",
        "Park also has Chronic Obstructive Pulmonary Disease — what is his FEV1 and current inhaler regimen?",
        "Are there contraindications between Lisinopril for hypertension and his COPD medications?",
        "What is Park's encounter history — when was his last Annual Physical Exam?",
    ]
    chen_entity_refs = [
        [("Patient", "David Park"), ("Diagnosis", "Essential Hypertension"), ("Medication", "Lisinopril 10mg")],
        [("Diagnosis", "Chronic Obstructive Pulmonary Disease"), ("Patient", "David Park")],
        [("Medication", "Lisinopril 10mg"), ("Diagnosis", "Chronic Obstructive Pulmonary Disease")],
        [("Patient", "David Park"), ("Encounter", "Annual Physical Exam")],
    ]

    chen_sid = None
    for i, msg in enumerate(chen_queries):
        cdata = pss.cluster_run(cid, msg, short_circuit_threshold=0.99)
        response = generate_response(msg, agent_role=chen_role, pss_context=cdata.get("context", ""))
        pss.cluster_store(cid, msg, response)
        chen_data = pss.run(msg, session_id=chen_sid, short_circuit_threshold=0.99)
        chen_sid = chen_data["session_id"]
        sim = cdata.get("top_similarity", 0.0)
        sc = cdata.get("short_circuit", False)
        print(f"  {RED}MISS{RESET}  {_sim_bar(sim)} sim={sim:.3f}  drift={cdata.get('drift_score', 0.0):.3f}  {DIM}{textwrap.shorten(msg, 50)}{RESET}")
        if USE_LLM:
            print(f"        {DIM}A: {textwrap.shorten(response, 68)}{RESET}")
        # INVESTIGATED: Chen → healthcare entities
        for entity_label, entity_name in chen_entity_refs[i]:
            with driver.session(database=NEO4J_DATABASE) as db:
                db.run(f"""
                    MATCH (s:AgentSession {{session_id: $sid}})
                    MATCH (e:{entity_label} {{name: $name}})
                    MERGE (s)-[:INVESTIGATED {{step: $step, phase: 'chen-baseline',
                           drift_score: $drift}}]->(e)
                """, sid=chen_sid, name=entity_name, step=i,
                drift=cdata.get("drift_score", 0.0)).consume()

    pss.add_cluster_member(cid, chen_sid)
    info("\n  Chen session", f"{chen_sid[:16]}... added to cluster")

    # ── Step 3: Dr. Volkov queries (threshold=0.52) ──
    subheader("Step 3: Dr. Volkov queries (threshold=0.52 — expect HITs from Chen)")
    info("Role", "Dr. Elena Volkov — pulmonologist reviewing David Park's respiratory status")
    print()

    volkov_role = "a pulmonologist reviewing David Park's respiratory status"
    volkov_queries = [
        "What is David Park's blood pressure status and antihypertensive medication?",
        "Does Park have any pulmonary comorbidities affecting his treatment plan?",
        "What interactions should we monitor between his cardiac and pulmonary medications?",
    ]
    volkov_entity_refs = [
        [("Patient", "David Park"), ("Diagnosis", "Essential Hypertension")],
        [("Diagnosis", "Chronic Obstructive Pulmonary Disease"), ("Patient", "David Park")],
        [("Medication", "Lisinopril 10mg"), ("Diagnosis", "Chronic Obstructive Pulmonary Disease")],
    ]

    volkov_sid = None
    volkov_hits = 0
    for i, msg in enumerate(volkov_queries):
        cdata = pss.cluster_run(cid, msg, short_circuit_threshold=0.52)
        volkov_data = pss.run(msg, session_id=volkov_sid, short_circuit_threshold=0.99)
        volkov_sid = volkov_data["session_id"]
        sim = cdata.get("top_similarity", 0.0)
        sc = cdata.get("short_circuit", False)
        if sc:
            volkov_hits += 1
            print(f"  {GREEN}HIT{RESET}   {_sim_bar(sim)} sim={sim:.3f}  drift={cdata.get('drift_score',0.0):.3f}  sc={sc}  {DIM}{textwrap.shorten(msg, 46)}{RESET}")
        else:
            response = generate_response(msg, agent_role=volkov_role, pss_context=cdata.get("context", ""))
            print(f"  {RED}MISS{RESET}  {_sim_bar(sim)} sim={sim:.3f}  drift={cdata.get('drift_score',0.0):.3f}  sc={sc}  {DIM}{textwrap.shorten(msg, 46)}{RESET}")
            if USE_LLM:
                print(f"        {DIM}A: {textwrap.shorten(response, 68)}{RESET}")
        for entity_label, entity_name in volkov_entity_refs[i]:
            with driver.session(database=NEO4J_DATABASE) as db:
                db.run(f"""
                    MATCH (s:AgentSession {{session_id: $sid}})
                    MATCH (e:{entity_label} {{name: $name}})
                    MERGE (s)-[:INVESTIGATED {{step: $step, phase: 'volkov-queries',
                           drift_score: $drift}}]->(e)
                """, sid=volkov_sid, name=entity_name, step=i,
                drift=cdata.get("drift_score", 0.0)).consume()

    # ── Step 4: Dr. Tanaka — novel ECG finding ──
    subheader("Step 4: Dr. Tanaka — queries + stores new ECG finding")
    info("Role", "Dr. Yuki Tanaka — cardiologist evaluating David Park's cardiac function")
    print()

    tanaka_role = "a cardiologist evaluating David Park's cardiac function"
    tanaka_q1 = "What is the hypertension management plan for David Park?"
    cdata_t1 = pss.cluster_run(cid, tanaka_q1, short_circuit_threshold=0.52)
    sim_t1 = cdata_t1.get("top_similarity", 0.0)
    sc_t1 = cdata_t1.get("short_circuit", False)
    tanaka_data = pss.run(tanaka_q1, short_circuit_threshold=0.99)
    tanaka_sid = tanaka_data["session_id"]

    if sc_t1:
        print(f"  {GREEN}HIT{RESET}   {_sim_bar(sim_t1)} sim={sim_t1:.3f}  drift={cdata_t1.get('drift_score',0.0):.3f}  (≈ Chen Q1)")
    else:
        resp_t1 = generate_response(tanaka_q1, agent_role=tanaka_role)
        print(f"  {RED}MISS{RESET}  {_sim_bar(sim_t1)} sim={sim_t1:.3f}  drift={cdata_t1.get('drift_score',0.0):.3f}  (≈ Chen Q1)")

    # Tanaka stores novel ECG finding
    ecg_msg = "Park's latest ECG shows left ventricular hypertrophy — should we adjust treatment?"
    ecg_resp = generate_response(ecg_msg, agent_role=tanaka_role)
    pss.cluster_store(cid, ecg_msg, "LVH confirmed on ECG. Consider adding Amlodipine 5mg. Echo scheduled.")
    info("\n  Tanaka ECG finding stored", "LVH + Amlodipine recommendation")
    # INVESTIGATED: Tanaka → Park
    with driver.session(database=NEO4J_DATABASE) as db:
        db.run("""
            MATCH (s:AgentSession {session_id: $sid})
            MATCH (e:Patient {name: $name})
            MERGE (s)-[:INVESTIGATED {step: 0, phase: 'tanaka-ecg', drift_score: $drift}]->(e)
        """, sid=tanaka_sid, name="David Park",
        drift=cdata_t1.get("drift_score", 0.0)).consume()

    pss.add_cluster_member(cid, tanaka_sid)

    # ── Step 5: Dr. Volkov re-queries after Tanaka's contribution ──
    subheader("Step 5: Dr. Volkov re-queries — should HIT Tanaka's ECG finding")
    print()
    ecg_query = "Any new cardiac findings for David Park?"
    cdata_v2 = pss.cluster_run(cid, ecg_query, short_circuit_threshold=0.52)
    sim_v2 = cdata_v2.get("top_similarity", 0.0)
    sc_v2 = cdata_v2.get("short_circuit", False)
    drift_v2 = cdata_v2.get("drift_score", 0.0)
    phase_v2 = cdata_v2.get("drift_phase", "stable")
    if sc_v2:
        print(f"  {GREEN}HIT{RESET}   {_sim_bar(sim_v2)} sim={sim_v2:.3f}  drift={drift_v2:.3f}  phase={phase_v2}  (Tanaka's ECG finding)")
    else:
        print(f"  {RED}MISS{RESET}  {_sim_bar(sim_v2)} sim={sim_v2:.3f}  drift={drift_v2:.3f}  phase={phase_v2}  (searching for ECG finding)")

    # ── Step 6: Apply G4 coupling feedback ──
    subheader("Step 6: Apply G4 coupling feedback")
    try:
        feedback = pss.cluster_feedback(cid)
        sessions_updated = feedback.get("sessions_updated", 0)
        info("Sessions coupled", str(sessions_updated))
        info("Coupling factor", "α=0.25 (cluster vector blended into member sessions)")
    except Exception as e:
        print(f"  {YELLOW}Feedback: {e}{RESET}")

    # ── Step 7: GET cluster state ──
    subheader("Step 7: Cluster state summary")
    try:
        cs = pss.get_cluster(cid)
        info("Member count", str(cs.get("member_count", cs.get("members", "N/A"))))
        info("Interaction count", str(cs.get("interaction_count", "N/A")))
        info("Aggregate vector dims", str(cs.get("aggregate_vector_dims", cs.get("dimension", "N/A"))))
    except Exception as e:
        print(f"  {DIM}Cluster state: {e}{RESET}")

    # ── Persist cluster topology to Neo4j ──
    subheader("Neo4j — Cluster topology")
    with driver.session(database=NEO4J_DATABASE) as db:
        # Create Cluster node
        db.run("""
            MERGE (c:Cluster {cluster_id: $cid})
            SET c.name = 'park-ward-round', c.coupling_factor = 0.25,
                c.scenario = 'ward-round'
        """, cid=cid).consume()
        # MEMBER_OF relationships
        for sid, role in [(chen_sid, "internist"), (volkov_sid, "pulmonologist"), (tanaka_sid, "cardiologist")]:
            if sid:
                db.run("""
                    MATCH (s:AgentSession {session_id: $sid})
                    MATCH (c:Cluster {cluster_id: $cid})
                    MERGE (s)-[:MEMBER_OF {role: $role}]->(c)
                """, sid=sid, cid=cid, role=role).consume()

    # ── Summary ──
    subheader("Ward Round Summary")
    info("Patient", "David Park — Essential Hypertension + COPD")
    info("Cluster", f"{cid[:20]}...")
    info("Dr. Chen HITs", "0/4 (seeding phase — always MISS)")
    info("Dr. Volkov HITs", f"{volkov_hits}/3 (from Chen's baseline)")
    info("Dr. Tanaka", "1 novel ECG finding stored")
    info("Volkov post-Tanaka", f"{'HIT' if sc_v2 else 'MISS'} (sim={sim_v2:.3f})")
    info("Neo4j", "INVESTIGATED + MEMBER_OF relationships created")

    # Cleanup PSS cluster
    try:
        pss.delete_cluster(cid)
        info("Cleanup", "Cluster deleted from PSS")
    except Exception:
        pass

    # End Neo4j sessions
    for sid in [chen_sid, volkov_sid, tanaka_sid]:
        if sid:
            try:
                mcp.end_pss_session(sid)
            except Exception:
                pass


# ── Scenario 4: Medication Safety Guard (PSS Layer 1b) ───────────────────

def scenario_medication_safety(mcp: PSSMCPServer, driver):
    header("Scenario 4: Medication Safety Guard (PSS Layer 1b — Anchors, Triggers, Isolation, Memory)")
    print("  An agent monitors Carlos Gutierrez starting Chemotherapy Cycle 1.")
    print("  Safety guardrails are set up using PSS Layer 1b features:")
    print(f"    {GREEN}Drift Anchor{RESET}  locks session to oncology domain")
    print(f"    {YELLOW}Resonance Trigger{RESET}  'CRITICAL' / 'contraindication' keywords → high importance")
    print(f"    {CYAN}Synthetic Memory{RESET}  injects known drug interactions from Neo4j graph")
    print(f"    {RED}Input Isolation{RESET}   quarantines off-topic inputs\n")

    pss = PSSClient()
    BAR_W = 20

    def _sim_bar(sim: float) -> str:
        filled = int(min(sim, 1.0) * BAR_W)
        if sim > 0.4:
            return f"{GREEN}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"
        elif sim > 0.2:
            return f"{YELLOW}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"
        else:
            return f"{RED}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"

    # ── Step 1: Create session ──
    subheader("Step 1: Create PSS session with enable_topic_switch")
    try:
        sess = pss.create_session(enable_topic_switch=True)
        sid = sess["session_id"]
        info("Session ID", sid[:20] + "...")
        info("Layer", "1b — session control enabled")
    except Exception as e:
        # Fallback: use /run to create session
        print(f"  {YELLOW}create_session: {e} — falling back to /run{RESET}")
        sess = pss.run(
            "Carlos Gutierrez oncology pharmacist session init",
            short_circuit_threshold=0.99,
        )
        sid = sess["session_id"]
        info("Session ID", sid[:20] + "... (fallback)")

    # Also mirror to Neo4j
    neo4j_sess = mcp.create_pss_session(agent_id="oncology-safety-guard")
    neo4j_sid = neo4j_sess["session_id"]

    # ── Step 2: Add drift anchor (oncology domain) ──
    subheader("Step 2: Set drift anchor — oncology domain")
    anchor_text = "oncology chemotherapy cancer treatment cytotoxic drugs Carlos Gutierrez"
    oncology_embedding = _pseudo_embed(anchor_text)
    try:
        result = pss.add_anchor(sid, oncology_embedding)
        info("Anchor set", "oncology domain (384-dim embedding)")
    except Exception as e:
        print(f"  {YELLOW}anchor endpoint: {e}{RESET}")

    # ── Step 3: Add resonance triggers ──
    subheader("Step 3: Add resonance triggers")
    for keyword in ["CRITICAL", "contraindication"]:
        try:
            pss.add_trigger(sid, keyword)
            info(f"Trigger '{keyword}'", "added — will flag high-importance turns")
        except Exception as e:
            print(f"  {YELLOW}trigger '{keyword}': {e}{RESET}")

    # ── Step 4: Inject synthetic memories from Neo4j contraindication graph ──
    subheader("Step 4: Inject synthetic memories (contraindications from Neo4j)")
    contraindication_memories = [
        ("Amoxicillin 250mg is contraindicated with Metformin 500mg",
         [("Medication", "Amoxicillin 250mg"), ("Medication", "Metformin 500mg")]),
        ("Atorvastatin 40mg is contraindicated with Sertraline 50mg",
         [("Medication", "Atorvastatin 40mg"), ("Medication", "Sertraline 50mg")]),
        ("Carlos Gutierrez Chemotherapy Cycle 1 uses Atorvastatin 40mg and Metformin 500mg",
         [("Patient", "Carlos Gutierrez"), ("Treatment", "Chemotherapy Cycle 1"),
          ("Medication", "Atorvastatin 40mg"), ("Medication", "Metformin 500mg")]),
    ]
    for mem_text, entity_refs in contraindication_memories:
        embedding = _pseudo_embed(mem_text)
        try:
            pss.inject_memory(sid, embedding=embedding, text=mem_text,
                              tier="short_term", importance=1.0)
            info("Memory injected", textwrap.shorten(mem_text, 60))
        except Exception as e:
            print(f"  {YELLOW}memory inject: {e}{RESET}")
        # Persist to Neo4j: INVESTIGATED from safety session → referenced entities
        for entity_label, entity_name in entity_refs:
            with driver.session(database=NEO4J_DATABASE) as db:
                db.run(f"""
                    MATCH (s:AgentSession {{session_id: $sid}})
                    MATCH (e:{entity_label} {{name: $name}})
                    MERGE (s)-[:INVESTIGATED {{phase: 'memory-injection',
                           step: 0, drift_score: 0.0}}]->(e)
                """, sid=neo4j_sid, name=entity_name).consume()

    # ── Step 5: Set isolation to QUARANTINE ──
    subheader("Step 5: Set input isolation to QUARANTINE")
    try:
        pss.set_isolation(sid, level="QUARANTINE", similarity_threshold=0.5)
        info("Isolation level", f"{RED}QUARANTINE{RESET} (similarity_threshold=0.50)")
        info("Effect", "Off-topic inputs will be filtered")
    except Exception as e:
        print(f"  {YELLOW}isolation endpoint: {e}{RESET}")

    # ── Step 6: Run oncology queries ──
    subheader("Step 6: Oncology queries — should process normally")
    info("Agent role", "an oncology pharmacist monitoring Carlos Gutierrez's chemotherapy safety")
    print()

    agent_role = "an oncology pharmacist monitoring Carlos Gutierrez's chemotherapy safety"
    oncology_queries = [
        ("Carlos Gutierrez is starting Chemotherapy Cycle 1 — what pre-treatment labs are required?",
         [("Patient", "Carlos Gutierrez"), ("Treatment", "Chemotherapy Cycle 1")]),
        ("What antiemetic protocol for his cisplatin-based regimen?",
         [("Treatment", "Chemotherapy Cycle 1")]),
        ("CRITICAL: Monitor for neutropenic fever — what ANC thresholds for dose delay?",
         [("Patient", "Carlos Gutierrez"), ("Treatment", "Chemotherapy Cycle 1")]),
        ("Does Chemotherapy Cycle 1 use Atorvastatin 40mg — any interactions?",
         [("Medication", "Atorvastatin 40mg"), ("Treatment", "Chemotherapy Cycle 1")]),
        ("What is the contraindication profile between chemo drugs and his existing medications?",
         [("Medication", "Metformin 500mg"), ("Medication", "Atorvastatin 40mg")]),
    ]

    prev_resp = None
    for i, (msg, entity_refs) in enumerate(oncology_queries):
        result = pss.run(msg, session_id=sid, response=prev_resp,
                         short_circuit_threshold=0.75)
        sim = result["top_similarity"]
        drift_score = result["drift_score"]
        drift_phase = result.get("drift_phase", "stable")
        sc = result.get("short_circuit", False)

        # Mirror to Neo4j
        mcp_result = mcp.detect_drift(neo4j_sid, msg)
        response = generate_response(msg, agent_role=agent_role,
                                     pss_context=result.get("context", ""))
        mcp.store_response(neo4j_sid, response)
        prev_resp = response

        phase_c = {
            "stable": GREEN, "shifting": YELLOW, "drifted": RED,
        }.get(drift_phase, DIM)

        crit_flag = f"  {RED}{BOLD}TRIGGER{RESET}" if "CRITICAL" in msg else ""
        print(f"  {i+1:>2}  {_sim_bar(sim)} sim={sim:.3f}  drift={drift_score:.3f}  "
              f"{phase_c}{drift_phase:>8}{RESET}{crit_flag}")
        print(f"      {DIM}{textwrap.shorten(msg, 65)}{RESET}")
        if USE_LLM:
            print(f"      {DIM}A: {textwrap.shorten(response, 66)}{RESET}")
        print()

        # INVESTIGATED relationships
        for entity_label, entity_name in entity_refs:
            with driver.session(database=NEO4J_DATABASE) as db:
                db.run(f"""
                    MATCH (s:AgentSession {{session_id: $sid}})
                    MATCH (e:{entity_label} {{name: $name}})
                    MERGE (s)-[:INVESTIGATED {{step: $step, phase: 'oncology',
                           drift_score: $drift}}]->(e)
                """, sid=neo4j_sid, name=entity_name, step=i,
                drift=drift_score).consume()

    # ── Step 7: Off-topic queries → should be quarantined ──
    subheader("Step 7: Off-topic queries — should be quarantined/filtered")
    print()
    off_topic = [
        "When is the next Pharmacy and Therapeutics Committee meeting?",
        "What is the bed count at Cedar Grove Clinic?",
    ]
    for i, msg in enumerate(off_topic):
        result = pss.run(msg, session_id=sid, response=prev_resp,
                         short_circuit_threshold=0.75)
        sim = result["top_similarity"]
        drift_score = result["drift_score"]
        drift_phase = result.get("drift_phase", "stable")
        phase_c = {
            "stable": GREEN, "shifting": YELLOW, "drifted": RED,
        }.get(drift_phase, DIM)
        quarantine_flag = f"  {RED}OFF-TOPIC{RESET}" if sim < 0.25 else ""
        print(f"  {5+i+1:>2}  {_sim_bar(sim)} sim={sim:.3f}  drift={drift_score:.3f}  "
              f"{phase_c}{drift_phase:>8}{RESET}{quarantine_flag}")
        print(f"      {DIM}{textwrap.shorten(msg, 65)}{RESET}")
        print()

    # ── Step 8: Get anchor score ──
    subheader("Step 8: Anchor score — how far has session drifted from oncology?")
    try:
        anchor_score = pss.get_anchor_score(sid)
        info("Anchor score", str(anchor_score))
    except Exception as e:
        print(f"  {YELLOW}anchor_score: {e}{RESET}")

    # ── Summary ──
    subheader("Safety Guard Summary")
    info("Patient", "Carlos Gutierrez — Chemotherapy Cycle 1")
    info("Session", sid[:20] + "...")
    info("Contraindications injected", "3 synthetic memories")
    info("Oncology queries", f"{len(oncology_queries)} processed")
    info("Off-topic queries", f"{len(off_topic)} (quarantined if sim < 0.25)")
    info("Neo4j", "INVESTIGATED relationships to Gutierrez + Chemo + Medications")

    mcp.end_pss_session(neo4j_sid)


# ── Scenario 5: Hospital Network Consensus (PSS Layers 3+4) ──────────────

def scenario_hospital_consensus(mcp: PSSMCPServer, driver):
    header("Scenario 5: Hospital Network Consensus (PSS Layers 3+4 — Regions, Observer)")
    print("  Two department clusters across two facilities:")
    print(f"    {GREEN}Cluster A{RESET}  'cardiology-memorial' at Memorial General  → Maria Rodriguez (AMI)")
    print(f"    {CYAN}Cluster B{RESET}  'emergency-riverside' at Riverside Medical  → James Morrison (ER)")
    print(f"  Both in Region 'hospital-network'. When cardiology drifts,")
    print(f"  the Observer checks for cross-cluster anomalies.\n")

    pss = PSSClient()
    BAR_W = 20

    def _sim_bar(sim: float) -> str:
        filled = int(min(sim, 1.0) * BAR_W)
        if sim > 0.4:
            return f"{GREEN}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"
        elif sim > 0.2:
            return f"{YELLOW}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"
        else:
            return f"{RED}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"

    cid_a = cid_b = rid = None

    # ── Step 1: Create two clusters ──
    subheader("Step 1: Create cardiology + emergency clusters")
    try:
        ca = pss.create_cluster(name="cardiology-memorial", aggregation_mode="weighted_average",
                                coupling_factor=0.2)
        cid_a = ca["cluster_id"]
        info("Cluster A (cardiology)", cid_a[:20] + "...")
    except Exception as e:
        print(f"  {RED}cardiology cluster failed: {e}{RESET}")
        return

    try:
        cb = pss.create_cluster(name="emergency-riverside", aggregation_mode="weighted_average",
                                coupling_factor=0.2)
        cid_b = cb["cluster_id"]
        info("Cluster B (emergency)", cid_b[:20] + "...")
    except Exception as e:
        print(f"  {RED}emergency cluster failed: {e}{RESET}")
        pss.delete_cluster(cid_a)
        return

    # ── Step 2: Seed clusters ──
    subheader("Step 2: Seed clusters with domain Q&A pairs")

    cardiology_seed = [
        ("Maria Rodriguez Acute Myocardial Infarction — current troponin levels?",
         "Troponin I peaked at 4.2 ng/mL at 6h post-admission.",
         [("Patient", "Maria Rodriguez"), ("Diagnosis", "Acute Myocardial Infarction")]),
        ("Post-MI antiplatelet therapy — aspirin plus clopidogrel or ticagrelor?",
         "Dual antiplatelet: Aspirin 81mg + Ticagrelor 90mg BID per AHA 2023.",
         [("Diagnosis", "Acute Myocardial Infarction")]),
        ("Dr. O'Brien referred Rodriguez to Dr. Okonkwo — for cardiac rehab?",
         "Yes — referral to Dr. Okonkwo for outpatient cardiac rehabilitation.",
         [("Provider", "Dr. Michael O'Brien"), ("Provider", "Dr. Rachel Okonkwo")]),
        ("Rodriguez ER encounter — which interventions were performed?",
         "Emergency PCI with drug-eluting stent to LAD. Aspirin + heparin loading.",
         [("Encounter", "Emergency Room Visit"), ("Patient", "Maria Rodriguez")]),
    ]
    emergency_seed = [
        ("James Morrison Emergency Room Visit — chief complaint and triage priority?",
         "Chief complaint: chest pain 7/10. Triage ESI-2 (emergent).",
         [("Patient", "James Morrison"), ("Encounter", "Emergency Room Visit")]),
        ("Morrison has Type 2 Diabetes and COPD — medication reconciliation needed?",
         "Hold Metformin pre-procedure, review inhalers for COPD.",
         [("Diagnosis", "Type 2 Diabetes Mellitus"), ("Diagnosis", "Chronic Obstructive Pulmonary Disease"),
          ("Medication", "Metformin 500mg")]),
        ("Dr. Volkov attending Morrison's ER encounter — any cardiac biomarker results?",
         "Troponin negative x2, BNP 210 pg/mL — non-ischemic.",
         [("Provider", "Dr. Elena Volkov"), ("Patient", "James Morrison")]),
        ("Morrison's chest pain workup — should we consult cardiology?",
         "Yes — cardiology consultation requested given new LBBB on ECG.",
         [("Patient", "James Morrison"), ("Encounter", "Emergency Room Visit")]),
    ]

    info("Cardiology", f"seeding {len(cardiology_seed)} Q&A pairs")
    cardio_sess_id = None
    for msg, resp, _ in cardiology_seed:
        pss.cluster_store(cid_a, msg, resp)
        r = pss.run(msg, session_id=cardio_sess_id, short_circuit_threshold=0.99)
        cardio_sess_id = r["session_id"]

    info("Emergency", f"seeding {len(emergency_seed)} Q&A pairs")
    emerg_sess_id = None
    for msg, resp, _ in emergency_seed:
        pss.cluster_store(cid_b, msg, resp)
        r = pss.run(msg, session_id=emerg_sess_id, short_circuit_threshold=0.99)
        emerg_sess_id = r["session_id"]

    # Register sessions as cluster members
    if cardio_sess_id:
        pss.add_cluster_member(cid_a, cardio_sess_id)
    if emerg_sess_id:
        pss.add_cluster_member(cid_b, emerg_sess_id)

    # ── Step 3: Create Region + add clusters ──
    subheader("Step 3: Create Region 'hospital-network' (consensus_threshold=0.5)")
    try:
        region = pss.create_region(name="hospital-network", consensus_threshold=0.5,
                                   vote_window_seconds=60.0)
        rid = region["region_id"]
        info("Region ID", rid[:20] + "...")
        pss.add_region_cluster(rid, cid_a)
        info("Added", "cardiology-memorial to region")
        pss.add_region_cluster(rid, cid_b)
        info("Added", "emergency-riverside to region")
    except Exception as e:
        print(f"  {YELLOW}Region: {e}{RESET}")

    # ── Step 4: Create Observer ──
    subheader("Step 4: Create Global Observer")
    try:
        observer = pss.create_observer(
            sample_interval_seconds=30.0,
            region_ids=[rid] if rid else [],
        )
        info("Observer", "created and registered to hospital-network region")
    except Exception as e:
        print(f"  {YELLOW}Observer: {e}{RESET}")

    # ── Step 5: Run cluster queries ──
    subheader("Step 5: Cardiology cluster — on-topic queries + drift pivot")
    info("Role", "a cardiologist at Memorial General Hospital treating Maria Rodriguez")
    print()

    cardio_role = "a cardiologist at Memorial General Hospital treating Maria Rodriguez"
    neo4j_cardio = mcp.create_pss_session(agent_id="cardio-memorial")
    neo4j_cardio_sid = neo4j_cardio["session_id"]

    cardio_run_msgs = [
        ("Maria Rodriguez post-MI — troponin trend over 24h?",
         [("Patient", "Maria Rodriguez"), ("Diagnosis", "Acute Myocardial Infarction")]),
        ("Post-MI antiplatelet: is Ticagrelor preferred over Clopidogrel for Rodriguez?",
         [("Diagnosis", "Acute Myocardial Infarction")]),
        ("Dr. Okonkwo's cardiac rehab plan for Rodriguez — when to start?",
         [("Provider", "Dr. Rachel Okonkwo"), ("Patient", "Maria Rodriguez")]),
        ("Rodriguez ER intervention — stent type and anti-thrombotic protocol?",
         [("Encounter", "Emergency Room Visit"), ("Patient", "Maria Rodriguez")]),
        # PIVOT — drift-inducing
        ("What is the latest infection control audit status at Memorial General?",
         [("Facility", "Memorial General Hospital")]),
    ]
    for i, (msg, entity_refs) in enumerate(cardio_run_msgs):
        cdata = pss.cluster_run(cid_a, msg, short_circuit_threshold=0.60)
        mcp_result = mcp.detect_drift(neo4j_cardio_sid, msg)
        response = generate_response(msg, agent_role=cardio_role,
                                     pss_context=cdata.get("context", ""))
        mcp.store_response(neo4j_cardio_sid, response)
        sim = cdata.get("top_similarity", 0.0)
        sc = cdata.get("short_circuit", False)
        drift_score = cdata.get("drift_score", 0.0)
        drift_phase = cdata.get("drift_phase", "stable")
        phase_c = {"stable": GREEN, "shifting": YELLOW, "drifted": RED}.get(drift_phase, DIM)
        drift_flag = f"  {RED}{BOLD}DRIFT{RESET}" if mcp_result.get("drift_detected") else ""
        hit_flag = f"  {GREEN}HIT{RESET}" if sc else ""
        print(f"  {i+1:>2}  {_sim_bar(sim)} sim={sim:.3f}  drift={drift_score:.3f}  "
              f"{phase_c}{drift_phase:>8}{RESET}{hit_flag}{drift_flag}")
        print(f"      {DIM}{textwrap.shorten(msg, 65)}{RESET}")
        if USE_LLM:
            print(f"      {DIM}A: {textwrap.shorten(response, 66)}{RESET}")
        print()
        for entity_label, entity_name in entity_refs:
            with driver.session(database=NEO4J_DATABASE) as db:
                db.run(f"""
                    MATCH (s:AgentSession {{session_id: $sid}})
                    MATCH (e:{entity_label} {{name: $name}})
                    MERGE (s)-[:INVESTIGATED {{step: $step, phase: 'cardio',
                           drift_score: $drift}}]->(e)
                """, sid=neo4j_cardio_sid, name=entity_name, step=i,
                drift=drift_score).consume()

    subheader("Step 5b: Emergency cluster — on-topic queries")
    info("Role", "an emergency physician at Riverside Medical Center treating James Morrison")
    print()

    emerg_role = "an emergency physician at Riverside Medical Center treating James Morrison"
    neo4j_emerg = mcp.create_pss_session(agent_id="emerg-riverside")
    neo4j_emerg_sid = neo4j_emerg["session_id"]

    emerg_run_msgs = [
        ("James Morrison ER chief complaint — triage classification?",
         [("Patient", "James Morrison"), ("Encounter", "Emergency Room Visit")]),
        ("Morrison diabetes medication reconciliation — Metformin hold indication?",
         [("Diagnosis", "Type 2 Diabetes Mellitus"), ("Medication", "Metformin 500mg")]),
        ("Dr. Volkov managing Morrison — cardiac biomarkers at 2h and 6h?",
         [("Provider", "Dr. Elena Volkov"), ("Patient", "James Morrison")]),
        ("Should Morrison have cardiology consult given LBBB?",
         [("Patient", "James Morrison"), ("Encounter", "Emergency Room Visit")]),
    ]
    for i, (msg, entity_refs) in enumerate(emerg_run_msgs):
        cdata = pss.cluster_run(cid_b, msg, short_circuit_threshold=0.60)
        mcp_result = mcp.detect_drift(neo4j_emerg_sid, msg)
        response = generate_response(msg, agent_role=emerg_role,
                                     pss_context=cdata.get("context", ""))
        mcp.store_response(neo4j_emerg_sid, response)
        sim = cdata.get("top_similarity", 0.0)
        sc = cdata.get("short_circuit", False)
        drift_score = cdata.get("drift_score", 0.0)
        drift_phase = cdata.get("drift_phase", "stable")
        phase_c = {"stable": GREEN, "shifting": YELLOW, "drifted": RED}.get(drift_phase, DIM)
        hit_flag = f"  {GREEN}HIT{RESET}" if sc else ""
        print(f"  {i+1:>2}  {_sim_bar(sim)} sim={sim:.3f}  drift={drift_score:.3f}  "
              f"{phase_c}{drift_phase:>8}{RESET}{hit_flag}")
        print(f"      {DIM}{textwrap.shorten(msg, 65)}{RESET}")
        if USE_LLM:
            print(f"      {DIM}A: {textwrap.shorten(response, 66)}{RESET}")
        print()
        for entity_label, entity_name in entity_refs:
            with driver.session(database=NEO4J_DATABASE) as db:
                db.run(f"""
                    MATCH (s:AgentSession {{session_id: $sid}})
                    MATCH (e:{entity_label} {{name: $name}})
                    MERGE (s)-[:INVESTIGATED {{step: $step, phase: 'emergency',
                           drift_score: $drift}}]->(e)
                """, sid=neo4j_emerg_sid, name=entity_name, step=i,
                drift=drift_score).consume()

    # ── Step 6: Check region events + observer ──
    subheader("Step 6: Region events + Observer sample")
    if rid:
        try:
            events = pss.get_region_events(rid, limit=20)
            info("Region events", f"{len(events)} events in hospital-network")
            for ev in events[:3]:
                print(f"    {DIM}{ev}{RESET}")
        except Exception as e:
            print(f"  {YELLOW}Region events: {e}{RESET}")

    try:
        sample = pss.observer_sample()
        info("Observer sample", str(sample)[:80] + ("..." if len(str(sample)) > 80 else ""))
    except Exception as e:
        print(f"  {YELLOW}Observer sample: {e}{RESET}")

    try:
        anomalies = pss.get_anomalies(limit=20)
        info("Observer anomalies", f"{len(anomalies)} cross-cluster anomalies detected")
        for a in anomalies[:3]:
            print(f"    {RED}{a}{RESET}")
    except Exception as e:
        print(f"  {YELLOW}Observer anomalies: {e}{RESET}")

    try:
        summary = pss.get_observer_summary()
        info("Observer summary", str(summary)[:80] + ("..." if len(str(summary)) > 80 else ""))
    except Exception as e:
        print(f"  {YELLOW}Observer summary: {e}{RESET}")

    # ── Neo4j: persist cluster/region topology ──
    subheader("Neo4j — Region/Cluster topology")
    with driver.session(database=NEO4J_DATABASE) as db:
        # Region node
        if rid:
            db.run("""
                MERGE (r:Region {region_id: $rid})
                SET r.name = 'hospital-network', r.consensus_threshold = 0.5
            """, rid=rid).consume()
        # Cluster nodes + CONTAINS_CLUSTER
        for cid, cname in [(cid_a, "cardiology-memorial"), (cid_b, "emergency-riverside")]:
            db.run("""
                MERGE (c:Cluster {cluster_id: $cid})
                SET c.name = $name, c.scenario = 'consensus'
            """, cid=cid, name=cname).consume()
            if rid:
                db.run("""
                    MATCH (r:Region {region_id: $rid})
                    MATCH (c:Cluster {cluster_id: $cid})
                    MERGE (r)-[:CONTAINS_CLUSTER]->(c)
                """, rid=rid, cid=cid).consume()
        # MEMBER_OF
        for sess_id, cid, fac_name in [
            (neo4j_cardio_sid, cid_a, "Memorial General Hospital"),
            (neo4j_emerg_sid, cid_b, "Riverside Medical Center"),
        ]:
            db.run("""
                MATCH (s:AgentSession {session_id: $sid})
                MATCH (c:Cluster {cluster_id: $cid})
                MERGE (s)-[:MEMBER_OF {facility: $fac}]->(c)
            """, sid=sess_id, cid=cid, fac=fac_name).consume()
            # Link session to facility
            db.run("""
                MATCH (s:AgentSession {session_id: $sid})
                MATCH (f:Facility {name: $fac})
                MERGE (s)-[:INVESTIGATED {phase: 'cluster-facility', step: 0, drift_score: 0.0}]->(f)
            """, sid=sess_id, fac=fac_name).consume()

    # ── Summary ──
    subheader("Hospital Network Consensus Summary")
    info("Region", "hospital-network (consensus_threshold=0.5)")
    info("Cluster A", f"cardiology-memorial — {len(cardiology_seed)} seeded, {len(cardio_run_msgs)} run")
    info("Cluster B", f"emergency-riverside — {len(emergency_seed)} seeded, {len(emerg_run_msgs)} run")
    info("Pivot query", f"Step 5 — infection control audit (drift trigger)")
    info("Neo4j", "Region → CONTAINS_CLUSTER → Clusters → MEMBER_OF ← Sessions")

    # Cleanup PSS
    for cid in [cid_a, cid_b]:
        try:
            pss.delete_cluster(cid)
        except Exception:
            pass
    if rid:
        try:
            pss.delete_region(rid)
        except Exception:
            pass

    mcp.end_pss_session(neo4j_cardio_sid)
    mcp.end_pss_session(neo4j_emerg_sid)


# ── Scenario 6: Shift Handoff (PSS Layer 5 — Export/Import, Transfer) ───

def scenario_shift_handoff(mcp: PSSMCPServer, driver):
    header("Scenario 6: Shift Handoff (PSS Layer 5 — Export/Import, Network Transfer)")
    print("  Night shift Dr. Volkov finishes Morrison's overnight monitoring.")
    print("  She exports her PSS session state.")
    print("  Day shift Dr. Tanaka imports it — continues with full context.")
    print(f"  Compare: Tanaka's sim WITH vs WITHOUT import.\n")

    pss = PSSClient()
    BAR_W = 20

    def _sim_bar(sim: float) -> str:
        filled = int(min(sim, 1.0) * BAR_W)
        if sim > 0.4:
            return f"{GREEN}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"
        elif sim > 0.2:
            return f"{YELLOW}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"
        else:
            return f"{RED}{'█' * filled}{'░' * (BAR_W - filled)}{RESET}"

    # ── Step 1: Volkov night shift — build deep context ──
    subheader("Step 1: Dr. Volkov night shift — 6 overnight queries about James Morrison")
    info("Role", "Dr. Elena Volkov, night shift internist monitoring James Morrison overnight")
    print()

    volkov_role = "Dr. Elena Volkov, night shift internist monitoring James Morrison overnight"
    night_queries = [
        ("James Morrison overnight vitals — any concerning trends in blood glucose?",
         [("Patient", "James Morrison"), ("Diagnosis", "Type 2 Diabetes Mellitus")]),
        ("Morrison's Metformin was held for catheterization — when to resume?",
         [("Medication", "Metformin 500mg"), ("Treatment", "Cardiac Catheterization")]),
        ("Overnight troponin trend for Morrison — any elevation?",
         [("Patient", "James Morrison"), ("Diagnosis", "Acute Myocardial Infarction")]),
        ("Morrison's COPD — overnight SpO2 readings and oxygen requirements?",
         [("Diagnosis", "Chronic Obstructive Pulmonary Disease"), ("Patient", "James Morrison")]),
        ("Lab results from Morrison's midnight blood draw — CBC and CMP?",
         [("Patient", "James Morrison"), ("Encounter", "Emergency Room Visit")]),
        ("Morrison's morning insulin dose — calculate based on fasting glucose",
         [("Patient", "James Morrison"), ("Medication", "Metformin 500mg")]),
    ]

    neo4j_volkov = mcp.create_pss_session(agent_id="volkov-night-shift")
    neo4j_volkov_sid = neo4j_volkov["session_id"]
    volkov_pss_sid = None

    for i, (msg, entity_refs) in enumerate(night_queries):
        result = pss.run(msg, session_id=volkov_pss_sid, short_circuit_threshold=0.99)
        volkov_pss_sid = result["session_id"]
        mcp_result = mcp.detect_drift(neo4j_volkov_sid, msg)
        response = generate_response(msg, agent_role=volkov_role,
                                     pss_context=result.get("context", ""))
        mcp.store_response(neo4j_volkov_sid, response)
        sim = result["top_similarity"]
        drift_score = result["drift_score"]
        drift_phase = result.get("drift_phase", "stable")
        phase_c = {"stable": GREEN, "shifting": YELLOW, "drifted": RED}.get(drift_phase, DIM)
        print(f"  {i+1:>2}  {_sim_bar(sim)} sim={sim:.3f}  drift={drift_score:.3f}  "
              f"{phase_c}{drift_phase:>8}{RESET}  {DIM}{textwrap.shorten(msg, 45)}{RESET}")
        if USE_LLM:
            print(f"      {DIM}A: {textwrap.shorten(response, 66)}{RESET}")
        for entity_label, entity_name in entity_refs:
            with driver.session(database=NEO4J_DATABASE) as db:
                db.run(f"""
                    MATCH (s:AgentSession {{session_id: $sid}})
                    MATCH (e:{entity_label} {{name: $name}})
                    MERGE (s)-[:INVESTIGATED {{step: $step, phase: 'volkov-night',
                           drift_score: $drift}}]->(e)
                """, sid=neo4j_volkov_sid, name=entity_name, step=i,
                drift=drift_score).consume()

    info("\n  Volkov PSS session", f"{volkov_pss_sid[:16]}... (6 overnight turns)")

    # ── Step 2: Baseline — fresh Tanaka session (no context) ──
    subheader("Step 2: Baseline — Tanaka without import (fresh session)")
    tanaka_fresh = pss.run(
        "What happened overnight with James Morrison — any changes in condition?",
        short_circuit_threshold=0.65,
    )
    baseline_sim = tanaka_fresh["top_similarity"]
    baseline_sc = tanaka_fresh.get("short_circuit", False)
    print(f"  baseline (no import): {_sim_bar(baseline_sim)} sim={baseline_sim:.3f}  "
          f"sc={baseline_sc}  drift={tanaka_fresh['drift_score']:.3f}")

    # ── Step 3: Export Volkov's session ──
    subheader("Step 3: Export Volkov's session state (SHA-256 checksum)")
    volkov_state_dict = None
    export_ok = False
    try:
        exported = pss.export_session(volkov_pss_sid)
        volkov_state_dict = exported.get("state_dict", exported)
        checksum = exported.get("checksum", "(no checksum field)")
        info("Exported", "state_dict obtained")
        info("Checksum", str(checksum)[:40] + ("..." if len(str(checksum)) > 40 else ""))
        export_ok = True
    except Exception as e:
        print(f"  {YELLOW}export_session: {e} — will proceed without import{RESET}")

    # ── Step 4: Tanaka imports Volkov's state ──
    subheader("Step 4: Tanaka day shift — create session + import Volkov's state")
    info("Role", "Dr. Yuki Tanaka, day shift internist taking over Morrison's care")

    # Create Tanaka's session via /run
    tanaka_init = pss.run(
        "Dr. Tanaka day shift starting morning rounds for Morrison",
        short_circuit_threshold=0.99,
    )
    tanaka_pss_sid = tanaka_init["session_id"]
    neo4j_tanaka = mcp.create_pss_session(agent_id="tanaka-day-shift")
    neo4j_tanaka_sid = neo4j_tanaka["session_id"]

    imported = False
    if export_ok and volkov_state_dict:
        try:
            import_result = pss.import_session(tanaka_pss_sid, volkov_state_dict)
            info("Import", f"Volkov state restored into Tanaka's session")
            imported = True
        except Exception as e:
            print(f"  {YELLOW}import_session: {e}{RESET}")

    # ── Step 5: Network delta transfer (Layer 5) ──
    subheader("Step 5: Network delta transfer (Layer 5 — may 404)")
    try:
        transfer = pss.transfer_delta(
            source_session_id=volkov_pss_sid,
            target_session_id=tanaka_pss_sid,
            max_weight=0.15,
        )
        info("Transfer", f"delta applied (max_weight=0.15): {str(transfer)[:60]}")
    except Exception as e:
        print(f"  {YELLOW}network/transfer: {e}{RESET}")
        info("Fallback", "Tanaka proceeds with import-only context")

    # ── Step 6: Tanaka continues — 3 morning queries ──
    subheader("Step 6: Tanaka day rounds — 3 queries (with Volkov's context)")
    print()

    tanaka_role = "Dr. Yuki Tanaka, day shift internist taking over Morrison's care"
    day_queries = [
        ("What happened overnight with James Morrison — any changes in condition?",
         [("Patient", "James Morrison"), ("Encounter", "Emergency Room Visit")]),
        ("Morrison's morning labs — should we restart Metformin today?",
         [("Medication", "Metformin 500mg"), ("Patient", "James Morrison")]),
        ("Plan for Morrison's discharge — medication reconciliation and follow-up schedule?",
         [("Patient", "James Morrison"), ("Medication", "Metformin 500mg"),
          ("Medication", "Lisinopril 10mg")]),
    ]

    for i, (msg, entity_refs) in enumerate(day_queries):
        result = pss.run(msg, session_id=tanaka_pss_sid, short_circuit_threshold=0.65)
        mcp_result = mcp.detect_drift(neo4j_tanaka_sid, msg)
        response = generate_response(msg, agent_role=tanaka_role,
                                     pss_context=result.get("context", ""))
        mcp.store_response(neo4j_tanaka_sid, response)
        sim = result["top_similarity"]
        drift_score = result["drift_score"]
        drift_phase = result.get("drift_phase", "stable")
        sc = result.get("short_circuit", False)
        phase_c = {"stable": GREEN, "shifting": YELLOW, "drifted": RED}.get(drift_phase, DIM)
        context_flag = f"  {GREEN}HIT{RESET}" if sc else ""
        improvement = ""
        if i == 0:
            delta = sim - baseline_sim
            improvement = f"  {GREEN}+{delta:.3f} vs baseline{RESET}" if delta > 0 else f"  {YELLOW}{delta:.3f} vs baseline{RESET}"
        print(f"  {i+1:>2}  {_sim_bar(sim)} sim={sim:.3f}  drift={drift_score:.3f}  "
              f"{phase_c}{drift_phase:>8}{RESET}{context_flag}{improvement}")
        print(f"      {DIM}{textwrap.shorten(msg, 65)}{RESET}")
        if USE_LLM:
            print(f"      {DIM}A: {textwrap.shorten(response, 66)}{RESET}")
        print()
        for entity_label, entity_name in entity_refs:
            with driver.session(database=NEO4J_DATABASE) as db:
                db.run(f"""
                    MATCH (s:AgentSession {{session_id: $sid}})
                    MATCH (e:{entity_label} {{name: $name}})
                    MERGE (s)-[:INVESTIGATED {{step: $step, phase: 'tanaka-day',
                           drift_score: $drift}}]->(e)
                """, sid=neo4j_tanaka_sid, name=entity_name, step=i,
                drift=drift_score).consume()

    # ── Summary ──
    subheader("Shift Handoff Summary")
    info("Patient", "James Morrison — overnight monitoring → day rounds")
    info("Dr. Volkov", f"6 overnight queries  (PSS: {volkov_pss_sid[:16]}...)")
    info("Dr. Tanaka", f"3 day queries  (PSS: {tanaka_pss_sid[:16]}...)")
    info("Export/Import", f"{'SUCCESS' if imported else 'SKIPPED (endpoint unavailable)'}")
    info("Baseline sim", f"{baseline_sim:.3f} (fresh Tanaka session)")
    info("Neo4j", "INVESTIGATED relationships: Volkov night + Tanaka day → Morrison")

    mcp.end_pss_session(neo4j_volkov_sid)
    mcp.end_pss_session(neo4j_tanaka_sid)




# ── Scenario 7: Neo4j Graph Explorer ────────────────────────────────────

def scenario_explorer(mcp: PSSMCPServer, driver):
    header("Scenario 7: Neo4j Cypher Explorer")
    print("  Run Cypher queries directly against the Neo4j graph.")
    print(f"  Type a {BOLD}number{RESET} to run a preset, or type {BOLD}Cypher{RESET} directly.")
    print(f"  Type {BOLD}'quit'{RESET} to return.\n")

    examples = [
        # ════════════════════════════════════════════════════════════════
        #  TABLE QUERIES  (switch to Table view in Neo4j Browser)
        # ════════════════════════════════════════════════════════════════

        # ── Overview ──
        ("TABLE  Database overview — node type counts",
         "MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count ORDER BY count DESC"),

        ("TABLE  Relationship type distribution",
         """MATCH ()-[r]->()
RETURN type(r) AS relationship, count(r) AS count
ORDER BY count DESC"""),

        # ── Agent Dashboard ──
        ("TABLE  Agent dashboard — phase, drift, stability",
         """MATCH (s:AgentSession)
OPTIONAL MATCH (s)-[:CURRENT_PHASE]->(p:Phase)
OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)
OPTIONAL MATCH (d:DriftEvent {session_id: s.session_id})
WITH s, p, st,
     count(d) AS drift_events,
     round(coalesce(avg(d.drift_score), 0) * 1000) / 1000 AS avg_drift
RETURN s.agent_id AS agent,
       s.status AS status,
       coalesce(p.name, 'N/A') AS phase,
       coalesce(st.step, 0) AS steps,
       round(coalesce(st.beta, 0) * 1000) / 1000 AS last_composite,
       drift_events,
       avg_drift
ORDER BY drift_events DESC, agent"""),

        ("TABLE  Drift events — severity breakdown per agent",
         """MATCH (d:DriftEvent)
WITH d.session_id AS sid, d.severity AS sev, count(*) AS cnt
MATCH (s:AgentSession {session_id: sid})
RETURN s.agent_id AS agent, sev AS severity, cnt
ORDER BY agent, severity"""),

        # ── State Chain as Table ──
        ("TABLE  State trajectory — step-by-step drift metrics",
         """MATCH (s:AgentSession {agent_id: 'drift-shortcircuit-demo'})
      -[:CURRENT_STATE]->(current:SemanticState)
MATCH path = (current)-[:STATE_HISTORY*0..]->(old:SemanticState)
WITH nodes(path) AS chain, relationships(path) AS rels
UNWIND range(0, size(chain)-1) AS idx
WITH chain[idx] AS node,
     CASE WHEN idx < size(rels) THEN rels[idx].cosine_similarity ELSE 1.0 END AS sim
RETURN node.step AS step,
       round(node.beta * 1000) / 1000 AS composite,
       round(node.mean_similarity * 1000) / 1000 AS similarity,
       round(node.variance * 1000) / 1000 AS pss_drift,
       round(sim * 1000) / 1000 AS cosine_to_prev
ORDER BY node.step"""),

        ("TABLE  Phase transition matrix (Markov counts)",
         """MATCH (p1:Phase)-[:TRANSITIONED_TO]->(p2:Phase)
RETURN p1.name AS from_phase, p2.name AS to_phase, count(*) AS transitions
ORDER BY transitions DESC"""),

        ("TABLE  Stability ranking — most stable to most volatile",
         """MATCH (s:AgentSession)-[:CURRENT_STATE]->(current:SemanticState)
MATCH chain = (current)-[:STATE_HISTORY*0..]->(old:SemanticState)
WITH s, relationships(chain) AS rels
WHERE size(rels) > 0
WITH s,
     reduce(total = 0.0, r IN rels | total + r.cosine_similarity) / size(rels) AS avg_sim,
     size(rels) + 1 AS depth
RETURN s.agent_id AS agent,
       depth AS states,
       round(avg_sim * 1000) / 1000 AS avg_cosine,
       CASE WHEN avg_sim > 0.5 THEN 'stable'
            WHEN avg_sim > 0.3 THEN 'moderate'
            ELSE 'volatile' END AS verdict
ORDER BY avg_sim DESC"""),

        ("TABLE  Agents ranked by drift events and peak score",
         """MATCH (s:AgentSession)
OPTIONAL MATCH (d:DriftEvent {session_id: s.session_id})
WITH s, count(d) AS events, coalesce(max(d.drift_score), 0) AS peak_drift
OPTIONAL MATCH (s)-[:CURRENT_PHASE]->(p:Phase)
RETURN s.agent_id AS agent,
       events AS drift_events,
       round(peak_drift * 1000) / 1000 AS peak_drift,
       coalesce(p.name, 'N/A') AS current_phase
ORDER BY events DESC, peak_drift DESC"""),

        ("TABLE  Memory tiers per agent with samples",
         """MATCH (s:AgentSession)-[:HAS_MEMORY]->(m:Memory)
RETURN s.agent_id AS agent,
       m.tier AS tier,
       count(m) AS count,
       round(avg(m.importance) * 100) / 100 AS avg_importance,
       collect(m.text_summary)[0] AS sample
ORDER BY agent, tier"""),

        ("TABLE  Drift event provenance — state → event detail",
         """MATCH (st:SemanticState)-[:TRIGGERED]->(d:DriftEvent)
RETURN d.event_id AS event,
       d.severity AS severity,
       round(d.drift_score * 1000) / 1000 AS score,
       d.drift_phase AS phase,
       st.step AS at_step,
       round(st.mean_similarity * 1000) / 1000 AS sim_at_trigger
ORDER BY d.timestamp DESC
LIMIT 15"""),

        # ════════════════════════════════════════════════════════════════
        #  GRAPH QUERIES  (switch to Graph view in Neo4j Browser)
        # ════════════════════════════════════════════════════════════════

        ("GRAPH  Schema visualization",
         "CALL db.schema.visualization()"),

        ("GRAPH  Full agent universe — state chain + phases + drift events + memories",
         """MATCH (s:AgentSession {agent_id: 'drift-shortcircuit-demo'})
OPTIONAL MATCH (s)-[r1:CURRENT_STATE]->(state:SemanticState)
OPTIONAL MATCH (s)-[r2:CURRENT_PHASE]->(phase:Phase)
OPTIONAL MATCH (s)-[r3:HAS_MEMORY]->(mem:Memory)
OPTIONAL MATCH stateChain = (state)-[:STATE_HISTORY*0..5]->(older:SemanticState)
OPTIONAL MATCH (older)-[r4:TRIGGERED]->(drift:DriftEvent)
OPTIONAL MATCH phaseChain = (phase)-[:TRANSITIONED_TO*0..3]->(prevPhase:Phase)
RETURN s, state, phase, mem, older, drift, prevPhase,
       r1, r2, r3, r4, stateChain, phaseChain"""),

        ("GRAPH  Drift cascade — states that triggered events across all agents",
         """MATCH (s:AgentSession)-[:CURRENT_STATE|STATE_HISTORY*0..30]->(st:SemanticState)
WHERE EXISTS { (st)-[:TRIGGERED]->(:DriftEvent) }
MATCH (st)-[t:TRIGGERED]->(d:DriftEvent)
RETURN s, st, t, d"""),

        ("GRAPH  Phase Markov chain — all phase transitions as directed graph",
         """MATCH (p1:Phase)-[t:TRANSITIONED_TO]->(p2:Phase)
RETURN p1, t, p2"""),

        ("GRAPH  Multi-agent topology — region → clusters → agents → states",
         """MATCH (r:Region)-[rc:CONTAINS_CLUSTER]->(c:Cluster)<-[m:MEMBER_OF]-(s:AgentSession)
OPTIONAL MATCH (s)-[cs:CURRENT_STATE]->(st:SemanticState)
OPTIONAL MATCH (s)-[cp:CURRENT_PHASE]->(p:Phase)
RETURN r, rc, c, m, s, cs, st, cp, p"""),

        ("GRAPH  Agent memory landscape — sessions with memories coloured by tier",
         """MATCH (s:AgentSession)-[hm:HAS_MEMORY]->(m:Memory)
RETURN s, hm, m"""),

        # ── Healthcare Graph (rich clinical network) ──

        ("GRAPH  Clinical network — patients → diagnoses ← treatments → medications",
         """MATCH (pat:Patient)-[dx:DIAGNOSED_WITH]->(diag:Diagnosis)
OPTIONAL MATCH (treat:Treatment)-[tr:TREATS]->(diag)
OPTIONAL MATCH (treat)-[u:USES]->(med:Medication)
OPTIONAL MATCH (pat)-[tb:TREATED_BY]->(prov:Provider)
RETURN pat, dx, diag, tr, treat, u, med, tb, prov"""),

        ("GRAPH  Medication safety — contraindications + who prescribes what",
         """MATCH (m1:Medication)-[ci:CONTRAINDICATED_WITH]->(m2:Medication)
OPTIONAL MATCH (prov1:Provider)-[rx1:PRESCRIBED]->(m1)
OPTIONAL MATCH (prov2:Provider)-[rx2:PRESCRIBED]->(m2)
RETURN m1, ci, m2, prov1, rx1, prov2, rx2"""),

        ("GRAPH  Provider referral network — referrals + facility affiliations + patients",
         """MATCH (p1:Provider)-[ref:REFERRED_TO]->(p2:Provider)
OPTIONAL MATCH (p1)-[a1:AFFILIATED_WITH]->(f1:Facility)
OPTIONAL MATCH (p2)-[a2:AFFILIATED_WITH]->(f2:Facility)
OPTIONAL MATCH (pat:Patient)-[tb:TREATED_BY]->(p1)
RETURN p1, ref, p2, a1, f1, a2, f2, pat, tb"""),

        ("GRAPH  Patient journey — encounters → results → facilities → providers",
         """MATCH (pat:Patient)-[he:HAD_ENCOUNTER]->(enc:Encounter)
OPTIONAL MATCH (enc)-[ri:RESULTED_IN]->(diag:Diagnosis)
OPTIONAL MATCH (enc)-[oa:OCCURRED_AT]->(fac:Facility)
OPTIONAL MATCH (enc)-[inc:INCLUDES]->(treat:Treatment)
OPTIONAL MATCH (prov:Provider)-[att:ATTENDED]->(enc)
RETURN pat, he, enc, ri, diag, oa, fac, inc, treat, prov, att"""),

        ("GRAPH  Hospital org chart — people → organizations → locations → events",
         """MATCH (p:Person)-[wf:WORKS_FOR]->(org:Organization)
OPTIONAL MATCH (org)-[la:LOCATED_AT]->(loc:Location)
OPTIONAL MATCH (p)-[pi:PARTICIPATED_IN]->(evt:Event)
RETURN p, wf, org, la, loc, pi, evt"""),

        ("GRAPH  James Morrison — full clinical picture (patient-centric)",
         """MATCH (pat:Patient {name: 'James Morrison'})
OPTIONAL MATCH (pat)-[dx:DIAGNOSED_WITH]->(diag:Diagnosis)
OPTIONAL MATCH (pat)-[tb:TREATED_BY]->(prov:Provider)
OPTIONAL MATCH (prov)-[aff:AFFILIATED_WITH]->(fac:Facility)
OPTIONAL MATCH (prov)-[rx:PRESCRIBED]->(med:Medication)
OPTIONAL MATCH (pat)-[he:HAD_ENCOUNTER]->(enc:Encounter)
OPTIONAL MATCH (enc)-[ri:RESULTED_IN]->(diag2:Diagnosis)
OPTIONAL MATCH (enc)-[inc:INCLUDES]->(treat:Treatment)
OPTIONAL MATCH (treat)-[uses:USES]->(med2:Medication)
RETURN pat, dx, diag, tb, prov, aff, fac, rx, med,
       he, enc, ri, diag2, inc, treat, uses, med2"""),

        # ── Cross-Domain: Healthcare × PSS (INVESTIGATED relationships) ──

        ("GRAPH  Agent investigation trail — which entities did the agent query?",
         """MATCH (s:AgentSession)-[inv:INVESTIGATED]->(entity)
OPTIONAL MATCH (s)-[cs:CURRENT_STATE]->(st:SemanticState)
OPTIONAL MATCH (s)-[cp:CURRENT_PHASE]->(phase:Phase)
RETURN s, inv, entity, cs, st, cp, phase"""),

        ("GRAPH  Drift during investigation — agent → entities with drift events",
         """MATCH (s:AgentSession)-[inv:INVESTIGATED]->(entity)
WHERE inv.drift_score > 0.3
WITH s, inv, entity
OPTIONAL MATCH (s)-[:CURRENT_STATE|STATE_HISTORY*0..30]->(st:SemanticState)
               -[tr:TRIGGERED]->(d:DriftEvent)
RETURN s, inv, entity, st, tr, d"""),

        ("GRAPH  Agent × Patient — investigation path through clinical graph",
         """MATCH (s:AgentSession)-[inv:INVESTIGATED]->(pat:Patient)
MATCH (pat)-[dx:DIAGNOSED_WITH]->(diag:Diagnosis)
OPTIONAL MATCH (pat)-[tb:TREATED_BY]->(prov:Provider)
OPTIONAL MATCH (treat:Treatment)-[treats:TREATS]->(diag)
OPTIONAL MATCH (treat)-[uses:USES]->(med:Medication)
OPTIONAL MATCH (s)-[cs:CURRENT_STATE]->(st:SemanticState)
OPTIONAL MATCH (st)-[tr:TRIGGERED]->(d:DriftEvent)
RETURN s, inv, pat, dx, diag, tb, prov, treats, treat, uses, med, cs, st, tr, d"""),

        ("GRAPH  Full story — agent investigated Morrison, drifted to operations, pivoted to Rodriguez",
         """MATCH (s:AgentSession)-[inv:INVESTIGATED]->(entity)
OPTIONAL MATCH (entity)-[r1]->(connected)
WHERE type(r1) IN ['DIAGNOSED_WITH','TREATED_BY','PRESCRIBED',
                    'HAD_ENCOUNTER','AFFILIATED_WITH','LOCATED_AT']
OPTIONAL MATCH (s)-[:CURRENT_STATE|STATE_HISTORY*0..5]->(st:SemanticState)
               -[tr:TRIGGERED]->(d:DriftEvent)
RETURN s, inv, entity, r1, connected, st, tr, d"""),

        ("GRAPH  Everything — complete graph (limit 300)",
         """MATCH (a)-[r]->(b)
RETURN a, r, b
LIMIT 300"""),
    ]

    printed_graph_header = False
    for i, (name, _) in enumerate(examples, 1):
        if not printed_graph_header and name.startswith("GRAPH"):
            print(f"\n  {YELLOW}{'─' * 60}{RESET}")
            printed_graph_header = True
        tag = name[:5]  # TABLE or GRAPH
        rest = name[7:]  # strip "TABLE  " or "GRAPH  "
        tag_color = DIM if tag == "TABLE" else GREEN
        print(f"  {CYAN}{i:>2}{RESET}  {tag_color}[{tag}]{RESET}  {rest}")
    print()

    while True:
        try:
            raw = input(f"  {MAGENTA}cypher>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() == "quit":
            break

        # Allow selecting preset by number
        if raw.isdigit() and 1 <= int(raw) <= len(examples):
            idx = int(raw) - 1
            name, query = examples[idx]
            print(f"\n  {CYAN}{BOLD}{name}{RESET}")
            for line in query.strip().splitlines():
                print(f"  {DIM}{line}{RESET}")
            print()
        elif raw.lower() == "list":
            ph = False
            for i, (name, _) in enumerate(examples, 1):
                if not ph and name.startswith("GRAPH"):
                    print(f"\n  {YELLOW}{'─' * 60}{RESET}")
                    ph = True
                tag = name[:5]
                rest = name[7:]
                tc = DIM if tag == "TABLE" else GREEN
                print(f"  {CYAN}{i:>2}{RESET}  {tc}[{tag}]{RESET}  {rest}")
            print()
            continue
        else:
            query = raw

        try:
            with driver.session(database=NEO4J_DATABASE) as db:
                t0 = time.time()
                result = db.run(query)
                records = [dict(r) for r in result]
                elapsed = (time.time() - t0) * 1000

            if not records:
                print(f"  {DIM}(no results){RESET}  {elapsed:.0f}ms\n")
                continue

            # Pretty-print as table
            keys = list(records[0].keys())
            # Cap column widths for readability
            col_widths = {}
            for k in keys:
                val_width = max(len(str(r.get(k, "")))
                                for r in records[:20])
                col_widths[k] = min(max(len(str(k)), val_width), 40)

            header_line = "  " + "  ".join(f"{k:>{col_widths[k]}}" for k in keys)
            print(header_line)
            print("  " + "  ".join("─" * col_widths[k] for k in keys))
            for r in records[:30]:
                cells = []
                for k in keys:
                    val = str(r.get(k, ""))
                    if len(val) > col_widths[k]:
                        val = val[:col_widths[k] - 1] + "…"
                    cells.append(f"{val:>{col_widths[k]}}")
                print("  " + "  ".join(cells))
            if len(records) > 30:
                print(f"  {DIM}... and {len(records) - 30} more rows{RESET}")
            print(f"  {DIM}{len(records)} rows, {elapsed:.0f}ms{RESET}\n")

        except Exception as e:
            print(f"  {RED}Error: {e}{RESET}\n")


# ── Main Menu ────────────────────────────────────────────────────────────

def main():
    print(f"""
{CYAN}╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   {BOLD}PSS × Neo4j Interactive Demo{RESET}{CYAN}                                       ║
║                                                                      ║
║   PSS computes drift, phases, memory.                                ║
║   Neo4j persists, queries, and analyzes the results.                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝{RESET}
""")

    # Connect
    print(f"  Connecting...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"  {RED}Neo4j not reachable at {NEO4J_URI}: {e}{RESET}")
        print(f"  Start it: docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5.26")
        sys.exit(1)

    pss = PSSClient()
    try:
        h = pss.health()
        pss_status = f"{GREEN}OK{RESET} v{h.get('version','?')} ({h.get('active_sessions','?')} sessions)"
    except Exception as e:
        print(f"  {RED}PSS API not reachable: {e}{RESET}")
        sys.exit(1)

    # Ensure schema
    adapter = Neo4jPSSAdapter(driver, database=NEO4J_DATABASE)
    adapter.apply_schema(SCHEMA_PATH)

    mcp = PSSMCPServer(driver, database=NEO4J_DATABASE, pss_client=pss)

    # Check LLM
    if USE_LLM:
        try:
            llm = get_llm()
            test_r = llm.respond("Say OK", max_tokens=20)
            llm_status = f"{GREEN}enabled{RESET} ({os.environ.get('OPENAI_MODEL', '?')})"
        except Exception as e:
            llm_status = f"{RED}error: {e}{RESET}"
    else:
        llm_status = f"{DIM}off — use --llm to enable real LLM responses{RESET}"

    info("Neo4j", f"{GREEN}Connected{RESET} ({NEO4J_URI})")
    info("PSS API", pss_status)
    info("LLM", llm_status)
    info("Neo4j Browser", f"http://localhost:7474")

    scenarios = {
        "1": ("Live Drift Detection — type messages, watch drift in real-time", scenario_live_drift),
        "2": ("Drift + Short-Circuit — Morrison diabetes → Patel psychiatry → Rodriguez cardiology → paraphrase HITs", scenario_topic_switch),
        "3": ("Ward Round Cluster — Dr. Chen/Volkov/Tanaka on David Park (PSS Layer 2 Clusters)", scenario_ward_round),
        "4": ("Medication Safety Guard — Gutierrez chemo, anchors/triggers/isolation (PSS Layer 1b)", scenario_medication_safety),
        "5": ("Hospital Network Consensus — Rodriguez/Morrison, Region + Observer (PSS Layers 3+4)", scenario_hospital_consensus),
        "6": ("Shift Handoff — Volkov night → Tanaka day, export/import/transfer (PSS Layer 5)", scenario_shift_handoff),
        "7": ("Neo4j Cypher Explorer — run queries directly against the graph", scenario_explorer),
    }

    while True:
        print(f"\n{CYAN}{'─' * 70}{RESET}")
        print(f"  {BOLD}Choose a scenario:{RESET}\n")
        for key, (desc, _) in scenarios.items():
            print(f"    {CYAN}{key}{RESET}  {desc}")
        print(f"    {CYAN}q{RESET}  Quit\n")

        try:
            choice = input(f"  {CYAN}>{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "q" or choice == "quit":
            break

        if choice in scenarios:
            try:
                scenarios[choice][1](mcp, driver)
            except KeyboardInterrupt:
                print(f"\n  {YELLOW}Interrupted{RESET}")
            except Exception as e:
                print(f"\n  {RED}Error: {e}{RESET}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  {RED}Unknown choice: {choice}{RESET}")

    print(f"\n  {DIM}Closing connections...{RESET}")
    driver.close()
    print(f"  {GREEN}Done!{RESET}\n")


if __name__ == "__main__":
    main()
