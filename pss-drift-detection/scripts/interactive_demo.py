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

        drift = result["drift_score"]
        pss_drift = result.get("pss_drift_score", drift)
        phase = result["drift_phase"]
        severity = result["severity"]
        sim = result["top_similarity"]
        sim_drop = result.get("similarity_drop", 0.0)

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
        info("Composite Score", f"{color_drift(drift)}  {bar(drift)}")
        info("PSS Drift Score", f"{pss_drift:.3f}")
        info("Similarity Drop", f"{color_drift(sim_drop)}  (1 - {sim:.3f})")
        info("Phase", color_phase(phase))
        info("Severity", severity)
        info("Drift Detected", f"{RED}YES{RESET}" if result.get("drift_detected") else f"{GREEN}no{RESET}")
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
    print(f"    {RED}Phase 2{RESET}  Supply chain logistics — maximum semantic distance")
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
    # Phase 2: MAXIMUM semantic distance — supply chain logistics (not healthcare)
    off_topic = [
        "Which tier-1 suppliers are within two hops of the production bottleneck?",
        "Find alternative shipping routes if the Rotterdam port is closed",
        "List all components that depend on the sole-source semiconductor supplier in Taiwan",
        "Show inventory-level shortfalls cascading from the chip shortage",
        "Which distribution centers report the highest average shipment delay?",
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
        + ["a supply chain logistics analyst"] * len(off_topic)
        + ["a cardiologist reviewing Maria Rodriguez's post-MI care"] * len(on_topic_2)
    )

    # Legend
    subheader("How to read the output")
    print(f"  Each step shows a {BOLD}similarity bar{RESET} — PSS top_similarity (how well the query")
    print(f"  matches the accumulated context).  High = on-topic, low = drifted.\n")
    print(f"  {GREEN}{'█' * 5}{RESET} = on-topic (sim > 0.4)     "
          f"{YELLOW}{'█' * 5}{RESET} = shifting (0.2-0.4)     "
          f"{RED}{'█' * 5}{RESET} = drifted (< 0.2)")
    print(f"  {BOLD}DRIFT{RESET} = similarity dropped > 0.25 below rolling avg (last 4 steps)")
    print(f"        Detection starts at step 4 (PSS needs context first).")
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
        # Phase 2: Supply chain (no healthcare entities)
        [], [], [], [], [],
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
            phase_names = {"ON-TOPIC": "Phase 1: Morrison Diabetes", "OFF-TOPIC": "Phase 2: Supply Chain", "NEW-TOPIC": "Phase 3: Rodriguez Cardiology"}
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
        drop = result.get("similarity_drop", 0.0)
        ravg = result.get("rolling_avg", 0.0)
        detected = result.get("drift_detected", False)
        drift_flag = f"  {RED}{BOLD}DRIFT{RESET} (avg={ravg:.2f}, drop={drop:.2f})" if detected else ""
        if detected:
            drift_events_count += 1

        # Main output: step, similarity bar, similarity value, message
        print(f"  {i:>2}  {_sim_bar(sim)} sim={sim:.2f}  {textwrap.shorten(msg, 52)}{drift_flag}")
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

    print(f"  Steps:  {total_steps} total  ({n1} diabetes + {n2} supply chain + {n3} cardiology + {len(short_circuit_queries)} paraphrase)")
    print(f"  Drift:  {drift_events_count} events stored in Neo4j")
    print(f"  HITs:   {sc_hits}/{len(short_circuit_queries)} paraphrases recognized  (threshold: sim >= {sc_threshold})")
    print(f"  LLM:    {total_steps - sc_hits} calls made, {sc_hits} skipped\n")

    print(f"  {BOLD}Drift detection:{RESET}")
    print(f"    Signal:    PSS top_similarity (cosine between query and accumulated context)")
    print(f"    Baseline:  rolling average of last 4 similarity values")
    print(f"    Trigger:   current sim drops > 0.25 below rolling avg (after step 4)")

    phase_info = mcp.get_phase(neo4j_sid)
    print(f"\n  Final phase: {color_phase(phase_info.get('phase', 'N/A') or 'N/A')}")

    mcp.end_pss_session(neo4j_sid)


# ── Scenario 3: Multi-Agent Comparison ──────────────────────────────────

def scenario_multi_agent(mcp: PSSMCPServer, driver):
    header("Scenario 3: Multi-Agent Parallel Comparison")
    print("  Three agents process messages simultaneously.")
    print("  Each has different behavior — Neo4j lets us compare them.\n")

    agents = {
        "focused-agent": [
            "Analyze Q3 revenue trends",
            "Break down Q3 revenue by product line",
            "Compare Q3 margins across regions",
            "Q3 revenue forecast accuracy assessment",
            "Q3 revenue risk factors analysis",
        ],
        "exploring-agent": [
            "Analyze Q3 revenue trends",
            "What about customer satisfaction scores?",
            "How is employee retention this quarter?",
            "Are there any supply chain disruptions?",
            "What's the competitor landscape looking like?",
        ],
        "chaotic-agent": [
            "Analyze Q3 revenue trends",
            "Can you write me a poem about summer?",
            "What's the weather in Tokyo right now?",
            "How do neural networks learn features?",
            "Best pizza recipe for a party of 20?",
        ],
    }

    sessions = {}
    for agent_id in agents:
        s = mcp.create_pss_session(agent_id=agent_id)
        sessions[agent_id] = s["session_id"]

    agent_roles = {
        "focused-agent": "a financial analyst specializing in quarterly revenue",
        "exploring-agent": "a business strategy consultant",
        "chaotic-agent": "a general-purpose assistant",
    }

    # Process all messages
    results = {a: [] for a in agents}
    max_steps = max(len(msgs) for msgs in agents.values())

    for step in range(max_steps):
        for agent_id, msgs in agents.items():
            if step < len(msgs):
                r = mcp.detect_drift(sessions[agent_id], msgs[step])
                response = generate_response(
                    msgs[step],
                    agent_role=agent_roles[agent_id],
                    pss_context=r.get("context", ""),
                )
                mcp.store_response(sessions[agent_id], response)
                results[agent_id].append(r)
                if USE_LLM:
                    print(f"  {DIM}[{agent_id}]{RESET} {textwrap.shorten(response, 70)}")

    # Comparison table
    subheader("Per-Step Drift Comparison")
    print(f"  {'Step':>4}  ", end="")
    for a in agents:
        print(f"{a:>18}  ", end="")
    print()
    print(f"  {'─'*4}  " + "  ".join("─" * 18 for _ in agents))

    for step in range(max_steps):
        print(f"  {step:>4}  ", end="")
        for agent_id in agents:
            if step < len(results[agent_id]):
                r = results[agent_id][step]
                d = r["drift_score"]
                det = "*" if r.get("drift_detected") else " "
                print(f"{color_drift(d):>28}{det} ", end="")
            else:
                print(f"{'—':>18}  ", end="")
        print()
    print(f"\n  {DIM}* = drift detected (topic switch){RESET}")

    # Final state comparison from Neo4j
    subheader("Final Agent States (from Neo4j)")
    print(f"  {'Agent':>18}  {'Phase':>14}  {'Steps':>5}  {'Drift Events':>12}  {'Stability':>10}")
    print(f"  {'─'*18}  {'─'*14}  {'─'*5}  {'─'*12}  {'─'*10}")

    state_store = Neo4jStateStore(driver, database=NEO4J_DATABASE)
    trajectory_analyzer = TrajectoryAnalyzer(state_store)

    for agent_id in agents:
        sid = sessions[agent_id]
        phase = mcp.get_phase(sid)
        events = mcp.query_drift_history(sid)
        stability = trajectory_analyzer.compute_stability(sid)
        traj = mcp.get_state_trajectory(sid, steps=50)

        phase_str = color_phase(phase.get("phase", "N/A"))
        print(f"  {agent_id:>18}  {phase_str:>24}  {len(traj):>5}  {len(events):>12}  {stability:>10.3f}")

    # Influence analysis
    subheader("Influence Analysis (Neo4j PageRank-like)")
    influence = InfluenceAnalyzer(driver, database=NEO4J_DATABASE)
    scores = influence.compute_influence_scores(list(sessions.values()))
    for sid, score in sorted(scores, key=lambda x: -x[1]):
        agent_name = [a for a, s in sessions.items() if s == sid][0]
        bar_str = bar(score * len(scores), width=25)
        print(f"  {agent_name:>18}  {score:.3f}  {bar_str}")

    for agent_id in agents:
        mcp.end_pss_session(sessions[agent_id])


# ── Scenario 4: Memory & Recall ─────────────────────────────────────────

def scenario_memory(mcp: PSSMCPServer, driver):
    header("Scenario 4: Memory & Recall System")
    print("  Shows PSS's multi-tier memory with Neo4j vector search.\n")

    session = mcp.create_pss_session(agent_id="memory-demo")
    sid = session["session_id"]

    # Store memories across tiers with embeddings
    memories = [
        ("Patient John Smith has penicillin allergy — confirmed anaphylaxis risk", 0.95, "short"),
        ("Heart failure protocol updated: first-line is now sacubitril/valsartan", 0.85, "short"),
        ("Lab reference range for BNP changed to 0-100 pg/mL per 2024 guidelines", 0.80, "short"),
        ("Dr. Garcia specializes in interventional cardiology, available MWF", 0.70, "medium"),
        ("Facility B has highest cardiac surgery volume in the network", 0.65, "medium"),
        ("Insurance pre-auth required for CT angiography — typical wait 2-3 days", 0.60, "medium"),
        ("Hospital accreditation review scheduled for next quarter", 0.40, "long"),
        ("Annual training on infection control completed last month", 0.30, "long"),
    ]

    subheader("Storing Memories")
    for text, importance, tier in memories:
        vec = _pseudo_embed(text)
        result = mcp.store_memory(sid, text, importance=importance, vector=vec, tier=tier)
        tier_color = {"short": YELLOW, "medium": CYAN, "long": DIM}[tier]
        print(f"  {tier_color}[{tier:>6}]{RESET}  imp={importance:.2f}  {textwrap.shorten(text, 60)}")

    # Query by similarity
    subheader("Similarity Search: 'cardiac patient medication allergy'")
    query_vec = _pseudo_embed("cardiac patient medication allergy")
    results = mcp.memory_query(sid, query_vec, limit=5)
    for i, r in enumerate(results):
        print(f"  {i+1}. {CYAN}sim={r['similarity']:.3f}{RESET}  [{r['tier']}]  {textwrap.shorten(r['text'], 55)}")

    subheader("Similarity Search: 'hospital operations and scheduling'")
    query_vec2 = _pseudo_embed("hospital operations and scheduling")
    results2 = mcp.memory_query(sid, query_vec2, limit=5)
    for i, r in enumerate(results2):
        print(f"  {i+1}. {CYAN}sim={r['similarity']:.3f}{RESET}  [{r['tier']}]  {textwrap.shorten(r['text'], 55)}")

    # Consolidation
    subheader("Memory Consolidation")
    consol = mcp.memory_consolidate(sid)
    print(f"  Consolidated: {consol['consolidated']} memories promoted")
    print(f"  Short: {consol['short']}  |  Medium: {consol['medium']}  |  Long: {consol['long']}")

    # Neo4j view
    subheader("Neo4j Memory Graph")
    with driver.session(database=NEO4J_DATABASE) as db:
        result = db.run("""
            MATCH (s:AgentSession {session_id: $sid})-[:HAS_MEMORY]->(m:Memory)
            RETURN m.tier AS tier, count(m) AS count,
                   avg(m.importance) AS avg_importance,
                   collect(m.text_summary)[0..2] AS samples
            ORDER BY tier
        """, sid=sid)
        for r in result:
            print(f"  {r['tier']:>8}  count={r['count']}  avg_imp={r['avg_importance']:.2f}  "
                  f"samples={r['samples']}")

    mcp.end_pss_session(sid)


# ── Scenario 5: Cross-Session Analytics ──────────────────────────────────

def scenario_analytics(mcp: PSSMCPServer, driver):
    header("Scenario 5: Cross-Session Analytics")
    print("  Uses Neo4j graph algorithms across all existing sessions.\n")

    # Get all active sessions from Neo4j
    with driver.session(database=NEO4J_DATABASE) as db:
        result = db.run("""
            MATCH (s:AgentSession)
            WHERE s.status IN ['active', 'closed']
            OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)
            OPTIONAL MATCH (s)-[:CURRENT_PHASE]->(p:Phase)
            RETURN s.session_id AS sid, s.agent_id AS agent,
                   s.status AS status, st.step AS steps, p.name AS phase
            ORDER BY s.agent_id
        """)
        sessions = [dict(r) for r in result]

    if not sessions:
        print(f"  {RED}No sessions found in Neo4j. Run the seed script first.{RESET}")
        return

    subheader(f"Found {len(sessions)} Sessions in Neo4j")
    for s in sessions:
        print(f"  {s['agent']:>25}  phase={color_phase(s.get('phase', 'N/A') or 'N/A'):>24}  "
              f"steps={s.get('steps', 0) or 0}  status={s['status']}")

    # Influence Analysis
    subheader("Influence Scores (PageRank-like, pure Cypher)")
    influence = InfluenceAnalyzer(driver, database=NEO4J_DATABASE)
    sids = [s["sid"] for s in sessions]
    scores = influence.compute_influence_scores(sids)
    for sid, score in sorted(scores, key=lambda x: -x[1]):
        agent = next((s["agent"] for s in sessions if s["sid"] == sid), sid[:12])
        bar_str = bar(score * len(scores), width=30)
        print(f"  {agent:>25}  {score:.4f}  {bar_str}")

    # Trajectory Stability
    subheader("Trajectory Stability (from Neo4j state chains)")
    state_store = Neo4jStateStore(driver, database=NEO4J_DATABASE)
    trajectory_analyzer = TrajectoryAnalyzer(state_store)
    stabilities = []
    for s in sessions:
        stability = trajectory_analyzer.compute_stability(s["sid"])
        stabilities.append((s["agent"], stability))
    for agent, stab in sorted(stabilities, key=lambda x: -x[1]):
        color = GREEN if stab > 0.7 else (YELLOW if stab > 0.4 else RED)
        print(f"  {agent:>25}  {color}{stab:.3f}{RESET}  {bar(stab)}")

    # Drift Points Detection
    subheader("Drift Points Detected (cosine drops > 0.3)")
    for s in sessions:
        points = trajectory_analyzer.find_drift_points(s["sid"], threshold=0.3)
        if points:
            for p in points:
                print(f"  {s['agent']:>25}  step={p['step']}  "
                      f"cosine={p['cosine_similarity']:.3f}  drop={p['drop']:.3f}")

    # Phase Transition Matrix
    subheader("Phase Transition Matrix (Neo4j Markov)")
    with driver.session(database=NEO4J_DATABASE) as db:
        result = db.run("""
            MATCH (p1:Phase)-[:TRANSITIONED_TO]->(p2:Phase)
            RETURN p1.name AS from_phase, p2.name AS to_phase, count(*) AS count
            ORDER BY count DESC
        """)
        transitions = list(result)
    if transitions:
        print(f"  {'From':>15}  →  {'To':>15}  {'Count':>5}")
        print(f"  {'─'*15}     {'─'*15}  {'─'*5}")
        for t in transitions:
            print(f"  {t['from_phase']:>15}  →  {t['to_phase']:>15}  {t['count']:>5}")
    else:
        print(f"  {DIM}No phase transitions recorded{RESET}")

    # Node/Relationship counts
    subheader("Neo4j Database Stats")
    with driver.session(database=NEO4J_DATABASE) as db:
        result = db.run("""
            MATCH (n)
            RETURN labels(n)[0] AS type, count(n) AS count
            ORDER BY count DESC
        """)
        for r in result:
            print(f"  {r['type']:>20}  {r['count']:>5} nodes")

        result = db.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS count
            ORDER BY count DESC
        """)
        print()
        for r in result:
            print(f"  {r['type']:>20}  {r['count']:>5} relationships")


# ── Scenario 6: Cluster Coupling & Knowledge Transfer (Layer 2) ─────────

def scenario_transfer(mcp: PSSMCPServer, driver):
    header("Scenario 6: Cluster Coupling & Knowledge Transfer (Layer 2)")
    print("  Expert seeds a shared cluster. Novice joins and gets HITs from")
    print("  the expert's cached knowledge — no LLM call needed.\n")

    pss = PSSClient()

    # Step 1: Create shared cluster
    subheader("Step 1: Create shared research cluster")
    try:
        cluster = pss.create_cluster("cardiology-team", aggregation_mode="weighted_average", coupling_factor=0.25)
        cid = cluster["cluster_id"]
        info("Cluster ID", cid)
        info("Coupling", "0.25 (moderate knowledge sharing)")
    except Exception as e:
        print(f"  {RED}Cluster creation failed: {e}{RESET}")
        return

    # Step 2: Expert seeds the cluster with cardiology Q&A
    # Expert also creates a personal PSS session (needed for coupling feedback)
    subheader("Step 2: Expert seeds the cluster with cardiology knowledge")
    expert_pss_sid = None
    expert_msgs = [
        ("What are the first-line treatments for heart failure with reduced ejection fraction?",
         "a senior cardiologist"),
        ("How do beta-blockers compare to ACE inhibitors after myocardial infarction?",
         "a senior cardiologist"),
        ("What anticoagulation strategy works best for atrial fibrillation patients?",
         "a senior cardiologist"),
        ("What are the current guidelines for cardiac biomarker interpretation in ACS?",
         "a senior cardiologist"),
    ]

    HIT_THRESHOLD = 0.52  # tuned: Q+A embeddings land ~0.55-0.80 for paraphrases

    for msg, role in expert_msgs:
        # Run against cluster (threshold=0.99 → always MISS during seeding)
        cdata = pss.cluster_run(cid, msg, short_circuit_threshold=0.99)
        response = generate_response(msg, agent_role=role, pss_context=cdata.get("context", ""))
        # Store Q+A in cluster
        pss.cluster_store(cid, msg, response)
        # Also run on expert's personal session (for coupling later)
        expert_data = pss.run(msg, session_id=expert_pss_sid)
        expert_pss_sid = expert_data["session_id"]
        sim = cdata.get("top_similarity", 0.0)
        print(f"  {RED}MISS{RESET}  sim={sim:.3f}  → LLM called")
        print(f"    Q: {textwrap.shorten(msg, 65)}")
        if USE_LLM:
            print(f"    A: {textwrap.shorten(response, 70)}")
        print()

    # Register expert as cluster member
    pss.add_cluster_member(cid, expert_pss_sid)
    info("Expert session", f"{expert_pss_sid[:16]}... added to cluster")

    # Step 3: Novice creates personal session, asks paraphrased questions
    subheader("Step 3: Novice asks — paraphrases should HIT from cluster cache")
    novice_pss_sid = None
    novice_queries = [
        ("Which drugs are recommended first for HFrEF patients?",
         "Paraphrase of expert Q1"),
        ("After a heart attack, should we use beta-blockers or ACE inhibitors?",
         "Paraphrase of expert Q2"),
        ("What is the best approach to anticoagulation in afib?",
         "Paraphrase of expert Q3"),
        ("How should we interpret troponin levels in acute coronary syndrome?",
         "Paraphrase of expert Q4"),
        ("What is the best recipe for chocolate chip cookies?",
         "Novel — completely off-domain"),
    ]

    hits = 0
    total = len(novice_queries)
    for msg, label in novice_queries:
        cdata = pss.cluster_run(cid, msg, short_circuit_threshold=HIT_THRESHOLD)
        sim = cdata.get("top_similarity", 0.0)
        sc = cdata.get("short_circuit", False)
        ctx = cdata.get("context", "")

        # Also run on novice's personal session
        novice_data = pss.run(msg, session_id=novice_pss_sid)
        novice_pss_sid = novice_data["session_id"]

        if sc:
            hits += 1
            print(f"  {GREEN}HIT{RESET}   sim={sim:.3f}  → LLM skipped ({label})")
            print(f"    Q: {textwrap.shorten(msg, 65)}")
        else:
            response = generate_response(msg, agent_role="a medical intern", pss_context=ctx)
            pss.cluster_store(cid, msg, response)
            print(f"  {RED}MISS{RESET}  sim={sim:.3f}  → LLM called  ({label})")
            print(f"    Q: {textwrap.shorten(msg, 65)}")
            if USE_LLM:
                print(f"    A: {textwrap.shorten(response, 70)}")
        print()

    # Register novice as cluster member
    pss.add_cluster_member(cid, novice_pss_sid)
    info("Novice session", f"{novice_pss_sid[:16]}... added to cluster")

    # Step 4: Apply coupling feedback — blend cluster vector into both sessions
    subheader("Step 4: Apply coupling feedback (G4)")
    try:
        feedback = pss.cluster_feedback(cid)
        sessions_updated = feedback.get("sessions_updated", 0)
        info("Sessions coupled", str(sessions_updated))
        if sessions_updated > 0:
            info("Effect", f"Cluster vector blended into {sessions_updated} member sessions (α=0.25)")
        else:
            info("Effect", f"{DIM}No sessions with initialised vectors to couple{RESET}")
    except Exception as e:
        print(f"  {YELLOW}Feedback: {e}{RESET}")

    # Summary
    subheader("Cluster Efficiency Summary")
    info("Total queries", str(total))
    info("Cluster HITs", f"{hits}/{total}  ({100*hits//total}% LLM cost saved)")
    info("LLM calls made", str(total - hits))
    info("Members", f"expert ({expert_pss_sid[:12]}...) + novice ({novice_pss_sid[:12]}...)")

    # Persist summary to Neo4j
    s = mcp.create_pss_session(agent_id="cluster-transfer-demo")
    for msg, _ in novice_queries[:2]:
        mcp.detect_drift(s["session_id"], msg)
    mcp.end_pss_session(s["session_id"])

    # Cleanup
    try:
        pss.delete_cluster(cid)
    except Exception:
        pass




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
        "2": ("Drift + Short-Circuit — Morrison diabetes → supply chain → Rodriguez cardiology → paraphrase HITs", scenario_topic_switch),
        "3": ("Multi-Agent Comparison — focused vs exploring vs chaotic agents", scenario_multi_agent),
        "4": ("Memory & Recall — store, search, consolidate multi-tier memories", scenario_memory),
        "5": ("Cross-Session Analytics — influence, stability, phase transitions", scenario_analytics),
        "6": ("Cluster Coupling — expert seeds cluster, novice gets HITs (Layer 2)", scenario_transfer),
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
