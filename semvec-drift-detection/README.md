# Semvec + Neo4j: Semantic Memory Meets Knowledge Graphs

**What happens when you give an AI agent infinite memory AND a knowledge graph?**

[Semvec](https://pypi.org/project/semvec/) compresses any conversation into a fixed-size context block — O(1) memory per turn, no matter how long the conversation. It runs entirely in-process: no API key, no network round-trips. Neo4j stores the structured world the agent operates in: patients, medications, providers, diagnoses, treatments, facilities, and all the relationships between them.

Alone, each is powerful. Together, they unlock capabilities that neither can achieve independently.

---

## Why Semvec + Neo4j?

### Semvec alone knows *what was said* — but not *what exists*

Semvec tracks semantic drift, detects topic switches, caches repeated queries, and manages multi-agent coordination. But it operates on unstructured text. It doesn't know that Metformin is contraindicated with renal failure, or that Dr. Volkov works at Memorial General.

### Neo4j alone knows *what exists* — but not *what was discussed*

Neo4j holds the complete healthcare knowledge graph: 5 patients, 5 providers, 5 diagnoses, 5 medications, their relationships (DIAGNOSED_WITH, PRESCRIBED, CONTRAINDICATED_WITH, REFERRED_TO). But it has no memory of which agent asked what, when they drifted off-topic, or whether a question was already answered.

### Together: agents that *remember* AND *reason over structure*

| Capability | Semvec | Neo4j | Semvec + Neo4j |
|---|:---:|:---:|:---:|
| Detect when agent switches from diabetes to psychiatry | yes | — | yes + graph shows which entities were investigated before/after drift |
| Know that Metformin is contraindicated with Amoxicillin | — | yes | yes + Semvec injects this as synthetic memory, agent recalls it in conversation |
| Skip LLM call when a paraphrased question was already answered | yes | — | yes + Neo4j shows which healthcare entity the cached answer relates to |
| Three specialists share findings on the same patient | partial | — | yes: Semvec cluster propagates HITs, Neo4j shows who investigated what |
| Detect that two hospital departments drifted simultaneously | — | — | yes: Semvec regions detect consensus drift, Neo4j stores the topology |
| Transfer night-shift knowledge to day-shift doctor | partial | — | yes: Semvec export/import + delta transfer, Neo4j traces which provider investigated which patient at which shift |

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            Your LLM Agent               │
                    └─────────┬───────────────┬───────────────┘
                              │               │
                    ┌─────────▼─────────┐   ┌─▼─────────────────┐
                    │  Semvec runtime   │   │   Neo4j            │
                    │  (in-process)     │   │                    │
                    │                   │   │ • Healthcare KG    │
                    │ • run → context   │   │   (patients,       │
                    │ • drift_detected  │   │    medications,    │
                    │ • short_circuit   │   │    diagnoses...)   │
                    │ • cluster HITs    │   │                    │
                    │ • region events   │   │ • Semvec state     │
                    │ • observer alerts │   │   chain (drift     │
                    │ • export/import   │   │   events, phases,  │
                    │ • anchors/triggers│   │   INVESTIGATED)    │
                    └───────────────────┘   └────────────────────┘
```

Semvec does all computation (drift scoring, context compression, short-circuit detection, multi-agent coordination) inside your Python process via `pip install semvec`. Neo4j persists the results and connects them to the structured knowledge graph. The `INVESTIGATED` relationship bridges both worlds: it links an agent session to the healthcare entities it queried, with the drift score at the time of investigation.

---

## What the Demo Shows

### Scenario 2: Drift Detection + Short-Circuit

An agent investigates **James Morrison** (Type 2 Diabetes, Metformin, Lisinopril, Atorvastatin) across 17 queries. Then switches to **Aisha Patel** (Major Depressive Disorder, Sertraline, CBT). Semvec detects the specialty switch (`drift_score=0.44, phase=drifted`). Returns to **Maria Rodriguez** (AMI, cardiology). Finally, paraphrases of Phase 1 queries trigger short-circuit — LLM skipped, cached context returned.

**Semvec signals**: `drift_detected`, `drift_score`, `drift_phase`, `top_similarity`, `short_circuit`
**Neo4j graph**: Agent → INVESTIGATED → Patient/Diagnosis/Medication + DriftEvent nodes at pivot points

### Scenario 3: Multi-Specialist Ward Round (Layer 2 — Clusters)

Three doctors examine **David Park** (Essential Hypertension + COPD):
- **Dr. Chen** (internist) seeds a shared Semvec cluster with 4 baseline Q&A pairs
- **Dr. Volkov** (pulmonologist) queries the cluster → gets HITs from Chen's findings (LLM skipped)
- **Dr. Tanaka** (cardiologist) stores a novel ECG finding → Volkov re-queries and gets it

**Semvec signals**: `cluster_run`, `short_circuit`, `cluster_store`, `cluster_feedback`
**Neo4j graph**: 3 Agents → MEMBER_OF → Cluster + INVESTIGATED → Patient/Diagnosis/Medication

### Scenario 4: Medication Safety Guard (Layer 1b — Anchors, Triggers, Isolation)

An oncology pharmacist monitors **Carlos Gutierrez** (Chemotherapy Cycle 1):
- **Drift Anchor** locks session to oncology domain (real 384-dim embedding via sentence-transformers)
- **Resonance Triggers** on "CRITICAL" and "contraindication" keywords
- **Synthetic Memory Injection**: contraindications from Neo4j graph (Amoxicillin↔Metformin, Atorvastatin↔Sertraline) injected into Semvec memory tiers
- **Input Isolation**: QUARANTINE mode filters off-topic queries
- **Anchor Score**: measures how far the session drifted from the oncology domain

**Semvec signals**: `create_session`, `add_anchor`, `get_anchor_score`, `add_trigger`, `set_isolation`, `inject_memory`
**Neo4j graph**: Agent → INVESTIGATED → Patient/Treatment/Medication + Memory nodes with contraindication knowledge

### Scenario 5: Hospital Network Consensus (Layers 3+4 — Regions, Observer)

Two department clusters across two facilities:
- **Cardiology** at Memorial General → Maria Rodriguez (AMI)
- **Emergency** at Riverside Medical → James Morrison (ER visit)

Both clusters are in a Semvec **Region** (`consensus_threshold=0.5`). A **Global Observer** monitors for cross-cluster anomalies. When both departments drift to admin topics simultaneously, the Region detects 2 drift events and the Observer samples the anomaly.

**Semvec signals**: `create_region`, `get_region_events`, `create_observer`, `observer_sample`, `get_anomalies`
**Neo4j graph**: Region → CONTAINS_CLUSTER → Clusters → MEMBER_OF ← Agents → INVESTIGATED → Patient/Facility

### Scenario 6: Shift Handoff (Layer 5 — Export/Import, Network Transfer)

Night shift **Dr. Volkov** monitors Morrison overnight (6 queries, builds deep Semvec context). She exports her session state (SHA-256 checksum). Day shift **Dr. Tanaka** imports the state + receives a semantic delta transfer. Tanaka's first query immediately shows `sim=0.578` (vs `0.000` baseline without import) — full context continuity across shift change.

**Semvec signals**: `export_session`, `import_session`, `transfer_delta`
**Neo4j graph**: Volkov → INVESTIGATED → Morrison (night) + Tanaka → INVESTIGATED → Morrison (day)

---

## The INVESTIGATED Relationship

This is where Semvec and Neo4j truly merge. Every time an agent queries about a healthcare entity, an `INVESTIGATED` relationship is created:

```cypher
(agent:AgentSession)-[:INVESTIGATED {step: 5, drift_score: 0.44, phase: "drifted"}]->(entity:Patient)
```

This enables queries that neither system could answer alone:

```cypher
// Which agents investigated Morrison, and did any drift during the investigation?
MATCH (s:AgentSession)-[inv:INVESTIGATED]->(p:Patient {name: 'James Morrison'})
OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)-[:TRIGGERED]->(d:DriftEvent)
RETURN s.agent_id, inv.step, inv.drift_score, d.severity

// Full picture: agent → patient → diagnoses → medications + drift events + cluster membership
MATCH (s:AgentSession)-[inv:INVESTIGATED]->(pat:Patient)
MATCH (pat)-[:DIAGNOSED_WITH]->(diag:Diagnosis)
OPTIONAL MATCH (treat:Treatment)-[:TREATS]->(diag)
OPTIONAL MATCH (treat)-[:USES]->(med:Medication)
OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)-[:TRIGGERED]->(drift:DriftEvent)
OPTIONAL MATCH (s)-[:MEMBER_OF]->(cluster:Cluster)
RETURN s, inv, pat, diag, treat, med, drift, cluster
```

---

## Quick Start

### Prerequisites

- **Neo4j 5.11+** (Docker: `docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5.26`)
- **Python 3.10+** with GPU recommended for embeddings (CPU works too)
- **OpenAI-compatible LLM endpoint** for agent responses *(optional — only needed for `interactive_demo.py` with LLM enabled)*

### Setup

```bash
cd semvec-drift-detection

# 1. Create the Python environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the project (pulls in semvec, neo4j, sentence-transformers, …)
uv pip install -e ".[dev]"     # or: pip install -e ".[dev]"

# 3. Configure environment — every variable below is required.
#    The demo and tests refuse to start with missing values rather
#    than falling back to silent defaults.
cp .env.example .env
$EDITOR .env                   # fill in NEO4J_TEST_PASSWORD + OPENAI_*

# 4. Generate healthcare test data
uvx create-context-graph healthcare-test-data --domain healthcare --framework pydanticai --demo-data
```

### Run the Demo

```bash
python scripts/interactive_demo.py              # LLM enabled (default)
python scripts/interactive_demo.py --no-llm     # skip LLM calls (shows N/A)
```

### Run Tests

```bash
NEO4J_TEST_PASSWORD=password python -m pytest tests/ -v
```

The unit tests for `SemvecClient` and the embedder run fully offline; the
Neo4j-backed integration tests need a running database.

---

## Semvec Coverage

| Layer | Capability | `SemvecClient` methods | Scenario |
|---|---|---|---|
| **1** | run, store, inline-store, drift detection, short-circuit | `run`, `store` | 1, 2, 3, 5, 6 |
| **1b** | session control, anchors, triggers, isolation, memory injection | `create_session`, `add_anchor`, `get_anchor_score`, `add_trigger`, `set_isolation`, `inject_memory` | 4 |
| **2** | cluster CRUD + run/store/feedback/members | `create_cluster`, `cluster_run`, `cluster_store`, `cluster_feedback`, `add_cluster_member`, `remove_cluster_member` | 3, 5 |
| **3** | region CRUD, attached clusters, drift events | `create_region`, `add_region_cluster`, `get_region_events` | 5 |
| **4** | global observer (sample, anomalies, summary) | `create_observer`, `observer_sample`, `get_anomalies`, `get_observer_summary` | 5 |
| **5** | export/import + network delta + consensus + trust | `export_session`, `import_session`, `transfer_delta`, `propose_consensus`, `get_trust_scores` | 6 |

---

## Neo4j Schema

```
Semvec Layer: AgentSession → SemanticState → DriftEvent
              AgentSession → Phase (TRANSITIONED_TO chain)
              AgentSession → Memory (tiered: short/medium/long)
              AgentSession → MEMBER_OF → Cluster
              Region → CONTAINS_CLUSTER → Cluster

Healthcare:   Patient → DIAGNOSED_WITH → Diagnosis
              Patient → TREATED_BY → Provider
              Provider → PRESCRIBED → Medication
              Treatment → TREATS → Diagnosis
              Treatment → USES → Medication
              Medication → CONTRAINDICATED_WITH → Medication
              Provider → REFERRED_TO → Provider
              Provider → AFFILIATED_WITH → Facility
              Patient → HAD_ENCOUNTER → Encounter

Bridge:       AgentSession → INVESTIGATED {step, drift_score, phase} → (any healthcare entity)
```

---

## Project Structure

```
semvec-drift-detection/
  src/
    core/
      semvec_client.py       # In-process Semvec client (Layers 1-5)
      embedder.py            # HashEmbedder (offline) + SentenceTransformerEmbedder
      drift_detector.py      # Thin bridge: Semvec → Neo4j (no heuristics)
    mcp/
      semvec_mcp_server.py      # MCP tools for agent frameworks
    persistence/             # Neo4j stores (session, state, phase, drift, memory, cluster, region)
    analytics/               # Similarity, influence, trajectories
  scripts/
    interactive_demo.py      # 7 scenarios showcasing Semvec × Neo4j
    seed_test_data.py        # Bulk seed healthcare + Semvec data
  schema/
    semvec_schema.cypher        # Neo4j constraints + vector indexes
  tests/                     # pytest suite — unit (offline) + integration (Neo4j)
  docs/
    API.md                   # API surface documentation
```

---

## License

Copyright (c) 2025 Michael Neuberger — All Rights Reserved.
