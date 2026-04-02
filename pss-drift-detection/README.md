# PSS + Neo4j: Semantic Memory Meets Knowledge Graphs

**What happens when you give an AI agent infinite memory AND a knowledge graph?**

PSS (Persistent Semantic State) compresses any conversation into a fixed-size context block — O(1) memory per turn, no matter how long the conversation. Neo4j stores the structured world the agent operates in: patients, medications, providers, diagnoses, treatments, facilities, and all the relationships between them.

Alone, each is powerful. Together, they unlock capabilities that neither can achieve independently.

---

## Why PSS + Neo4j?

### PSS alone knows *what was said* — but not *what exists*

PSS tracks semantic drift, detects topic switches, caches repeated queries, and manages multi-agent coordination. But it operates on unstructured text. It doesn't know that Metformin is contraindicated with renal failure, or that Dr. Volkov works at Memorial General.

### Neo4j alone knows *what exists* — but not *what was discussed*

Neo4j holds the complete healthcare knowledge graph: 5 patients, 5 providers, 5 diagnoses, 5 medications, their relationships (DIAGNOSED_WITH, PRESCRIBED, CONTRAINDICATED_WITH, REFERRED_TO). But it has no memory of which agent asked what, when they drifted off-topic, or whether a question was already answered.

### Together: agents that *remember* AND *reason over structure*

| Capability | PSS | Neo4j | PSS + Neo4j |
|---|:---:|:---:|:---:|
| Detect when agent switches from diabetes to psychiatry | yes | — | yes + graph shows which entities were investigated before/after drift |
| Know that Metformin is contraindicated with Amoxicillin | — | yes | yes + PSS injects this as synthetic memory, agent recalls it in conversation |
| Skip LLM call when a paraphrased question was already answered | yes | — | yes + Neo4j shows which healthcare entity the cached answer relates to |
| Three specialists share findings on the same patient | partial | — | yes: PSS cluster propagates HITs, Neo4j shows who investigated what |
| Detect that two hospital departments drifted simultaneously | — | — | yes: PSS regions detect consensus drift, Neo4j stores the topology |
| Transfer night-shift knowledge to day-shift doctor | partial | — | yes: PSS export/import + delta transfer, Neo4j traces which provider investigated which patient at which shift |

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            Your LLM Agent               │
                    └─────────┬───────────────┬───────────────┘
                              │               │
                    ┌─────────▼─────────┐   ┌─▼───────────────┐
                    │   PSS API         │   │   Neo4j          │
                    │                   │   │                   │
                    │ • /run → context  │   │ • Healthcare KG   │
                    │ • drift_detected  │   │   (patients,      │
                    │ • short_circuit   │   │    medications,    │
                    │ • cluster HITs    │   │    diagnoses...)   │
                    │ • region events   │   │                   │
                    │ • observer alerts │   │ • PSS state chain  │
                    │ • export/import   │   │   (drift events,   │
                    │ • anchors/triggers│   │    phases,          │
                    │                   │   │    INVESTIGATED)    │
                    └───────────────────┘   └───────────────────┘
```

PSS does all computation (drift scoring, context compression, short-circuit detection, multi-agent coordination). Neo4j persists the results and connects them to the structured knowledge graph. The `INVESTIGATED` relationship bridges both worlds: it links an agent session to the healthcare entities it queried, with the drift score at the time of investigation.

---

## What the Demo Shows

### Scenario 2: Drift Detection + Short-Circuit

An agent investigates **James Morrison** (Type 2 Diabetes, Metformin, Lisinopril, Atorvastatin) across 17 queries. Then switches to **Aisha Patel** (Major Depressive Disorder, Sertraline, CBT). PSS detects the specialty switch (`drift_score=0.44, phase=drifted`). Returns to **Maria Rodriguez** (AMI, cardiology). Finally, paraphrases of Phase 1 queries trigger short-circuit — LLM skipped, cached context returned.

**PSS signals**: `drift_detected`, `drift_score`, `drift_phase`, `top_similarity`, `short_circuit`
**Neo4j graph**: Agent → INVESTIGATED → Patient/Diagnosis/Medication + DriftEvent nodes at pivot points

### Scenario 3: Multi-Specialist Ward Round (Layer 2 — Clusters)

Three doctors examine **David Park** (Essential Hypertension + COPD):
- **Dr. Chen** (internist) seeds a shared PSS cluster with 4 baseline Q&A pairs
- **Dr. Volkov** (pulmonologist) queries the cluster → gets HITs from Chen's findings (LLM skipped)
- **Dr. Tanaka** (cardiologist) stores a novel ECG finding → Volkov re-queries and gets it

**PSS signals**: `cluster_run`, `short_circuit`, `cluster_store`, `cluster_feedback`
**Neo4j graph**: 3 Agents → MEMBER_OF → Cluster + INVESTIGATED → Patient/Diagnosis/Medication

### Scenario 4: Medication Safety Guard (Layer 1b — Anchors, Triggers, Isolation)

An oncology pharmacist monitors **Carlos Gutierrez** (Chemotherapy Cycle 1):
- **Drift Anchor** locks session to oncology domain (real 384-dim embedding via sentence-transformers)
- **Resonance Triggers** on "CRITICAL" and "contraindication" keywords
- **Synthetic Memory Injection**: contraindications from Neo4j graph (Amoxicillin↔Metformin, Atorvastatin↔Sertraline) injected into PSS memory tiers
- **Input Isolation**: QUARANTINE mode filters off-topic queries
- **Anchor Score**: measures how far the session drifted from the oncology domain

**PSS signals**: `/session/create`, `/anchor`, `/anchor_score`, `/trigger`, `/isolation`, `/memory`
**Neo4j graph**: Agent → INVESTIGATED → Patient/Treatment/Medication + Memory nodes with contraindication knowledge

### Scenario 5: Hospital Network Consensus (Layers 3+4 — Regions, Observer)

Two department clusters across two facilities:
- **Cardiology** at Memorial General → Maria Rodriguez (AMI)
- **Emergency** at Riverside Medical → James Morrison (ER visit)

Both clusters are in a PSS **Region** (`consensus_threshold=0.5`). A **Global Observer** monitors for cross-cluster anomalies. When both departments drift to admin topics simultaneously, the Region detects 2 drift events and the Observer samples the anomaly.

**PSS signals**: `/region/create`, `/region/events`, `/observer/create`, `/observer/sample`, `/observer/anomalies`
**Neo4j graph**: Region → CONTAINS_CLUSTER → Clusters → MEMBER_OF ← Agents → INVESTIGATED → Patient/Facility

### Scenario 6: Shift Handoff (Layer 5 — Export/Import, Network Transfer)

Night shift **Dr. Volkov** monitors Morrison overnight (6 queries, builds deep PSS context). She exports her session state (SHA-256 checksum). Day shift **Dr. Tanaka** imports the state + receives a semantic delta transfer. Tanaka's first query immediately shows `sim=0.578` (vs `0.000` baseline without import) — full context continuity across shift change.

**PSS signals**: `/session/export`, `/session/import`, `/network/transfer`
**Neo4j graph**: Volkov → INVESTIGATED → Morrison (night) + Tanaka → INVESTIGATED → Morrison (day)

---

## The INVESTIGATED Relationship

This is where PSS and Neo4j truly merge. Every time an agent queries about a healthcare entity, an `INVESTIGATED` relationship is created:

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
- **PSS API key** (get one at [pss.versino.de](https://pss.versino.de))
- **Python 3.10+** with GPU recommended for embeddings (CPU works too)
- **OpenAI-compatible LLM endpoint** for agent responses

### Setup

```bash
cd pss-drift-detection
cp .env.example .env          # add your PSS_API_KEY, OPENAI_BASE_URL, OPENAI_API_KEY

uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Generate healthcare test data
uvx create-context-graph pss-test-data --domain healthcare --framework pydanticai --demo-data
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

---

## PSS API Coverage

| Layer | Feature | Scenario |
|---|---|---|
| **1** | `/run`, `/store`, inline-store, drift detection, short-circuit | 1, 2, 3, 5, 6 |
| **1b** | `/session/create`, `/anchor`, `/trigger`, `/isolation`, `/memory` | 4 |
| **2** | `/cluster/*` (create, run, store, feedback, members) | 3, 5 |
| **3** | `/region/*` (create, clusters, events, consensus) | 5 |
| **4** | `/observer/*` (create, sample, anomalies, summary) | 5 |
| **5** | `/session/export`, `/session/import`, `/network/transfer` | 6 |

---

## Neo4j Schema

```
PSS Layer:    AgentSession → SemanticState → DriftEvent
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
pss-drift-detection/
  src/
    core/
      pss_client.py          # PSS API client (Layers 1-5)
      drift_detector.py      # Thin bridge: PSS → Neo4j (no heuristics)
    mcp/
      pss_mcp_server.py      # MCP tools for agent frameworks
    persistence/             # Neo4j stores (session, state, phase, drift, memory, cluster, region)
    analytics/               # Similarity, influence, trajectories
  scripts/
    interactive_demo.py      # 7 scenarios showcasing PSS × Neo4j
    seed_test_data.py        # Bulk seed healthcare + PSS data
  schema/
    pss_schema.cypher        # Neo4j constraints + vector indexes
  tests/                     # 148 tests (pytest + real PSS API + Neo4j)
  docs/
    API.md                   # Full PSS API documentation
```

---

## License

Copyright (c) 2025 Michael Neuberger — All Rights Reserved.
