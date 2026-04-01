# Implementation Plan: Interactive Demo Redesign

## Goal

Rewrite scenarios 3–6 of `scripts/interactive_demo.py` so that every scenario:
1. Uses a **different PSS API layer** (Layer 2, 1b, 3+4, 5)
2. References **real healthcare fixture data** from Neo4j (patients, providers, diagnoses, medications, treatments, encounters, facilities)
3. Creates **INVESTIGATED** relationships from AgentSession → healthcare entities
4. Works both with `--llm` (real LLM) and without (placeholder responses)
5. All PSS signals come directly from the API — no client-side heuristics

Scenarios 1, 2, 7 stay unchanged.

---

## Neo4j Fixture Entities Available

```
Patient:    James Morrison, Maria Rodriguez, Aisha Patel, David Park, Carlos Gutierrez
Provider:   Dr. Sarah Chen, Dr. Michael O'Brien, Dr. Elena Volkov, Dr. Yuki Tanaka, Dr. Rachel Okonkwo
Diagnosis:  Type 2 Diabetes Mellitus, Essential Hypertension, Acute Myocardial Infarction,
            Major Depressive Disorder, Chronic Obstructive Pulmonary Disease
Treatment:  Cardiac Catheterization, Physical Therapy Program, Cognitive Behavioral Therapy,
            Chemotherapy Cycle 1, Joint Replacement Surgery
Medication: Metformin 500mg, Lisinopril 10mg, Atorvastatin 40mg, Amoxicillin 250mg, Sertraline 50mg
Encounter:  Annual Physical Exam, Emergency Room Visit, Follow-Up Appointment,
            Surgical Consultation, Lab Work Review
Facility:   Memorial General Hospital, Riverside Medical Center, St. Mary's Regional Hospital,
            University Health System, Cedar Grove Clinic
```

Key relationships: DIAGNOSED_WITH, TREATED_BY, PRESCRIBED, HAD_ENCOUNTER,
AFFILIATED_WITH, CONTRAINDICATED_WITH, REFERRED_TO, OCCURRED_AT, INCLUDES, USES, TREATS

---

## Scenario 3: Multi-Specialist Ward Round (PSS Layer 2 — Clusters)

### Story
Three specialists examine **David Park** (Essential Hypertension + COPD).
They share a PSS **cluster** so findings propagate between agents.

### PSS API calls
- `POST /cluster/` — create "park-ward-round" cluster (coupling_factor=0.25)
- `POST /cluster/{id}/store` — Dr. Chen seeds 4 baseline Q&A pairs about Park
- `POST /cluster/{id}/run` — Dr. Volkov queries (should get HITs from Chen's findings)
- `POST /cluster/{id}/run` — Dr. Tanaka queries (also gets HITs)
- `POST /cluster/{id}/store` — Tanaka stores new finding
- `POST /cluster/{id}/run` — Volkov queries again → gets Tanaka's new finding (HIT)
- `POST /cluster/{id}/feedback` — apply G4 coupling
- `POST /cluster/{id}/members` — register all 3 sessions
- `GET /cluster/{id}` — show cluster state (member_count, interaction_count, aggregate_vector dims)
- `DELETE /cluster/{id}` — cleanup

### Healthcare queries (David Park)
Phase A — Dr. Chen seeds baseline (threshold=0.99, always MISS):
1. "David Park presents with Essential Hypertension — current BP readings and medication?"
2. "Park also has Chronic Obstructive Pulmonary Disease — what is his FEV1 and current inhaler regimen?"
3. "Are there contraindications between Lisinopril for hypertension and his COPD medications?"
4. "What is Park's encounter history — when was his last Annual Physical Exam?"

Phase B — Dr. Volkov queries (threshold=0.52, expect HITs from Chen):
1. "What is David Park's blood pressure status and antihypertensive medication?" (≈ Chen Q1)
2. "Does Park have any pulmonary comorbidities affecting his treatment plan?" (≈ Chen Q2)
3. "What interactions should we monitor between his cardiac and pulmonary medications?" (≈ Chen Q3, novel angle)

Phase C — Dr. Tanaka queries + contributes new finding:
1. "What is the hypertension management plan for David Park?" (≈ Chen Q1, expect HIT)
2. "Park's latest ECG shows left ventricular hypertrophy — should we adjust treatment?" (NOVEL)
Store: Tanaka's ECG finding into cluster

Phase D — Dr. Volkov re-queries after Tanaka's contribution:
1. "Any new cardiac findings for David Park?" (should HIT Tanaka's ECG finding)

### Neo4j
- Create INVESTIGATED from each AgentSession → Patient David Park, relevant Diagnosis, Medication, Encounter
- Keep Cluster in Neo4j (don't delete) with MEMBER_OF relationships
- Show: HIT rate per specialist, coupling effect, who contributed what

### LLM roles
- Dr. Chen: "an internist performing baseline workup for David Park"
- Dr. Volkov: "a pulmonologist reviewing David Park's respiratory status"
- Dr. Tanaka: "a cardiologist evaluating David Park's cardiac function"

---

## Scenario 4: Medication Safety Guard (PSS Layer 1b — Anchors, Triggers, Isolation, Memory)

### Story
An agent monitors **Carlos Gutierrez** starting Chemotherapy Cycle 1.
Safety guardrails are set up using PSS Layer 1b features:
- **Drift Anchor** locks the session to oncology domain
- **Resonance Trigger** on "CRITICAL" and "contraindication" forces high importance
- **Synthetic Memory Injection** seeds known drug interactions
- **Input Isolation** quarantines off-topic inputs

### PSS API calls
- `POST /session/create` — create session with enable_topic_switch=true
- `POST /session/{id}/anchor` — anchor on oncology embedding (generate via LLM or pseudo)
- `POST /session/{id}/trigger` — add keyword trigger "CRITICAL"
- `POST /session/{id}/trigger` — add keyword trigger "contraindication"
- `POST /session/{id}/memory` — inject 3 synthetic memories:
  - "Amoxicillin 250mg is contraindicated with Metformin 500mg" (from CONTRAINDICATED_WITH)
  - "Atorvastatin 40mg is contraindicated with Sertraline 50mg" (from CONTRAINDICATED_WITH)
  - "Carlos Gutierrez Chemotherapy Cycle 1 uses Atorvastatin 40mg and Metformin 500mg" (from USES)
- `PUT /session/{id}/isolation` — set to QUARANTINE with exclusion embeddings
- `POST /run` × 5 — oncology queries about Gutierrez chemo (should work normally)
- `POST /run` × 2 — off-topic queries (hospital admin) → should be quarantined/filtered
- `GET /session/{id}/anchor_score` — show anchor drift status
- `POST /session/{id}/isolation/release` — release quarantined items
- `POST /run` with "CRITICAL: Gutierrez neutropenic fever" → trigger fires, high importance

### Healthcare queries (Carlos Gutierrez)
Oncology phase:
1. "Carlos Gutierrez is starting Chemotherapy Cycle 1 — what pre-treatment labs are required?"
2. "What antiemetic protocol for his cisplatin-based regimen?"
3. "CRITICAL: Monitor for neutropenic fever — what ANC thresholds for dose delay?"
4. "Does Chemotherapy Cycle 1 use Atorvastatin 40mg — any interactions?"
5. "What is the contraindication profile between chemo drugs and his existing medications?"

Off-topic (should trigger isolation):
6. "When is the next Pharmacy and Therapeutics Committee meeting?" (quarantined)
7. "What is the bed count at Cedar Grove Clinic?" (quarantined)

### Neo4j
- INVESTIGATED: AgentSession → Patient Gutierrez, Treatment Chemo, Medication (Atorvastatin, Metformin), Diagnosis
- Memory nodes with REFERENCES to Medication/Diagnosis
- Anchor score tracking in SemanticState.beta

### LLM role
- "an oncology pharmacist monitoring Carlos Gutierrez's chemotherapy safety"

---

## Scenario 5: Hospital Network Consensus (PSS Layers 3+4 — Regions, Observer)

### Story
Two department clusters across two facilities:
- **Cluster A** "cardiology-memorial" at Memorial General: investigating Maria Rodriguez (AMI)
- **Cluster B** "emergency-riverside" at Riverside Medical: investigating James Morrison (ER visit)

Both clusters are in **Region** "hospital-network".
When cardiology detects drift (topic shift during Rodriguez investigation),
the region checks if emergency also drifted → **consensus**.
**Observer** samples the region and detects cross-cluster anomalies.

### PSS API calls
- `POST /cluster/` × 2 — create cardiology + emergency clusters
- `POST /cluster/{id}/store` — seed each with 4 domain Q&A pairs
- `POST /cluster/{id}/members` — register agent sessions
- `POST /region/` — create "hospital-network" region (consensus_threshold=0.5)
- `POST /region/{id}/clusters` × 2 — add both clusters
- `POST /observer/` — create observer, register region
- `POST /run` × 4 per cluster — on-topic queries
- `POST /run` — pivot query in cardiology cluster → drift detected
- `GET /region/{id}/events` — check for drift events
- `POST /observer/sample` — trigger manual sample
- `GET /observer/anomalies` — check for cross-cluster anomalies
- `GET /observer/summary` — full snapshot
- Cleanup: DELETE clusters, region, observer anomalies

### Healthcare queries
Cardiology cluster (Rodriguez):
1. "Maria Rodriguez Acute Myocardial Infarction — current troponin levels?"
2. "Post-MI antiplatelet therapy — aspirin plus clopidogrel or ticagrelor?"
3. "Dr. O'Brien referred Rodriguez to Dr. Okonkwo — for cardiac rehab?"
4. "Rodriguez ER encounter — which interventions were performed?"
5. PIVOT: "What is the latest infection control audit status at Memorial General?" (drift)

Emergency cluster (Morrison):
1. "James Morrison Emergency Room Visit — chief complaint and triage priority?"
2. "Morrison has Type 2 Diabetes and COPD — medication reconciliation needed?"
3. "Dr. Volkov attending Morrison's ER encounter — any cardiac biomarker results?"
4. "Morrison's chest pain workup — should we consult cardiology?"

### Neo4j
- Cluster A → MEMBER_OF ← sessions, Region → CONTAINS_CLUSTER → Clusters
- INVESTIGATED from sessions → Rodriguez, Morrison, respective diagnoses
- Facility Memorial, Riverside linked to Clusters
- AnomalyEvent, ConsensusEvent nodes if detected

### LLM roles
- Cardiology: "a cardiologist at Memorial General Hospital treating Maria Rodriguez"
- Emergency: "an emergency physician at Riverside Medical Center treating James Morrison"

---

## Scenario 6: Shift Handoff (PSS Layer 5 — Export/Import, Network Transfer)

### Story
Night shift **Dr. Volkov** finishes Morrison's overnight monitoring.
She exports her PSS session state.
Day shift **Dr. Tanaka** imports the state and receives a delta transfer
of Volkov's accumulated semantic vector. Tanaka continues with full context.

### PSS API calls
- `POST /session/create` — Volkov's night shift session
- `POST /run` × 6 — Volkov's overnight queries about Morrison
- `GET /session/{id}/export` — export Volkov's state (JSON + SHA-256 checksum)
- `POST /session/create` — Tanaka's day shift session
- `POST /session/{id}/import` — import Volkov's state into Tanaka's session
- `POST /session/{id}/verify` — verify behavioral consistency (optional, may 404)
- `POST /network/transfer` — delta transfer Volkov → Tanaka (max_weight=0.15)
- `POST /run` × 3 — Tanaka continues investigation (should have full context)
- Compare: Tanaka's top_similarity on first query WITH vs WITHOUT import

### Healthcare queries
Volkov night shift (Morrison):
1. "James Morrison overnight vitals — any concerning trends in blood glucose?"
2. "Morrison's Metformin was held for catheterization — when to resume?"
3. "Overnight troponin trend for Morrison — any elevation?"
4. "Morrison's COPD — overnight SpO2 readings and oxygen requirements?"
5. "Lab results from Morrison's midnight blood draw — CBC and CMP?"
6. "Morrison's morning insulin dose — calculate based on fasting glucose"

Tanaka day shift (continues Morrison):
1. "What happened overnight with James Morrison — any changes in condition?"
2. "Morrison's morning labs — should we restart Metformin today?"
3. "Plan for Morrison's discharge — medication reconciliation and follow-up schedule?"

### Neo4j
- Provider Volkov → INVESTIGATED → Morrison (night, steps 0-5)
- Provider Tanaka → INVESTIGATED → Morrison (day, steps 6-8)
- Show context continuity: Tanaka's first query should have high similarity
  if import worked (vs low similarity without)

### LLM roles
- Volkov: "Dr. Elena Volkov, night shift internist monitoring James Morrison overnight"
- Tanaka: "Dr. Yuki Tanaka, day shift internist taking over Morrison's care"

---

## Testing Strategy

Each scenario should have tests that verify:

1. **PSS API calls return expected fields** — drift_score, drift_detected, top_similarity, short_circuit, context
2. **Neo4j nodes created** — correct AgentSession, SemanticState, Phase, DriftEvent, INVESTIGATED relationships
3. **Healthcare entity references** — INVESTIGATED edges point to correct Patient/Provider/Diagnosis/Medication
4. **Layer-specific features work** — cluster HITs, anchor scores, region events, observer anomalies, export/import

Test files:
- `tests/test_scenario3_cluster.py` — cluster create/run/store/feedback/members
- `tests/test_scenario4_safety.py` — anchor, trigger, isolation, memory injection
- `tests/test_scenario5_consensus.py` — region, observer, cross-cluster drift
- `tests/test_scenario6_handoff.py` — export/import, delta transfer

Each test uses real PSS API (skipped if unavailable) + Neo4j testcontainer.

---

## Files to modify

- `scripts/interactive_demo.py` — replace scenarios 3-6
- `src/core/pss_client.py` — may need new methods for Layer 1b (anchor, trigger, isolation, memory, entities)
- No changes to `src/core/drift_detector.py` (stays as pure pass-through)
- No changes to `src/mcp/pss_mcp_server.py` (scenarios use raw PSSClient directly)
