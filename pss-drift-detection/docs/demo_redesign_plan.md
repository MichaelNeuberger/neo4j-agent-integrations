# Demo Redesign Plan — PSS × Neo4j Healthcare

## Narrative: One hospital, five patients, multiple specialists

All scenarios play in the same hospital network (Memorial General, Riverside, St. Mary's, University Health, Cedar Grove).
Each scenario shows a different PSS capability using the real fixture data.

---

## Scenario 1: Live Drift Detection (interactive, keep as-is)

User types messages, sees PSS signals + Neo4j state chain in real-time.

---

## Scenario 2: Drift + Short-Circuit (keep as-is, already done)

Morrison diabetes → Patel psychiatry → Rodriguez cardiology → paraphrase HITs.
Shows: drift_detected, drift_score, drift_phase, short_circuit, top_similarity.
PSS: Layer 1 (/run, /store, inline-store).

---

## Scenario 3: Multi-Specialist Ward Round (Layer 2 — Clusters)

**Story:** Three specialists examine the same patient (David Park: Hypertension + COPD + Depression).
Each specialist has their own PSS session but they share a **cluster** so findings propagate.

- Dr. Sarah Chen (internal medicine) seeds the cluster with Park's baseline workup
- Dr. Elena Volkov (endocrinology) queries the cluster → gets HITs from Chen's findings
- Dr. Yuki Tanaka (cardiology) queries → also gets HITs
- When Tanaka discovers something new (ECG finding), it's stored → next specialist gets it

**PSS Features:** /cluster/create, /cluster/run, /cluster/store, /cluster/feedback, /cluster/members
**Neo4j:** Patient David Park → DIAGNOSED_WITH → Hypertension, COPD
           Provider Dr. Chen, Volkov, Tanaka → INVESTIGATED → David Park
           Cluster → MEMBER_OF ← AgentSession (3 specialists)
           Show: cross-agent HIT rate, coupling effect, who contributed what

**Graph entities used:** Patient, Provider, Diagnosis, Medication (Lisinopril, Amoxicillin), Encounter, Facility

---

## Scenario 4: Medication Safety Guard (Layer 1b — Anchors, Isolation, Triggers)

**Story:** An agent reviews Carlos Gutierrez (Chemotherapy Cycle 1). We set up safety guardrails:
- **Drift Anchor** on the oncology domain (embedding of "chemotherapy patient monitoring")
- **Resonance Trigger** on keyword "CRITICAL" and "contraindication"
- **Input Isolation** set to QUARANTINE for off-topic inputs
- Agent processes oncology queries normally, then someone asks about unrelated admin topic → quarantined

**PSS Features:** /session/create, /session/{id}/anchor, /session/{id}/anchor_score,
                  /session/{id}/trigger, /session/{id}/isolation, /session/{id}/isolation/release,
                  /session/{id}/memory (inject known contraindications)
**Neo4j:** Patient Carlos Gutierrez → Treatment Chemotherapy Cycle 1
           Medication interactions (CONTRAINDICATED_WITH)
           DriftEvent when admin question arrives, Anchor score tracking
           Memory nodes for injected contraindication knowledge

**Graph entities used:** Patient (Gutierrez), Treatment (Chemo), Medication (all 5 for interactions),
                         Diagnosis, Provider (Dr. O'Brien oncology), Facility

---

## Scenario 5: Hospital Network Consensus (Layers 3+4 — Regions, Observer)

**Story:** Two departments (clusters) in two facilities:
- Cluster A: "Cardiology" at Memorial General (agents: Dr. O'Brien, Dr. Okonkwo)
- Cluster B: "Emergency" at Riverside Medical (agents: Dr. Chen, Dr. Tanaka)

Both investigate related patients. When cardiology detects drift (patient deterioration topic shift),
the **region** checks if emergency also drifted → **consensus** triggers realignment.
**Observer** samples both regions and detects cross-cluster convergence anomaly.

**PSS Features:** /region/create, /region/clusters, /region/events,
                  /observer/create, /observer/sample, /observer/anomalies, /observer/summary,
                  /cluster/create, /cluster/run, /cluster/members
**Neo4j:** Region → CONTAINS_CLUSTER → Cluster → MEMBER_OF ← AgentSession
           Facility (Memorial, Riverside) linked to Clusters
           ConsensusEvent, AnomalyEvent nodes created
           Provider → AFFILIATED_WITH → Facility → hosting Cluster

**Graph entities used:** Facility (Memorial, Riverside), Provider (O'Brien, Okonkwo, Chen, Tanaka),
                         Patient (Rodriguez for cardio, Morrison for emergency), Diagnosis, Encounter

---

## Scenario 6: Shift Handoff (Layer 5 — Network Transfer, Session Export/Import)

**Story:** Night shift doctor (Dr. Volkov) finishes and exports her session state.
Day shift doctor (Dr. Tanaka) imports the state + receives delta transfer of overnight findings.
Then Tanaka continues the investigation with full context.

- Volkov works through 8 queries on Morrison's overnight labs
- Export session state (with checksum)
- Tanaka imports the state
- Delta transfer of Volkov's session semantic vector to Tanaka
- Tanaka queries → sees accumulated context from Volkov's shift

**PSS Features:** /session/{id}/export, /session/{id}/import, /session/{id}/verify,
                  /network/transfer (delta vector), /network/users/switch (partition)
**Neo4j:** Provider Dr. Volkov → INVESTIGATED → Patient Morrison (night shift)
           Provider Dr. Tanaka → INVESTIGATED → Patient Morrison (day shift)
           Session export/import metadata nodes
           Show: context continuity across shift handoff

**Graph entities used:** Provider (Volkov, Tanaka), Patient (Morrison), Encounter (Lab Work Review),
                         Medication, Diagnosis, Facility (Memorial General)

---

## Scenario 7: Neo4j Graph Explorer (keep, update queries)

Interactive Cypher explorer with pre-built queries.
Update queries to reflect new scenarios (Clusters, Regions, Observer, Anchors).

---

## PSS API Coverage After Redesign

| Layer | Feature | Scenario |
|---|---|---|
| 1 | /run, /store, inline-store | 1, 2, 3, 4, 5, 6 |
| 1 | drift_detected, drift_score, drift_phase | 2, 5 |
| 1 | short_circuit, top_similarity | 2, 3 |
| 1 | reset_context | 5 |
| 1b | /session/create (template) | 4, 6 |
| 1b | /session/{id}/anchor + /anchor_score | 4 |
| 1b | /session/{id}/trigger | 4 |
| 1b | /session/{id}/isolation + /release | 4 |
| 1b | /session/{id}/memory | 4 |
| 1b | /session/{id}/export + /import + /verify | 6 |
| 2 | /cluster/* (create, run, store, feedback, members) | 3, 5 |
| 3 | /region/* (create, clusters, events) | 5 |
| 4 | /observer/* (create, sample, anomalies, summary) | 5 |
| 5 | /network/transfer | 6 |
| 5 | /network/users/switch | 6 |

## Neo4j Entity Coverage After Redesign

| Entity | Scenario 2 | Scenario 3 | Scenario 4 | Scenario 5 | Scenario 6 |
|---|---|---|---|---|---|
| Patient Morrison | ✓ | | | ✓ | ✓ |
| Patient Rodriguez | ✓ | | | ✓ | |
| Patient Patel | ✓ | | | | |
| Patient Park | | ✓ | | | |
| Patient Gutierrez | | | ✓ | | |
| All 5 Providers | ✓ | ✓ | ✓ | ✓ | ✓ |
| All 5 Diagnoses | ✓ | ✓ | ✓ | ✓ | ✓ |
| All 5 Medications | ✓ | ✓ | ✓ | | ✓ |
| All 5 Treatments | | | ✓ | | |
| All 5 Encounters | ✓ | | | ✓ | ✓ |
| All 5 Facilities | | | | ✓ | ✓ |
| Cluster | | ✓ | | ✓ | |
| Region | | | | ✓ | |
| Observer | | | | ✓ | |
