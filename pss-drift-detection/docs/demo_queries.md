# Neo4j Browser Demo Queries

Open http://localhost:7474 — Login: `neo4j` / `testpassword`

Copy each query into the top bar and hit **Play** (or Ctrl+Enter).
Switch between **Graph** / **Table** view with the icons on the left of the result.

---

## 1. Schema Overview

Shows all node types and their relationships as a diagram:

```cypher
CALL db.schema.visualization()
```

## 2. What's in the database?

```cypher
MATCH (n)
RETURN labels(n)[0] AS type, count(n) AS count
ORDER BY count DESC
```

## 3. Healthcare Knowledge Graph

Patients with their diagnoses, treatments, providers:

```cypher
MATCH (p:Patient)-[r]-(n)
RETURN p, r, n
LIMIT 100
```

Provider referral network:

```cypher
MATCH (p1:Provider)-[ref:REFERRED_TO]->(p2:Provider)
OPTIONAL MATCH (p1)-[:AFFILIATED_WITH]->(f:Facility)
RETURN p1, ref, p2, f
```

## 4. PSS Agent Dashboard

All agents with their current phase and drift status — **TABLE view recommended**:

```cypher
MATCH (s:AgentSession)-[:CURRENT_PHASE]->(phase:Phase)
MATCH (s)-[:CURRENT_STATE]->(state:SemanticState)
OPTIONAL MATCH (s)-[:MEMBER_OF]->(c:Cluster)
OPTIONAL MATCH (d:DriftEvent {session_id: s.session_id})
WITH s, phase, state, c,
     count(d) AS drift_events,
     avg(d.drift_score) AS avg_drift
RETURN s.agent_id AS agent,
       phase.name AS current_phase,
       state.step AS steps,
       round(state.beta * 1000) / 1000 AS beta,
       c.name AS cluster,
       drift_events,
       round(coalesce(avg_drift, 0) * 100) / 100 AS avg_drift_score
ORDER BY avg_drift_score DESC
```

## 5. Semantic State Chain (GRAPH view)

Shows how one agent's semantic state evolved over time — **switch to GRAPH view**:

```cypher
MATCH (s:AgentSession {agent_id: 'cardiology-researcher'})
      -[:CURRENT_STATE]->(current:SemanticState)
MATCH path = (current)-[:STATE_HISTORY*0..]->(old:SemanticState)
RETURN path
```

Try also: `oncology-researcher`, `er-triage-agent`, `care-coordinator`

## 6. Drift Events — Where Agents Lost Focus

```cypher
MATCH (st:SemanticState)-[t:TRIGGERED]->(d:DriftEvent)
RETURN st, t, d
```

Highest drift scores — **TABLE view**:

```cypher
MATCH (d:DriftEvent)
RETURN d.session_id AS agent,
       d.severity AS severity,
       round(d.drift_score * 100) / 100 AS drift_score,
       d.topic_switch AS topic_switch
ORDER BY d.drift_score DESC
```

## 7. Phase Transitions

How conversations evolved through phases — **GRAPH view**:

```cypher
MATCH (p1:Phase)-[t:TRANSITIONED_TO]->(p2:Phase)
RETURN p1, t, p2
```

Transition matrix — **TABLE view**:

```cypher
MATCH (p1:Phase)-[:TRANSITIONED_TO]->(p2:Phase)
RETURN p1.name AS from_phase, p2.name AS to_phase, count(*) AS transitions
ORDER BY transitions DESC
```

## 8. Multi-Agent Cluster Topology (GRAPH view)

The full MetaPSS hierarchy: Region → Clusters → Agents:

```cypher
MATCH (s:AgentSession)-[m:MEMBER_OF]->(c:Cluster)
MATCH (c)<-[cc:CONTAINS_CLUSTER]-(r:Region)
RETURN s, m, c, cc, r
```

## 9. Cluster Drift Comparison

Which cluster drifts more?

```cypher
MATCH (s:AgentSession)-[:MEMBER_OF]->(c:Cluster)
MATCH (d:DriftEvent {session_id: s.session_id})
RETURN c.name AS cluster,
       count(DISTINCT s) AS agents,
       count(d) AS drift_events,
       round(avg(d.drift_score) * 100) / 100 AS avg_drift
ORDER BY avg_drift DESC
```

## 10. Agent Memories

What each agent remembers:

```cypher
MATCH (s:AgentSession)-[:HAS_MEMORY]->(m:Memory)
RETURN s.agent_id AS agent,
       m.text_summary AS memory,
       m.importance AS importance,
       m.tier AS tier
ORDER BY m.importance DESC
LIMIT 20
```

## 11. Full Agent View (GRAPH view)

One agent with ALL its connections — state chain, phases, drift events, memories, cluster:

```cypher
MATCH (s:AgentSession {agent_id: 'oncology-researcher'})
OPTIONAL MATCH (s)-[r1:CURRENT_STATE]->(state:SemanticState)
OPTIONAL MATCH (s)-[r2:CURRENT_PHASE]->(phase:Phase)
OPTIONAL MATCH (s)-[r3:MEMBER_OF]->(cluster:Cluster)
OPTIONAL MATCH (s)-[r4:HAS_MEMORY]->(mem:Memory)
OPTIONAL MATCH (state)-[r5:STATE_HISTORY*0..3]->(old:SemanticState)
OPTIONAL MATCH (state)-[r6:TRIGGERED]->(drift:DriftEvent)
RETURN s, state, phase, cluster, mem, old, drift,
       r1, r2, r3, r4, r5, r6
```

## 12. Combined View: Healthcare + PSS

Healthcare entities alongside agent drift monitoring:

```cypher
MATCH (p:Patient)-[:DIAGNOSED_WITH]->(d:Diagnosis)
WITH d, count(p) AS patient_count
ORDER BY patient_count DESC
LIMIT 5
WITH collect(d.name) AS top_diagnoses
MATCH (s:AgentSession)-[:CURRENT_PHASE]->(phase:Phase)
OPTIONAL MATCH (drift:DriftEvent {session_id: s.session_id})
RETURN s.agent_id AS agent,
       phase.name AS phase,
       count(drift) AS drift_events,
       top_diagnoses
```
