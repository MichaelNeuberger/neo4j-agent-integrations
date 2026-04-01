# Implementation Plan: PSS Drift Detection Integration for Neo4j Agent Ecosystem

## Document Information

- **Status:** Draft
- **Version:** 1.0
- **Date:** 2026-03-30
- **Audience:** Neo4j Product Manager, Engineering Lead, PSS Integration Team
- **Repository:** neo4j-agent-integrations

---

## 1. Executive Summary

This plan describes how to integrate the Persistent Semantic State (PSS) drift detection system into Neo4j's agent ecosystem. PSS solves the LLM context window problem by maintaining a compressed O(1) semantic state (384-dimensional vector) instead of full conversation history, and it provides a 3-layer drift detection system and a 4-layer multi-agent coordination architecture (MetaPSS).

Neo4j's current agent stack offers 20+ framework integrations via MCP, passive memory storage (neo4j-agent-memory), and raw checkpointing (Neo4jSaver), but has no drift detection, no phase detection, and no multi-agent coordination intelligence. PSS fills all three gaps.

The integration strategy is threefold:

1. **Neo4j as PSS persistence layer** -- replace PSS's in-memory data structures with graph-native storage, enabling queryable drift history, traversable phase transitions, and graph-algorithm-enhanced analytics.
2. **MCP tool exposure** -- surface drift detection capabilities as MCP tools so all 20+ existing framework integrations (LangGraph, CrewAI, OpenAI SDK, Claude SDK, etc.) gain drift awareness without per-framework work.
3. **Enhancement of existing products** -- add an active monitoring layer over neo4j-agent-memory and upgrade Neo4jSaver checkpoints with semantic state and phase metadata.

---

## 2. Requirements

### 2.1 Functional Requirements

- FR-1: Persist PSS semantic states (384-dim vectors) as Neo4j nodes with vector index support.
- FR-2: Store and query drift events with temporal relationships (PRECEDED_BY, TRIGGERED_BY).
- FR-3: Track phase transitions as traversable graph paths across the 6 conversation phases.
- FR-4: Support MetaPSS multi-agent topology: LocalPSSInstance, ClusterManager, RegionalManager, GlobalObserver.
- FR-5: Expose core drift detection operations as MCP tools consumable by all 20+ frameworks.
- FR-6: Integrate drift-triggered memory consolidation with neo4j-agent-memory.
- FR-7: Enable graph algorithm analysis (PageRank, community detection) on agent influence and drift correlation networks.
- FR-8: Provide a REST API layer compatible with PSS's existing Layer 1-4 endpoint structure.

### 2.2 Non-Functional Requirements

- NFR-1: Drift detection latency under 50ms for single-agent Layer 1 operations.
- NFR-2: Support 1000+ concurrent agent sessions per Neo4j instance.
- NFR-3: Semantic state vector similarity queries must use Neo4j native vector index (not brute-force).
- NFR-4: Backward compatible with existing neo4j-agent-integrations patterns (MCP stdio/HTTP, driver-based).
- NFR-5: No disruption to existing 20+ framework integrations.

### 2.3 Assumptions

- Neo4j 5.x or later with vector index support (available since 5.11).
- Neo4j Graph Data Science (GDS) library available for graph algorithm operations.
- PSS embedding model produces 384-dimensional vectors (sentence-transformers/all-MiniLM-L6-v2 or equivalent).
- The Neo4j MCP server (https://github.com/neo4j/mcp) can be extended with custom tool definitions.
- Agent frameworks consume tools via MCP; no per-framework SDK work is required for basic drift detection.

---

## 3. Architecture Overview

### 3.1 PSS Layers Mapped to Neo4j Graph Structure

```
PSS Architecture                          Neo4j Graph Representation
=================                         ==========================

Layer 4: GlobalObserver ─────────────────> (:GlobalObserver)
  - pattern_vector                           {pattern_vector: [float], anomaly_type: str}
  - anomaly detection                        -[:OBSERVES]->(:Region)
                                             -[:DETECTED]->(:AnomalyEvent)

Layer 3: RegionalManager ────────────────> (:Region)
  - DriftEventBus                            {name: str, consensus_threshold: 0.6}
  - weighted voting                          -[:CONTAINS]->(:Cluster)
  - consensus protocol                       -[:CONSENSUS_EVENT]->(:ConsensusEvent)
                                             -[:DRIFT_EVENT]->(:DriftEvent)

Layer 2: ClusterManager ─────────────────> (:Cluster)
  - aggregation strategy                     {strategy: 'weighted_avg'|'attention',
  - coupling feedback                         coupling_strength: float}
                                             -[:MEMBER]->(:AgentSession)
                                             -[:AGGREGATED_STATE]->(:SemanticState)

Layer 1: LocalPSSInstance ───────────────> (:AgentSession)
  - session semantic_state                   {session_id: str, agent_id: str}
  - drift detection                          -[:CURRENT_STATE]->(:SemanticState)
  - phase detection                          -[:STATE_HISTORY]->(:SemanticState)
                                             -[:CURRENT_PHASE]->(:Phase)
                                             -[:MEMORY_TIER]->(:MemoryTier)
```

### 3.2 Data Flow: Agent Input to MCP Exposure

```
                                    Neo4j Graph Database
                                    ====================
  Agent Framework                   |                    |
  (any of 20+)                      | (:SemanticState)   |
       |                            | (:DriftEvent)      |
       v                            | (:Phase)           |
  +----------+    MCP Protocol      | (:AgentSession)    |     +----------------+
  | LangGraph|---+                  | (:Cluster)         |     | Graph Algorithms|
  | CrewAI   |   |  +-----------+  | (:Region)          |     | (GDS Library)  |
  | OpenAI   |   +->| Neo4j MCP |  | (:Memory)          |     +-------+--------+
  | Claude   |   |  | Server    |  |                    |             |
  | Strands  |   |  |           |  +--------------------+             |
  | ADK      |---+  | +-------+|         ^    |                      |
  | ...      |      | | PSS   ||  write  |    | read                 |
  +----------+      | | Drift ||─────────+    v                      |
                    | | Tools ||  Cypher   +--------+                |
                    | +-------+|<──────────| Query  |<───────────────+
                    +-----------+  results  | Engine |  algorithm results
                         |                 +--------+
                         |
                    +-----------+
                    | PSS Core  |
                    | Library   |
                    |           |
                    | - core.py |
                    | - phase.py|
                    | - memory  |
                    | - network |
                    +-----------+
```

### 3.3 How Existing Frameworks Consume Drift Detection via MCP

```
  Framework                  MCP Connection              PSS Drift Tools
  =========                  ==============              ===============

  LangGraph ──── langchain-mcp-adapters ──┐
  LangChain ─── langchain-mcp-adapters ───┤
  OpenAI SDK ── MCPServerStreamableHttp ──┤
  Claude SDK ── mcp_servers param ────────┤       +─────────────────────+
  CrewAI ────── mcp adapter ──────────────┤       │ Neo4j MCP Server    │
  Google ADK ── McpToolset ───────────────┼──────>│                     │
  Strands ───── MCPClient ────────────────┤       │  Existing tools:    │
  Pydantic AI ─ mcp adapter ─────────────┤       │   - get-schema      │
  Haystack ──── mcp adapter ──────────────┤       │   - execute-query   │
  n8n ───────── MCP node ─────────────────┤       │                     │
  Databricks ── MCP Catalog ──────────────┘       │  NEW PSS tools:     │
                                                  │   - detect_drift    │
  All frameworks get drift                        │   - get_phase       │
  detection automatically via                     │   - query_drift_    │
  the same MCP connection they                    │     history         │
  already use for Neo4j queries.                  │   - cluster_state   │
                                                  │   - network_        │
                                                  │     anomalies      │
                                                  │   - memory_         │
                                                  │     consolidate    │
                                                  │   - consensus_     │
                                                  │     status         │
                                                  +─────────────────────+
```

---

## 4. Neo4j Graph Data Model for PSS

### 4.1 Node Types

#### Core Session Nodes

| Node Label       | Properties                                                                                           | Indexes                                      |
|------------------|------------------------------------------------------------------------------------------------------|----------------------------------------------|
| `:AgentSession`  | `session_id` (str, unique), `agent_id` (str), `created_at` (datetime), `last_active` (datetime), `status` (str) | UNIQUE on session_id, INDEX on agent_id      |
| `:Agent`         | `agent_id` (str, unique), `name` (str), `framework` (str), `version` (str)                          | UNIQUE on agent_id                           |

#### Semantic State Nodes

| Node Label        | Properties                                                                                                                  | Indexes                                         |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| `:SemanticState`  | `state_id` (str, unique), `vector` (float[], 384-dim), `timestamp` (datetime), `step` (int), `beta` (float), `mean_similarity` (float), `variance` (float) | UNIQUE on state_id, VECTOR INDEX on vector (cosine, 384) |

#### Drift Detection Nodes

| Node Label    | Properties                                                                                                                                     | Indexes                          |
|---------------|------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|
| `:DriftEvent` | `event_id` (str, unique), `timestamp` (datetime), `drift_score` (float), `topic_switch` (boolean), `cosine_drop` (float), `mean_sim` (float), `variance` (float), `severity` (str: low/medium/high/critical) | UNIQUE on event_id, INDEX on timestamp, INDEX on severity |

#### Phase Detection Nodes

| Node Label | Properties                                                                                                    | Indexes              |
|------------|---------------------------------------------------------------------------------------------------------------|----------------------|
| `:Phase`   | `phase_id` (str, unique), `name` (str: initialization/exploration/convergence/resonance/stability/instability), `entered_at` (datetime), `exited_at` (datetime), `srs_score` (float), `tc_score` (float), `fsm_state` (str), `be_score` (float), `markov_probability` (float), `rule_score` (float) | UNIQUE on phase_id, INDEX on name |

#### Memory Tier Nodes

| Node Label    | Properties                                                                                                                | Indexes                          |
|---------------|---------------------------------------------------------------------------------------------------------------------------|----------------------------------|
| `:Memory`     | `memory_id` (str, unique), `tier` (str: short/medium/long), `content_vector` (float[], 384-dim), `importance` (float, 0-1), `recency` (float, 0-1), `access_count` (int), `created_at` (datetime), `last_accessed` (datetime), `text_summary` (str) | UNIQUE on memory_id, INDEX on tier, VECTOR INDEX on content_vector |
| `:MemoryCluster` | `cluster_id` (str, unique), `centroid` (float[], 384-dim), `size` (int), `tier` (str)                                | UNIQUE on cluster_id             |

#### Multi-Agent Coordination Nodes (MetaPSS Layers 2-4)

| Node Label        | Properties                                                                                                              | Indexes                    |
|-------------------|-------------------------------------------------------------------------------------------------------------------------|----------------------------|
| `:Cluster`        | `cluster_id` (str, unique), `name` (str), `strategy` (str: weighted_avg/attention), `coupling_strength` (float, 0-1), `created_at` (datetime) | UNIQUE on cluster_id       |
| `:Region`         | `region_id` (str, unique), `name` (str), `consensus_threshold` (float, default 0.6), `voting_weight_scheme` (str)      | UNIQUE on region_id        |
| `:GlobalObserver` | `observer_id` (str, unique), `pattern_vector` (float[], 384-dim), `last_scan` (datetime)                               | UNIQUE on observer_id      |
| `:ConsensusEvent` | `event_id` (str, unique), `timestamp` (datetime), `outcome` (str: agreed/disagreed/timeout), `vote_count` (int), `threshold_used` (float), `drift_score_consensus` (float) | UNIQUE on event_id, INDEX on timestamp |
| `:AnomalyEvent`   | `event_id` (str, unique), `timestamp` (datetime), `type` (str: cross_cluster_convergence/systemic_drift/cluster_divergence), `severity` (float), `affected_clusters` (str[]) | UNIQUE on event_id         |

### 4.2 Relationship Types

#### Session and State Relationships

| Relationship              | From              | To                | Properties                          | Meaning                                |
|---------------------------|-------------------|-------------------|-------------------------------------|----------------------------------------|
| `:CURRENT_STATE`          | AgentSession      | SemanticState     | (none)                              | Points to the latest semantic state    |
| `:STATE_HISTORY`          | SemanticState     | SemanticState     | `cosine_similarity` (float)         | Temporal chain of state evolution      |
| `:OWNS_SESSION`           | Agent             | AgentSession      | (none)                              | Agent owns this session                |
| `:TRIGGERED`              | SemanticState     | DriftEvent        | (none)                              | This state transition triggered drift  |

#### Phase Relationships

| Relationship              | From              | To                | Properties                          | Meaning                                |
|---------------------------|-------------------|-------------------|-------------------------------------|----------------------------------------|
| `:CURRENT_PHASE`          | AgentSession      | Phase             | (none)                              | Session is currently in this phase     |
| `:TRANSITIONED_TO`        | Phase             | Phase             | `probability` (float), `trigger` (str: markov/rule/hybrid), `timestamp` (datetime) | Phase transition with Markov probability |
| `:PHASE_AT`               | SemanticState     | Phase             | (none)                              | What phase was active at this state    |

#### Memory Relationships

| Relationship              | From              | To                | Properties                          | Meaning                                |
|---------------------------|-------------------|-------------------|-------------------------------------|----------------------------------------|
| `:HAS_MEMORY`             | AgentSession      | Memory            | (none)                              | Session owns this memory               |
| `:CONSOLIDATED_INTO`      | Memory            | Memory            | `timestamp` (datetime)              | Short-term consolidated to medium/long |
| `:BELONGS_TO_CLUSTER`     | Memory            | MemoryCluster     | (none)                              | K-means++ clustering result            |
| `:SIMILAR_TO`             | Memory            | Memory            | `similarity` (float)                | Vector similarity above threshold      |

#### Multi-Agent Topology Relationships

| Relationship              | From              | To                | Properties                          | Meaning                                |
|---------------------------|-------------------|-------------------|-------------------------------------|----------------------------------------|
| `:MEMBER_OF`              | AgentSession      | Cluster           | `weight` (float), `joined_at` (datetime) | Agent session belongs to cluster  |
| `:CONTAINS_CLUSTER`       | Region            | Cluster           | (none)                              | Region contains cluster                |
| `:OBSERVES`               | GlobalObserver    | Region            | (none)                              | Observer monitors this region          |
| `:AGGREGATED_STATE`       | Cluster           | SemanticState     | `timestamp` (datetime)              | Cluster-level aggregated state         |
| `:CONSENSUS_ON`           | ConsensusEvent    | DriftEvent        | (none)                              | Consensus reached about this drift     |
| `:VOTED_IN`               | AgentSession      | ConsensusEvent    | `vote` (float), `weight` (float)    | Agent's vote in consensus              |
| `:DETECTED_BY`            | AnomalyEvent      | GlobalObserver    | (none)                              | Observer detected this anomaly         |
| `:CORRELATED_DRIFT`       | DriftEvent        | DriftEvent        | `correlation` (float), `lag` (int)  | Cross-agent drift correlation          |

### 4.3 Indexes and Constraints (Cypher DDL)

```cypher
// Uniqueness constraints
CREATE CONSTRAINT agent_session_unique IF NOT EXISTS
  FOR (s:AgentSession) REQUIRE s.session_id IS UNIQUE;

CREATE CONSTRAINT semantic_state_unique IF NOT EXISTS
  FOR (s:SemanticState) REQUIRE s.state_id IS UNIQUE;

CREATE CONSTRAINT drift_event_unique IF NOT EXISTS
  FOR (d:DriftEvent) REQUIRE d.event_id IS UNIQUE;

CREATE CONSTRAINT phase_unique IF NOT EXISTS
  FOR (p:Phase) REQUIRE p.phase_id IS UNIQUE;

CREATE CONSTRAINT memory_unique IF NOT EXISTS
  FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE;

CREATE CONSTRAINT cluster_unique IF NOT EXISTS
  FOR (c:Cluster) REQUIRE c.cluster_id IS UNIQUE;

CREATE CONSTRAINT region_unique IF NOT EXISTS
  FOR (r:Region) REQUIRE r.region_id IS UNIQUE;

// Vector indexes for similarity search
CREATE VECTOR INDEX semantic_state_vector IF NOT EXISTS
  FOR (s:SemanticState) ON (s.vector)
  OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }};

CREATE VECTOR INDEX memory_content_vector IF NOT EXISTS
  FOR (m:Memory) ON (m.content_vector)
  OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }};

// Temporal indexes for range queries
CREATE INDEX drift_event_timestamp IF NOT EXISTS
  FOR (d:DriftEvent) ON (d.timestamp);

CREATE INDEX phase_entered IF NOT EXISTS
  FOR (p:Phase) ON (p.entered_at);

CREATE INDEX semantic_state_timestamp IF NOT EXISTS
  FOR (s:SemanticState) ON (s.timestamp);

// Composite indexes for common query patterns
CREATE INDEX drift_severity_time IF NOT EXISTS
  FOR (d:DriftEvent) ON (d.severity, d.timestamp);

CREATE INDEX memory_tier_importance IF NOT EXISTS
  FOR (m:Memory) ON (m.tier, m.importance);
```

### 4.4 Example Subgraph

```
(:Agent {agent_id: "agent-1", name: "ResearchBot", framework: "langgraph"})
  -[:OWNS_SESSION]->
    (:AgentSession {session_id: "sess-abc", status: "active"})
      -[:CURRENT_STATE]->
        (:SemanticState {vector: [...384 floats...], step: 47, beta: 0.82})
          -[:STATE_HISTORY {cosine_similarity: 0.91}]->
            (:SemanticState {vector: [...], step: 46, beta: 0.79})
              -[:STATE_HISTORY {cosine_similarity: 0.62}]->  // <-- drop > 0.3 = topic switch
                (:SemanticState {vector: [...], step: 45, beta: 0.71})
          -[:TRIGGERED]->
            (:DriftEvent {drift_score: 0.74, topic_switch: true, severity: "high"})
          -[:PHASE_AT]->
            (:Phase {name: "exploration"})
              -[:TRANSITIONED_TO {probability: 0.65, trigger: "hybrid"}]->
                (:Phase {name: "convergence"})
      -[:CURRENT_PHASE]->
        (:Phase {name: "convergence"})
      -[:HAS_MEMORY]->
        (:Memory {tier: "short", importance: 0.9, text_summary: "User wants Q3 earnings..."})
      -[:MEMBER_OF {weight: 0.8}]->
        (:Cluster {cluster_id: "cluster-finance", strategy: "weighted_avg"})
          -[:AGGREGATED_STATE]->
            (:SemanticState {vector: [...], step: 120})
```

---

## 5. PSS as Neo4j-Native Persistence Layer

This section describes how each of PSS's in-memory data structures maps to Neo4j graph operations, replacing Python deques, dicts, and lists with Cypher queries.

### 5.1 Semantic State History (replaces deque in PSS_State_V4)

**Current PSS:** `self.state_history = deque(maxlen=50)` -- a rolling window of past semantic state vectors.

**Neo4j replacement:** A linked list of `:SemanticState` nodes connected by `:STATE_HISTORY` relationships, with a `:CURRENT_STATE` pointer from the session.

**Write new state:**
```cypher
// Called on each PSS update step
MATCH (session:AgentSession {session_id: $session_id})-[:CURRENT_STATE]->(prev:SemanticState)
CREATE (new:SemanticState {
  state_id: $state_id,
  vector: $vector,
  timestamp: datetime(),
  step: $step,
  beta: $beta,
  mean_similarity: $mean_sim,
  variance: $variance
})
CREATE (new)-[:STATE_HISTORY {cosine_similarity: $cosine_sim}]->(prev)
// Move the CURRENT_STATE pointer
WITH session, new, prev
MATCH (session)-[old_rel:CURRENT_STATE]->(prev)
DELETE old_rel
CREATE (session)-[:CURRENT_STATE]->(new)
```

**Read recent history (replaces deque iteration):**
```cypher
MATCH (session:AgentSession {session_id: $session_id})-[:CURRENT_STATE]->(current:SemanticState)
MATCH path = (current)-[:STATE_HISTORY*0..49]->(ancestor:SemanticState)
RETURN [node IN nodes(path) | {
  vector: node.vector,
  step: node.step,
  beta: node.beta,
  timestamp: node.timestamp
}] AS state_history
```

**Advantage over deque:** History is never lost. The maxlen=50 window becomes a query parameter, not a data deletion. Old states remain in the graph for long-term analysis. Time-travel queries become trivial.

### 5.2 Phase Transition History (replaces list in PhaseDetector)

**Current PSS:** `self.phase_history = []` with appended (phase, timestamp) tuples. The 6x6 Markov transition matrix is computed from this list.

**Neo4j replacement:** A chain of `:Phase` nodes connected by `:TRANSITIONED_TO` relationships. The Markov matrix is computed via Cypher aggregation.

**Compute Markov transition matrix from graph:**
```cypher
// Count all transitions between phase pairs for this session
MATCH (session:AgentSession {session_id: $session_id})
MATCH (session)-[:CURRENT_PHASE|PHASE_AT*]->(p1:Phase)-[:TRANSITIONED_TO]->(p2:Phase)
WITH p1.name AS from_phase, p2.name AS to_phase, count(*) AS transitions
WITH from_phase, collect({to: to_phase, count: transitions}) AS targets,
     sum(transitions) AS total
RETURN from_phase,
       [t IN targets | {to: t.to, probability: toFloat(t.count) / total}] AS transition_probs
```

**Advantage:** Transition probabilities are always up to date. No need to maintain a separate matrix. Cross-session phase patterns become queryable.

### 5.3 Memory Tiers (replaces MultiResolutionMemory deques)

**Current PSS:** Three deques -- `short_term(maxlen=15)`, `medium_term(maxlen=50)`, `long_term(maxlen=100)` -- with K-means++ consolidation.

**Neo4j replacement:** `:Memory` nodes with `tier` property, connected to sessions via `:HAS_MEMORY`. Tier capacity is enforced by Cypher queries that prune lowest-importance items when tier is full.

**Consolidation (K-means++ via graph):**
```cypher
// Find memories in short-term tier for this session, ordered by importance
MATCH (session:AgentSession {session_id: $session_id})-[:HAS_MEMORY]->(m:Memory {tier: 'short'})
WITH m ORDER BY m.importance DESC
WITH collect(m) AS memories
WHERE size(memories) > 15
// Bottom N memories get consolidated
WITH memories[15..] AS to_consolidate
UNWIND to_consolidate AS mem
SET mem.tier = 'medium',
    mem.importance = mem.importance * 0.8  // decay on consolidation
```

**Vector-based memory retrieval (replaces linear scan):**
```cypher
MATCH (session:AgentSession {session_id: $session_id})-[:HAS_MEMORY]->(m:Memory)
WHERE m.tier IN ['short', 'medium', 'long']
CALL db.index.vector.queryNodes('memory_content_vector', $k, $query_vector)
YIELD node, score
WHERE node = m
RETURN m.memory_id, m.text_summary, m.tier, m.importance, score
ORDER BY score * m.importance DESC
LIMIT $limit
```

**Advantage:** Memory retrieval is O(log n) via vector index instead of O(n) linear scan. Cross-session memory sharing becomes possible via graph relationships. Memory importance can factor in graph centrality.

### 5.4 Cluster Aggregation (replaces in-memory ClusterManager)

**Current PSS:** `ClusterManager` holds a dict of `LocalPSSInstance` references, computes weighted average or attention-based aggregation in memory.

**Neo4j replacement:** Aggregation computed via Cypher over cluster member states.

**Weighted average aggregation:**
```cypher
MATCH (c:Cluster {cluster_id: $cluster_id})<-[mem:MEMBER_OF]-(session:AgentSession)
      -[:CURRENT_STATE]->(state:SemanticState)
WITH c, collect({vector: state.vector, weight: mem.weight}) AS member_states,
     sum(mem.weight) AS total_weight
// Weighted average computed application-side from returned vectors and weights
// (Neo4j does not natively support element-wise vector arithmetic in Cypher)
RETURN member_states, total_weight
```

Note: Element-wise vector averaging must be performed application-side after retrieving member vectors, since Cypher does not support per-element vector operations. This is a thin computation layer on top of the graph query.

### 5.5 Consensus Protocol (replaces in-memory ConsensusManager)

**Current PSS:** `ConsensusManager` collects votes from regional members, applies weighted voting, checks threshold 0.6.

**Neo4j replacement:** Votes stored as relationships, consensus computed via aggregation.

```cypher
// Record a vote
MATCH (session:AgentSession {session_id: $session_id})
MATCH (ce:ConsensusEvent {event_id: $consensus_event_id})
CREATE (session)-[:VOTED_IN {vote: $vote_value, weight: $agent_weight}]->(ce)

// Compute consensus
MATCH (ce:ConsensusEvent {event_id: $consensus_event_id})<-[v:VOTED_IN]-(session)
WITH ce, sum(v.vote * v.weight) / sum(v.weight) AS weighted_score, count(v) AS vote_count
SET ce.drift_score_consensus = weighted_score,
    ce.vote_count = vote_count,
    ce.outcome = CASE WHEN weighted_score >= ce.threshold_used THEN 'agreed' ELSE 'disagreed' END
RETURN ce.outcome, weighted_score, vote_count
```

---

## 6. MCP Tool Integration

New tools to be added to the Neo4j MCP Server, exposing PSS drift detection to all 20+ framework integrations without any per-framework changes.

### 6.1 Tool Definitions

#### Layer 1: Session-Level Drift Detection

| Tool Name              | Parameters                                               | Returns                                                    | Description                                    |
|------------------------|----------------------------------------------------------|------------------------------------------------------------|------------------------------------------------|
| `detect_drift`         | `session_id` (str), `input_text` (str)                   | `{drift_score, topic_switch, phase, severity, state_id}`   | Process new input, update semantic state, return drift analysis |
| `get_phase`            | `session_id` (str)                                       | `{phase, entered_at, srs, tc, fsm, be, markov_prob}`      | Get current conversation phase for a session   |
| `get_drift_score`      | `session_id` (str)                                       | `{drift_score, components: {topic, mean_sim, variance}}`   | Get current API-level drift score              |
| `query_drift_history`  | `session_id` (str), `limit` (int), `min_severity` (str)  | `[{event_id, timestamp, drift_score, severity, topic_switch}]` | Query past drift events with filters       |
| `get_state_trajectory` | `session_id` (str), `steps` (int)                        | `[{step, beta, mean_similarity, variance, phase}]`         | Get recent semantic state evolution            |

#### Layer 2: Cluster Operations

| Tool Name              | Parameters                                               | Returns                                                    | Description                                    |
|------------------------|----------------------------------------------------------|------------------------------------------------------------|------------------------------------------------|
| `cluster_state`        | `cluster_id` (str)                                       | `{cluster_id, member_count, coupling_strength, aggregated_drift, strategy}` | Get cluster-level aggregated state |
| `cluster_members`      | `cluster_id` (str)                                       | `[{session_id, agent_id, weight, current_phase, drift_score}]` | List all members with their current states |

#### Layer 3: Regional Operations

| Tool Name              | Parameters                                               | Returns                                                    | Description                                    |
|------------------------|----------------------------------------------------------|------------------------------------------------------------|------------------------------------------------|
| `consensus_status`     | `region_id` (str)                                        | `{latest_consensus: {outcome, score, votes}, pending_count}` | Get consensus status for a region           |
| `region_drift_summary` | `region_id` (str), `window_minutes` (int)                | `{clusters: [{id, drift_score}], regional_drift, consensus_events}` | Regional drift overview                  |

#### Layer 4: Global Observation

| Tool Name              | Parameters                                               | Returns                                                    | Description                                    |
|------------------------|----------------------------------------------------------|------------------------------------------------------------|------------------------------------------------|
| `network_anomalies`    | `observer_id` (str), `window_minutes` (int)              | `[{type, severity, affected_clusters, timestamp}]`         | Get detected anomalies across the network      |
| `global_drift_map`     | `observer_id` (str)                                      | `{regions: [{id, drift_score, anomaly_count}], systemic_drift_score}` | Global view of drift across all regions |

#### Memory Operations

| Tool Name              | Parameters                                               | Returns                                                    | Description                                    |
|------------------------|----------------------------------------------------------|------------------------------------------------------------|------------------------------------------------|
| `memory_consolidate`   | `session_id` (str)                                       | `{consolidated: int, short: int, medium: int, long: int}`  | Trigger memory tier consolidation              |
| `memory_query`         | `session_id` (str), `query` (str), `tier` (str), `limit` (int) | `[{memory_id, text, tier, importance, similarity}]`  | Semantic search across memory tiers            |

#### Session Management

| Tool Name              | Parameters                                               | Returns                                                    | Description                                    |
|------------------------|----------------------------------------------------------|------------------------------------------------------------|------------------------------------------------|
| `create_pss_session`   | `agent_id` (str), `cluster_id` (str, optional)           | `{session_id, agent_id, cluster_id}`                       | Initialize a new PSS-tracked session           |
| `end_pss_session`      | `session_id` (str)                                       | `{status, final_phase, total_drift_events, duration}`      | Close session and return summary               |

### 6.2 MCP Tool Registration Pattern

These tools follow the same pattern as the existing Neo4j MCP server tools. They would be registered as additional tool definitions in the MCP server configuration:

```
Neo4j MCP Server
  |
  +-- Existing tools:
  |     get-schema
  |     execute-query (read)
  |     execute-write-query
  |
  +-- PSS Drift tools (new):
        detect_drift
        get_phase
        get_drift_score
        query_drift_history
        get_state_trajectory
        cluster_state
        cluster_members
        consensus_status
        region_drift_summary
        network_anomalies
        global_drift_map
        memory_consolidate
        memory_query
        create_pss_session
        end_pss_session
```

### 6.3 Framework Usage Example (Generic MCP Pattern)

Since all 20+ frameworks connect via MCP, the usage pattern is identical regardless of framework:

```
Agent instruction: "You have access to drift detection tools. Before responding
to the user, call detect_drift with the user's message. If drift_score > 0.7
or topic_switch is true, call memory_consolidate to reorganize context, then
adjust your response strategy based on the current phase from get_phase."
```

No framework-specific SDK code is needed. The tools appear alongside the existing Neo4j query tools in whatever MCP client the framework uses.

---

## 7. Enhancement of neo4j-agent-memory

The existing `neo4j-agent-memory` library (https://github.com/neo4j-labs/agent-memory) provides passive memory storage -- entity extraction and semantic memory with no monitoring of how that memory is being used or when it becomes stale. PSS adds an active monitoring layer.

### 7.1 Drift-Triggered Memory Consolidation

**Current behavior:** neo4j-agent-memory stores entities and relationships as they are extracted from conversation. There is no trigger mechanism; storage is purely append-based.

**PSS enhancement:** When PSS detects a topic switch (Layer 1 cosine drop > 0.3), it triggers memory consolidation:

1. Short-term memories from the previous topic are evaluated for importance.
2. High-importance memories are promoted to medium-term with updated importance scores.
3. Medium-term memories that have been accessed frequently are promoted to long-term.
4. The memory vector index is updated to reflect the new topic context.

This means memory reorganization happens at natural conversation boundaries (topic switches) rather than at arbitrary intervals or capacity limits.

### 7.2 Phase-Aware Memory Retrieval

**Current behavior:** neo4j-agent-memory retrieves memories based on vector similarity to the current query. There is no awareness of what phase the conversation is in.

**PSS enhancement:** Memory retrieval is weighted by conversation phase:

| Phase          | Retrieval Strategy                                                 |
|----------------|--------------------------------------------------------------------|
| initialization | Favor long-term memories (established context)                     |
| exploration    | Broad retrieval across all tiers, higher diversity                  |
| convergence    | Favor recent short-term memories, narrow to current topic          |
| resonance      | Balanced retrieval, emphasize high-importance across tiers          |
| stability      | Minimal retrieval, current context is sufficient                    |
| instability    | Aggressive retrieval across all tiers, attempt to re-anchor context |

This is implemented as a phase-dependent weighting function applied to the memory query results.

### 7.3 Quality Scoring for Stored Memories

**Current behavior:** neo4j-agent-memory does not score the quality or relevance of stored memories over time.

**PSS enhancement:** Each memory node gets an evolving quality score:

```
quality = importance * recency_decay * access_frequency * phase_relevance
```

Where:
- `importance`: Initial extraction confidence from neo4j-agent-memory
- `recency_decay`: Exponential decay based on time since creation
- `access_frequency`: How often this memory is retrieved (log-scaled)
- `phase_relevance`: How relevant this memory is to the current conversation phase

Low-quality memories are candidates for consolidation or pruning. This prevents memory bloat while preserving genuinely useful context.

### 7.4 Integration Architecture

```
  neo4j-agent-memory (existing)          PSS Layer (new)
  ==============================          ===============

  +-----------------------+               +-------------------+
  | MemoryClient          |               | PSSMemoryMonitor  |
  |                       |  <-- event -->|                   |
  | - store_memory()      |    hooks      | - on_drift()      |
  | - search_memory()     |               | - on_phase_change()|
  | - extract_entities()  |               | - score_quality() |
  +-----------+-----------+               +---------+---------+
              |                                     |
              v                                     v
  +-----------------------+               +-------------------+
  | Neo4j Graph           |               | Same Neo4j Graph  |
  | (:Memory) nodes       | <-- shared -->| (:SemanticState)  |
  | (:Entity) nodes       |    graph      | (:DriftEvent)     |
  | (:Relationship) edges |               | (:Phase)          |
  +-----------------------+               +-------------------+
```

The key insight: both systems write to the same Neo4j graph. PSS monitors the conversation flow and triggers operations on the memory nodes that neo4j-agent-memory created. No data duplication; shared graph.

---

## 8. Graph Algorithm Synergies

Neo4j's Graph Data Science (GDS) library provides algorithms that directly enhance PSS capabilities beyond what the in-memory implementation can achieve.

### 8.1 PageRank on Agent Influence Networks

**Use case:** In a MetaPSS multi-agent system, some agents are more influential than others. Their drift events propagate more widely; their phase transitions affect cluster state more.

**Implementation:** Project agent sessions and their drift correlations as a graph, then run PageRank.

```cypher
// Project drift correlation network
MATCH (s1:AgentSession)-[:CURRENT_STATE]->(st1:SemanticState)-[:TRIGGERED]->(d1:DriftEvent)
      -[:CORRELATED_DRIFT]->(d2:DriftEvent)<-[:TRIGGERED]-(st2:SemanticState)
      <-[:CURRENT_STATE]-(s2:AgentSession)
WHERE s1 <> s2
WITH s1, s2, count(*) AS correlation_count, avg(d1.drift_score) AS avg_drift
// Use GDS to compute PageRank on this projected network
// Agents with high PageRank are "drift leaders" -- their state changes cascade
```

**Value:** Identifies which agents in a multi-agent system are most likely to cause systemic drift. These agents can receive tighter monitoring or different consensus weights.

### 8.2 Community Detection on Drift Correlation Patterns

**Use case:** Identify groups of agents that drift together. These natural communities may represent agents working on related tasks, or they may reveal unwanted coupling.

**Implementation:** Run Louvain community detection on the drift correlation graph.

```cypher
// Agents that frequently drift together form communities
// Louvain reveals these clusters, which may differ from the
// manually assigned MetaPSS clusters
```

**Value:** Validates or challenges the manual cluster assignments in MetaPSS. If Louvain communities do not align with MetaPSS clusters, the cluster topology may need adjustment.

### 8.3 Temporal Path Analysis on State Evolution

**Use case:** Analyze how semantic states evolve over time. Find the shortest path between two semantic states to understand what sequence of topic changes led from one context to another.

**Implementation:** Use Neo4j path-finding on the `:STATE_HISTORY` chain.

```cypher
// Find the state evolution path between two known states
MATCH path = shortestPath(
  (s1:SemanticState {state_id: $start_state})-[:STATE_HISTORY*]->
  (s2:SemanticState {state_id: $end_state})
)
RETURN [node IN nodes(path) | {
  step: node.step,
  beta: node.beta,
  timestamp: node.timestamp
}] AS evolution_path,
[rel IN relationships(path) | rel.cosine_similarity] AS similarity_chain
```

**Value:** Enables "conversation replay" -- understand how the agent got from context A to context B. Useful for debugging agent behavior and for audit trails.

### 8.4 Vector Similarity on Semantic States

**Use case:** Find sessions with similar semantic states across the entire agent network. This enables "experience transfer" -- if one agent has successfully navigated a similar conversation state, its memory and phase history can inform another agent.

**Implementation:** Use Neo4j's native vector index.

```cypher
// Find sessions with similar current states
MATCH (target:AgentSession {session_id: $session_id})-[:CURRENT_STATE]->(target_state:SemanticState)
CALL db.index.vector.queryNodes('semantic_state_vector', 10, target_state.vector)
YIELD node AS similar_state, score
WHERE score > 0.85
MATCH (other_session:AgentSession)-[:CURRENT_STATE]->(similar_state)
WHERE other_session.session_id <> $session_id
RETURN other_session.session_id, similar_state.step, score,
       [(other_session)-[:CURRENT_PHASE]->(p:Phase) | p.name][0] AS phase
ORDER BY score DESC
```

**Value:** Cross-session learning. An agent struggling with instability can discover that a peer agent in a similar state is in the convergence phase, and adopt its strategy.

### 8.5 Betweenness Centrality on Phase Transition Networks

**Use case:** Identify which phases are critical "bridge" points in conversation flow. A phase with high betweenness centrality is one that many conversation paths must pass through.

**Value:** Understanding that "exploration" has high betweenness centrality (most conversations pass through it between initialization and convergence) helps optimize agent prompts for that phase.

### 8.6 Node Similarity for Memory Deduplication

**Use case:** Identify near-duplicate memories across sessions or within a single session's long-term memory.

**Implementation:** Run Node Similarity on `:Memory` nodes using their content vectors.

**Value:** Prevents memory bloat. When two memories are 95%+ similar, they can be merged, keeping the higher-importance version and incrementing its access count.

---

## 9. Multi-Agent Coordination via Graph

The graph structure provides capabilities that PSS's in-memory implementation cannot achieve.

### 9.1 Cluster Topology as Queryable Graph

**In-memory limitation:** PSS's `ClusterManager` maintains an in-memory dict of `LocalPSSInstance` references. There is no way to query "which clusters have the highest drift?" or "which agents are in multiple clusters?" without iterating all structures.

**Graph advantage:** The cluster topology is a first-class queryable structure.

```cypher
// Find clusters experiencing high drift
MATCH (c:Cluster)<-[:MEMBER_OF]-(s:AgentSession)-[:CURRENT_STATE]->(st:SemanticState)
      -[:TRIGGERED]->(d:DriftEvent)
WHERE d.timestamp > datetime() - duration('PT1H')
  AND d.severity IN ['high', 'critical']
WITH c, count(d) AS high_drift_events, avg(d.drift_score) AS avg_drift
RETURN c.cluster_id, c.name, high_drift_events, avg_drift
ORDER BY avg_drift DESC

// Find agents that bridge multiple clusters (potential coordination bottlenecks)
MATCH (s:AgentSession)-[:MEMBER_OF]->(c:Cluster)
WITH s, collect(c.cluster_id) AS clusters, count(c) AS cluster_count
WHERE cluster_count > 1
RETURN s.session_id, s.agent_id, clusters, cluster_count
ORDER BY cluster_count DESC
```

### 9.2 Consensus History as Auditable Trail

**In-memory limitation:** PSS's `ConsensusManager` computes votes and discards them. There is no history of how consensus was reached, who voted what, or whether consensus outcomes were correct.

**Graph advantage:** Every consensus event, every vote, and every outcome is persisted as a queryable trail.

```cypher
// Audit trail: How was this drift event handled by consensus?
MATCH (de:DriftEvent {event_id: $drift_event_id})
      <-[:CONSENSUS_ON]-(ce:ConsensusEvent)
      <-[v:VOTED_IN]-(s:AgentSession)
RETURN ce.outcome, ce.drift_score_consensus,
       collect({agent: s.agent_id, vote: v.vote, weight: v.weight}) AS votes
```

### 9.3 Cross-Region Drift Patterns via Graph Queries

**In-memory limitation:** PSS's `GlobalObserver` can detect anomalies but cannot correlate drift patterns across regions over time without loading all data into memory.

**Graph advantage:** Temporal graph queries reveal cross-region drift cascades.

```cypher
// Detect drift cascade: region A drifts, then region B drifts within 5 minutes
MATCH (r1:Region)-[:DRIFT_EVENT]->(d1:DriftEvent)
MATCH (r2:Region)-[:DRIFT_EVENT]->(d2:DriftEvent)
WHERE r1 <> r2
  AND d2.timestamp > d1.timestamp
  AND d2.timestamp < d1.timestamp + duration('PT5M')
  AND d1.severity IN ['high', 'critical']
  AND d2.severity IN ['high', 'critical']
WITH r1, r2, count(*) AS cascade_count
WHERE cascade_count > 3
RETURN r1.name AS source_region, r2.name AS affected_region, cascade_count
ORDER BY cascade_count DESC
```

### 9.4 Dynamic Cluster Rebalancing

**Graph advantage:** Because cluster membership is a relationship, not a hard-coded assignment, agents can be dynamically reassigned based on semantic similarity.

```cypher
// Find agents whose current state is more similar to another cluster's centroid
MATCH (s:AgentSession)-[:MEMBER_OF]->(current_cluster:Cluster)
MATCH (s)-[:CURRENT_STATE]->(state:SemanticState)
MATCH (other_cluster:Cluster)-[:AGGREGATED_STATE]->(agg_state:SemanticState)
WHERE other_cluster <> current_cluster
WITH s, current_cluster, other_cluster, state,
     gds.similarity.cosine(state.vector, agg_state.vector) AS similarity_to_other
WHERE similarity_to_other > 0.9
RETURN s.session_id, current_cluster.cluster_id AS current,
       other_cluster.cluster_id AS better_fit, similarity_to_other
```

---

## 10. Implementation Phases

### Phase 1: Graph Model and Persistence Layer (Weeks 1-3)

**Objective:** Establish the Neo4j graph schema and replace PSS's in-memory persistence with Neo4j-backed storage.

**Deliverables:**

| Step | Task                                                    | File/Location                                            | Dependencies | Risk   |
|------|---------------------------------------------------------|----------------------------------------------------------|-------------|--------|
| 1.1  | Create graph schema DDL script (constraints, indexes)   | `pss-drift-detection/schema/pss_schema.cypher`           | None        | Low    |
| 1.2  | Implement Neo4jPSSStateStore (replaces deque)            | `pss-drift-detection/src/persistence/state_store.py`     | 1.1         | Medium |
| 1.3  | Implement Neo4jPhaseStore (replaces phase_history list)  | `pss-drift-detection/src/persistence/phase_store.py`     | 1.1         | Low    |
| 1.4  | Implement Neo4jMemoryStore (replaces MultiResolutionMemory deques) | `pss-drift-detection/src/persistence/memory_store.py` | 1.1      | Medium |
| 1.5  | Implement Neo4jSessionManager (session lifecycle)        | `pss-drift-detection/src/persistence/session_manager.py` | 1.1         | Low    |
| 1.6  | Create PSS-to-Neo4j adapter layer (facade over stores)   | `pss-drift-detection/src/persistence/adapter.py`         | 1.2-1.5    | Medium |
| 1.7  | Write integration tests against a test Neo4j instance    | `pss-drift-detection/tests/test_persistence.py`          | 1.6         | Low    |
| 1.8  | Benchmark: compare latency of Neo4j persistence vs in-memory | `pss-drift-detection/benchmarks/persistence_bench.py` | 1.6         | Low    |

**Key risk:** Vector index query latency for 384-dim vectors. Mitigation: benchmark early in step 1.8; if latency exceeds 50ms, consider caching the current state in-memory while persisting asynchronously.

### Phase 2: Core Drift Detection with Neo4j Backend (Weeks 3-5)

**Objective:** Port PSS's core drift detection (Layer 1) to run on top of the Neo4j persistence layer.

**Deliverables:**

| Step | Task                                                    | File/Location                                            | Dependencies | Risk   |
|------|---------------------------------------------------------|----------------------------------------------------------|-------------|--------|
| 2.1  | Adapt PSS_State_V4 to use Neo4jPSSStateStore             | `pss-drift-detection/src/core/pss_state.py`              | Phase 1     | Medium |
| 2.2  | Adapt PhaseDetector to use Neo4jPhaseStore                | `pss-drift-detection/src/core/phase_detector.py`         | Phase 1     | Medium |
| 2.3  | Adapt MultiResolutionMemory to use Neo4jMemoryStore       | `pss-drift-detection/src/core/memory.py`                 | Phase 1     | Medium |
| 2.4  | Implement DriftEvent creation and persistence             | `pss-drift-detection/src/core/drift_events.py`           | 2.1         | Low    |
| 2.5  | Implement the 3-layer drift score computation            | `pss-drift-detection/src/core/drift_score.py`            | 2.1, 2.2   | Low    |
| 2.6  | Implement beta calculation (9-step) with Neo4j state      | `pss-drift-detection/src/core/beta.py`                   | 2.1         | Medium |
| 2.7  | End-to-end test: single session drift detection           | `pss-drift-detection/tests/test_drift_detection.py`      | 2.1-2.6    | Low    |

**Key risk:** The 9-step beta calculation requires reading the last N states efficiently. Mitigation: the `:STATE_HISTORY` chain with depth-limited traversal (Section 5.1) handles this.

### Phase 3: MCP Tools for Framework Integration (Weeks 5-7)

**Objective:** Expose drift detection as MCP tools so all 20+ framework integrations gain drift awareness.

**Deliverables:**

| Step | Task                                                    | File/Location                                            | Dependencies | Risk   |
|------|---------------------------------------------------------|----------------------------------------------------------|-------------|--------|
| 3.1  | Define MCP tool schemas (JSON Schema for all tools)      | `pss-drift-detection/mcp/tool_schemas.json`              | None        | Low    |
| 3.2  | Implement detect_drift MCP tool handler                  | `pss-drift-detection/mcp/tools/detect_drift.py`          | Phase 2     | Low    |
| 3.3  | Implement get_phase, get_drift_score tool handlers        | `pss-drift-detection/mcp/tools/phase_tools.py`           | Phase 2     | Low    |
| 3.4  | Implement query_drift_history, get_state_trajectory       | `pss-drift-detection/mcp/tools/history_tools.py`         | Phase 2     | Low    |
| 3.5  | Implement memory_consolidate, memory_query                | `pss-drift-detection/mcp/tools/memory_tools.py`          | Phase 2     | Low    |
| 3.6  | Implement session management tools                       | `pss-drift-detection/mcp/tools/session_tools.py`         | Phase 2     | Low    |
| 3.7  | Register tools with Neo4j MCP server (extend config)      | `pss-drift-detection/mcp/register.py`                    | 3.2-3.6    | Medium |
| 3.8  | Test with LangGraph integration (reference framework)     | `pss-drift-detection/tests/test_mcp_langgraph.py`        | 3.7         | Medium |
| 3.9  | Test with OpenAI Agents SDK (second framework)            | `pss-drift-detection/tests/test_mcp_openai.py`           | 3.7         | Low    |
| 3.10 | Write integration README and example notebook             | `pss-drift-detection/README.md`                          | 3.7         | Low    |

**Key risk:** MCP server extension model. The Neo4j MCP server (https://github.com/neo4j/mcp) may need to support plugin tools. Mitigation: if the MCP server does not support plugins, deploy PSS drift tools as a separate MCP server that runs alongside the Neo4j MCP server. Both servers can be registered with the same MCP client (as shown in the LangGraph `MultiServerMCPClient` pattern).

### Phase 4: Multi-Agent Coordination -- MetaPSS Layers 2-4 (Weeks 7-10)

**Objective:** Implement cluster management, regional consensus, and global observation using the graph.

**Deliverables:**

| Step | Task                                                    | File/Location                                            | Dependencies | Risk   |
|------|---------------------------------------------------------|----------------------------------------------------------|-------------|--------|
| 4.1  | Implement ClusterManager with Neo4j backend               | `pss-drift-detection/src/network/cluster_manager.py`     | Phase 1     | Medium |
| 4.2  | Implement aggregation strategies (WeightedAvg, Attention) | `pss-drift-detection/src/network/aggregation.py`         | 4.1         | Medium |
| 4.3  | Implement RegionalManager with consensus protocol         | `pss-drift-detection/src/network/regional_manager.py`    | 4.1         | High   |
| 4.4  | Implement DriftEventBus using Neo4j as event store        | `pss-drift-detection/src/network/drift_event_bus.py`     | 4.3         | Medium |
| 4.5  | Implement GlobalObserver with anomaly detection           | `pss-drift-detection/src/network/global_observer.py`     | 4.3         | High   |
| 4.6  | Implement MCP tools for Layers 2-4 (cluster_state, etc.) | `pss-drift-detection/mcp/tools/network_tools.py`         | 4.1-4.5    | Low    |
| 4.7  | Implement UserPartitionManager for session routing        | `pss-drift-detection/src/network/partition_manager.py`    | 4.1         | Medium |
| 4.8  | Multi-agent integration test (5+ concurrent sessions)     | `pss-drift-detection/tests/test_multi_agent.py`          | 4.1-4.7    | High   |

**Key risk:** Consensus protocol performance under concurrent writes. Multiple agents writing votes simultaneously can cause write contention in Neo4j. Mitigation: use optimistic concurrency with retry logic; consider Neo4j's causal cluster for write distribution.

### Phase 5: Graph Algorithm Enhancements (Weeks 10-12)

**Objective:** Leverage GDS algorithms to provide analytics not possible with PSS alone.

**Deliverables:**

| Step | Task                                                    | File/Location                                            | Dependencies | Risk   |
|------|---------------------------------------------------------|----------------------------------------------------------|-------------|--------|
| 5.1  | Implement PageRank on agent influence network             | `pss-drift-detection/src/analytics/influence.py`         | Phase 4     | Low    |
| 5.2  | Implement Louvain community detection on drift correlations | `pss-drift-detection/src/analytics/communities.py`     | Phase 4     | Medium |
| 5.3  | Implement temporal path analysis on state evolution       | `pss-drift-detection/src/analytics/trajectories.py`      | Phase 2     | Low    |
| 5.4  | Implement cross-session similarity via vector search      | `pss-drift-detection/src/analytics/similarity.py`        | Phase 2     | Low    |
| 5.5  | Implement Node Similarity for memory deduplication        | `pss-drift-detection/src/analytics/deduplication.py`     | Phase 2     | Low    |
| 5.6  | Create analytics dashboard queries (Cypher templates)     | `pss-drift-detection/src/analytics/dashboard_queries.cy`  | 5.1-5.5   | Low    |
| 5.7  | Expose analytics as MCP tools                            | `pss-drift-detection/mcp/tools/analytics_tools.py`       | 5.1-5.5    | Low    |

**Key risk:** GDS library availability. Not all Neo4j deployments include GDS. Mitigation: make GDS-dependent features optional. Core drift detection (Phases 1-4) must work without GDS. Analytics (Phase 5) degrade gracefully when GDS is unavailable.

### Phase 6: Production Deployment Patterns (Weeks 12-14)

**Objective:** Provide deployment guides and production-ready configurations.

**Deliverables:**

| Step | Task                                                    | File/Location                                            | Dependencies | Risk   |
|------|---------------------------------------------------------|----------------------------------------------------------|-------------|--------|
| 6.1  | Docker Compose setup (Neo4j + PSS MCP server)            | `pss-drift-detection/deploy/docker-compose.yml`          | Phase 3     | Low    |
| 6.2  | AWS deployment guide (ECS/Fargate, following aws-agentcore pattern) | `pss-drift-detection/deploy/aws/README.md`     | Phase 3     | Medium |
| 6.3  | Neo4j Aura configuration guide                           | `pss-drift-detection/deploy/aura/README.md`              | Phase 3     | Low    |
| 6.4  | Performance tuning guide (connection pooling, caching)    | `pss-drift-detection/docs/performance.md`                | Phase 5     | Low    |
| 6.5  | Monitoring and observability setup (drift metrics export) | `pss-drift-detection/src/observability/metrics.py`       | Phase 4     | Medium |
| 6.6  | Security guide (authentication, authorization, data isolation) | `pss-drift-detection/docs/security.md`             | Phase 3     | Low    |
| 6.7  | Migration guide for existing neo4j-agent-memory users     | `pss-drift-detection/docs/migration.md`                  | Phase 2     | Low    |

---

## 11. Testing Strategy

### 11.1 Unit Tests

| Component                     | Test File                                          | What to Test                                              |
|-------------------------------|----------------------------------------------------|-----------------------------------------------------------|
| Neo4j persistence stores      | `tests/test_persistence.py`                        | CRUD operations, constraint enforcement, vector indexing   |
| Drift score computation       | `tests/test_drift_score.py`                        | 3-layer formula, edge cases (zero variance, identical vectors) |
| Phase detection               | `tests/test_phase_detection.py`                    | 6 phases, Markov transitions, hybrid scoring              |
| Memory consolidation          | `tests/test_memory.py`                             | Tier promotion, K-means++ clustering, quality scoring      |
| Aggregation strategies        | `tests/test_aggregation.py`                        | Weighted average, attention, empty cluster handling         |
| Consensus protocol            | `tests/test_consensus.py`                          | Threshold enforcement, vote weighting, timeout handling     |

### 11.2 Integration Tests

| Flow                          | Test File                                          | What to Test                                              |
|-------------------------------|----------------------------------------------------|-----------------------------------------------------------|
| End-to-end single session     | `tests/integration/test_single_session.py`         | Input -> drift detection -> Neo4j persistence -> MCP query |
| Multi-agent coordination      | `tests/integration/test_multi_agent.py`            | 5 agents, cluster formation, consensus, global observation |
| MCP tool round-trip           | `tests/integration/test_mcp_tools.py`              | Each MCP tool end-to-end with real Neo4j                  |
| neo4j-agent-memory integration | `tests/integration/test_memory_integration.py`    | Drift-triggered consolidation, phase-aware retrieval       |

### 11.3 End-to-End Tests

| User Journey                  | Test File                                          | What to Test                                              |
|-------------------------------|----------------------------------------------------|-----------------------------------------------------------|
| LangGraph + drift detection   | `tests/e2e/test_langgraph_drift.py`                | LangGraph agent uses detect_drift tool via MCP            |
| Multi-framework consistency   | `tests/e2e/test_cross_framework.py`                | Same drift tools produce same results via different MCP clients |
| Production deployment         | `tests/e2e/test_docker_deploy.py`                  | Docker Compose up, MCP health check, basic drift detection |

### 11.4 Performance Tests

| Scenario                      | Target                                             | Method                                                    |
|-------------------------------|----------------------------------------------------|-----------------------------------------------------------|
| Single-agent drift latency    | < 50ms per detect_drift call                       | Benchmark 1000 sequential calls                           |
| Concurrent sessions           | 1000 sessions, < 100ms p99 latency                 | Load test with concurrent MCP clients                     |
| Vector index query            | < 20ms for 384-dim cosine similarity top-10        | Benchmark with 100K SemanticState nodes                   |
| Consensus protocol            | < 500ms for 50-agent consensus round               | Simulate concurrent vote submission                       |

---

## 12. Risks and Mitigations

| Risk                                      | Severity | Likelihood | Mitigation                                                                                  |
|-------------------------------------------|----------|------------|---------------------------------------------------------------------------------------------|
| Vector index latency exceeds 50ms         | High     | Low        | Benchmark in Phase 1.8. Fallback: cache current state in-memory, persist async.             |
| Neo4j MCP server does not support plugins | High     | Medium     | Deploy PSS tools as a separate MCP server. Frameworks like LangGraph support MultiServerMCPClient. |
| Write contention during consensus         | Medium   | Medium     | Optimistic concurrency with retry. Batch vote writes. Consider causal cluster for writes.    |
| GDS library not available in all deployments | Medium | High       | Make Phase 5 features optional. Core drift detection works without GDS.                      |
| PSS embedding model mismatch              | Medium   | Low        | Make embedding dimensionality configurable. Support 384-dim (MiniLM) and 768-dim (larger models). |
| Existing neo4j-agent-memory API changes   | Low      | Medium     | Use event hooks/callbacks rather than modifying neo4j-agent-memory internals. Loose coupling. |
| Neo4j version incompatibility             | Medium   | Low        | Require Neo4j 5.11+ for vector index. Document minimum version clearly.                     |
| Conversation history chain grows unbounded | Medium  | High       | Implement configurable TTL on SemanticState nodes. Prune chains older than configurable window (e.g., 30 days). |

---

## 13. Success Criteria

- [ ] Single-agent drift detection works end-to-end via MCP with < 50ms latency.
- [ ] All 3 drift detection layers (topic switch, phase, API score) produce correct results validated against PSS's in-memory reference implementation.
- [ ] Phase detection correctly identifies all 6 phases with Markov+Rules hybrid (40/60 split).
- [ ] At least 2 framework integrations (LangGraph, OpenAI SDK) demonstrate drift tools via MCP without framework-specific code.
- [ ] Multi-agent coordination (5+ agents) completes consensus protocol via Neo4j persistence.
- [ ] Graph algorithms (PageRank, community detection) run on drift correlation networks and produce actionable insights.
- [ ] neo4j-agent-memory integration demonstrates drift-triggered consolidation and phase-aware retrieval.
- [ ] Docker Compose deployment starts and passes health checks in under 60 seconds.
- [ ] Documentation sufficient for a developer unfamiliar with PSS to integrate drift detection into a new framework in under 2 hours.

---

## 14. Repository Structure

```
pss-drift-detection/
|-- README.md                              # Integration guide (follows INTEGRATION_TEMPLATE.md)
|-- schema/
|   |-- pss_schema.cypher                  # Neo4j DDL: constraints, indexes, vector indexes
|-- src/
|   |-- core/
|   |   |-- pss_state.py                   # PSS_State_V4 adapted for Neo4j
|   |   |-- phase_detector.py              # PhaseDetector with Neo4j persistence
|   |   |-- drift_score.py                 # 3-layer drift score computation
|   |   |-- drift_events.py                # DriftEvent creation and persistence
|   |   |-- beta.py                        # 9-step beta calculation
|   |   |-- memory.py                      # MultiResolutionMemory with Neo4j backend
|   |-- persistence/
|   |   |-- adapter.py                     # Facade over all persistence stores
|   |   |-- state_store.py                 # Neo4jPSSStateStore
|   |   |-- phase_store.py                 # Neo4jPhaseStore
|   |   |-- memory_store.py                # Neo4jMemoryStore
|   |   |-- session_manager.py             # Neo4jSessionManager
|   |-- network/
|   |   |-- cluster_manager.py             # MetaPSS Layer 2
|   |   |-- aggregation.py                 # WeightedAvg, Attention strategies
|   |   |-- regional_manager.py            # MetaPSS Layer 3
|   |   |-- drift_event_bus.py             # Event bus backed by Neo4j
|   |   |-- global_observer.py             # MetaPSS Layer 4
|   |   |-- partition_manager.py           # Session routing
|   |-- analytics/
|   |   |-- influence.py                   # PageRank on agent networks
|   |   |-- communities.py                 # Louvain on drift correlations
|   |   |-- trajectories.py                # Temporal path analysis
|   |   |-- similarity.py                  # Cross-session vector search
|   |   |-- deduplication.py               # Node Similarity for memory
|   |   |-- dashboard_queries.cy           # Cypher templates for dashboards
|   |-- observability/
|   |   |-- metrics.py                     # Prometheus/OpenTelemetry metrics export
|-- mcp/
|   |-- tool_schemas.json                  # MCP tool definitions (JSON Schema)
|   |-- register.py                        # Tool registration with MCP server
|   |-- tools/
|   |   |-- detect_drift.py                # detect_drift handler
|   |   |-- phase_tools.py                 # get_phase, get_drift_score
|   |   |-- history_tools.py               # query_drift_history, get_state_trajectory
|   |   |-- memory_tools.py                # memory_consolidate, memory_query
|   |   |-- session_tools.py               # create_pss_session, end_pss_session
|   |   |-- network_tools.py               # cluster_state, consensus_status, etc.
|   |   |-- analytics_tools.py             # Graph algorithm-based tools
|-- tests/
|   |-- test_persistence.py
|   |-- test_drift_score.py
|   |-- test_phase_detection.py
|   |-- test_memory.py
|   |-- test_aggregation.py
|   |-- test_consensus.py
|   |-- integration/
|   |   |-- test_single_session.py
|   |   |-- test_multi_agent.py
|   |   |-- test_mcp_tools.py
|   |   |-- test_memory_integration.py
|   |-- e2e/
|   |   |-- test_langgraph_drift.py
|   |   |-- test_cross_framework.py
|   |   |-- test_docker_deploy.py
|-- benchmarks/
|   |-- persistence_bench.py
|-- deploy/
|   |-- docker-compose.yml
|   |-- aws/
|   |   |-- README.md
|   |-- aura/
|   |   |-- README.md
|-- docs/
|   |-- plans.md                           # This document
|   |-- performance.md
|   |-- security.md
|   |-- migration.md
```

---

## 15. Open Questions for Neo4j Product Team

1. **MCP server extensibility:** Does the Neo4j MCP server (https://github.com/neo4j/mcp) support registering custom tools as plugins, or should PSS drift tools be deployed as a separate MCP server? The MultiServerMCPClient pattern in LangGraph supports multiple servers, but a single server would be simpler for users.

2. **Neo4j Aura vector index availability:** Are 384-dim vector indexes available on all Aura tiers, or only on specific plans? This affects the minimum deployment requirement.

3. **GDS licensing:** GDS algorithms (PageRank, Louvain) require GDS licensing. Should Phase 5 features be positioned as a premium tier, or should we use the GDS Community Edition subset?

4. **neo4j-agent-memory integration depth:** Should PSS be a separate package that hooks into neo4j-agent-memory via events/callbacks, or should PSS become part of the neo4j-agent-memory library itself? The former is lower risk; the latter provides tighter integration.

5. **Multi-tenancy:** In a SaaS deployment, should PSS data be isolated per tenant via separate databases, separate labels/properties, or Neo4j's multi-database feature? This affects the graph schema.

6. **Embedding model standardization:** Should Neo4j standardize on a specific embedding model for PSS semantic states, or should the model be configurable? PSS defaults to all-MiniLM-L6-v2 (384-dim) but Neo4j's existing vector examples use various models including 768-dim and 1536-dim.

---

## 16. Glossary

| Term                  | Definition                                                                                     |
|-----------------------|------------------------------------------------------------------------------------------------|
| PSS                   | Persistent Semantic State -- compressed O(1) representation of conversation meaning            |
| Drift Score           | 0-1 value: 0.6 * topic_switch + 0.3 * (1 - mean_similarity) + 0.1 * variance                 |
| Topic Switch          | Detected when cosine similarity between consecutive semantic states drops > 0.3                 |
| Phase                 | One of 6 conversation phases: initialization, exploration, convergence, resonance, stability, instability |
| Beta                  | Adaptation rate calculated via 9-step process, controls how fast semantic state updates         |
| MetaPSS               | Multi-agent extension: LocalPSSInstance, ClusterManager, RegionalManager, GlobalObserver       |
| Coupling Strength     | 0-1 value indicating how tightly a cluster's agents are semantically linked                     |
| Consensus Threshold   | Default 0.6 -- the weighted vote score required for a region to agree on a drift classification |
| MCP                   | Model Context Protocol -- standardized interface for LLM tools                                  |
| GDS                   | Graph Data Science -- Neo4j's library for graph algorithms                                      |
| Neo4jSaver            | LangGraph's checkpointer that stores conversation state in Neo4j                                |
| neo4j-agent-memory    | Neo4j Labs library for entity extraction and semantic memory storage                            |

---

*End of Plan*