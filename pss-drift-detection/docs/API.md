# PSS SDK Documentation

Give your LLM agent infinite memory in 3 lines of code.

PSS compresses any conversation — no matter how long — into a fixed-size context block (~150-350 tokens) that your LLM can use as a system prompt. Memory cost stays constant at O(1) per turn.

---

## Quick Start

```python
import httpx

pss = httpx.Client(
    base_url="https://pss.versino.de/api/v1",
    headers={"X-API-Key": "pss_your-key-here"},
)

# 1. Send user message → get compressed context
run = pss.post("/run", json={"message": "What is Kubernetes?"}).json()
session_id = run["session_id"]

# 2. Feed context to your LLM
llm_response = your_llm.complete(system=run["context"], user="What is Kubernetes?")

# 3. Store the response → PSS learns
pss.post("/store", json={"session_id": session_id, "response": llm_response})
```

That's it. On the next turn, pass the `session_id` back and PSS returns context that incorporates everything discussed so far.

---

## How It Works

```
    Your App                             PSS
      |                                   |
      |  POST /run  {"message": "..."}         |
      |---------------------------------->|  embed → retrieve memories → compress
      |  {session_id, context, ...}       |
      |<----------------------------------|
      |                                   |
      |  [you call your LLM with context] |
      |                                   |
      |  POST /store {session_id, response}    |
      |---------------------------------->|  learn from Q&A → update state
      |  {session_id}                     |
      |<----------------------------------|
      |                                   |
      |  POST /run  {message, session_id}      |
      |---------------------------------->|  (repeat — context grows smarter)
```

Every `/run` + `/store` cycle is one "turn." PSS remembers what matters and forgets noise.

---

## API Reference

### `POST /api/v1/run`

Send a user message, get context back.

**Request:**

```json
{
  "message": "How does Kubernetes scale?",
  "session_id": "abc-123",
  "response": "Previous LLM answer (optional)",
  "short_circuit_threshold": 0.85,
  "reset_context": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | **yes** | — | The user's message |
| `session_id` | string | no | — | Reuse an existing session. Omit to start a new one |
| `response` | string | no | — | Store the previous LLM response inline (saves a separate `/store` call) |
| `short_circuit_threshold` | float | no | `0.85` | Similarity threshold for short-circuit flag (-1.0 to 1.0) |
| `reset_context` | bool | no | `false` | Wipe all memories and restart the session from scratch |

**Response:**

```json
{
  "session_id": "abc-123",
  "context": "[PSS Context | Turn 12 | 8 memories]\nRelevant context:\n  1. [0.87] Kubernetes uses pods as the smallest deployable unit...\n  2. [0.72] Container orchestration manages lifecycle...",
  "top_similarity": 0.87,
  "short_circuit": true,
  "drift_score": 0.12,
  "drift_detected": false,
  "drift_phase": "stable"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Your session identifier (new or existing) |
| `context` | string | **Inject this into your LLM's system prompt.** Fixed ~150-350 tokens no matter how long the conversation |
| `top_similarity` | float | How closely the top stored memory matches this query (0.0 = no match, 1.0 = exact) |
| `short_circuit` | bool | `true` when `top_similarity >= threshold` — PSS already knows this topic well |
| `drift_score` | float | How much the topic has shifted (0.0 = same topic, 1.0 = completely different) |
| `drift_detected` | bool | `true` when `drift_score >= 0.5` |
| `drift_phase` | string | `stable` / `shifting` / `drifted` |

---

### `POST /api/v1/store`

Store the LLM's response so PSS can learn from it.

**Request:**

```json
{
  "session_id": "abc-123",
  "response": "Kubernetes scales horizontally by adding more pod replicas..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | **yes** | Session from a previous `/run` |
| `response` | string | **yes** | The LLM's answer |

**Response:**

```json
{
  "session_id": "abc-123"
}
```

---

### `GET /api/v1/health`

Check if the service is up. No API key needed.

```json
{"status": "ok", "active_sessions": 42, "version": "1.0.0"}
```

---

## Patterns

### Single-Request Mode

Skip the separate `/store` call by sending the previous LLM response along with the next message:

```python
# Turn 2 onward: store + run in one call
run = pss.post("/run", json={
    "message": "How does it handle failures?",
    "session_id": session_id,
    "response": previous_llm_response,  # stores the Q&A from last turn
}).json()
```

### Short-Circuit: Skip Expensive Lookups

If `short_circuit` is `true`, PSS already has strong context — you can skip external data sources:

```python
run = pss.post("/run", json={
    "message": user_query,
    "session_id": sid,
    "short_circuit_threshold": 0.90,
}).json()

if run["short_circuit"]:
    answer = llm.complete(system=run["context"], user=user_query)
else:
    extra_data = fetch_from_database(user_query)
    answer = llm.complete(system=run["context"] + extra_data, user=user_query)
```

### Drift Detection: Know When the Topic Changed

Use `drift_detected` to trigger context-sensitive actions when the user switches topics:

```python
run = pss.post("/run", json={
    "message": user_query,
    "session_id": sid,
}).json()

if run["drift_detected"]:
    # User changed topics — reload relevant data
    reload_context_for(user_query)
    run = pss.post("/run", json={
        "message": user_query,
        "session_id": sid,
        "reset_context": True,  # start fresh
    }).json()
```

| `drift_phase` | Score | What It Means |
|---------------|-------|---------------|
| `stable` | < 0.3 | Same topic as before |
| `shifting` | 0.3 - 0.5 | Topic is starting to drift |
| `drifted` | >= 0.5 | Different domain — consider resetting |

### Context Reset

Wipe a session's memory without creating a new session:

```python
pss.post("/run", json={
    "message": "Starting a completely new topic",
    "session_id": sid,
    "reset_context": True,
})
```

The session ID stays the same. All memories, phase, and semantic state are cleared.

---

## Error Handling

Errors return JSON with a `detail` field:

```json
{"detail": "Session not found"}
```

| Status | When |
|--------|------|
| 401 | Invalid or missing API key |
| 404 | Session ID doesn't exist |
| 422 | Bad request (empty message, missing required field) |

---

## Performance

| | |
|---|---|
| Throughput | ~330 req/s |
| Median latency | ~130ms |
| p99 latency | ~300ms |
| Context size | ~150-350 tokens (constant, any conversation length) |

---

## Multi-Agent API (Layer 2–4)

The following endpoints extend PSS for multi-agent systems. They build on top of
the base `/run` and `/store` endpoints — no changes to existing behavior.

### Layer 2 — Cluster Sessions

A **cluster** is a named group of agents sharing one PSS session.
`cluster_id` doubles as the PSS `session_id` — no separate session ID is issued.
When Agent A stores a finding, Agent B's next query against the same cluster
returns `short_circuit=True` — LLM and data source skipped.

#### `POST /api/v1/cluster/`

Create a cluster. Automatically creates the underlying shared PSS session.

```json
// Request
{
  "name": "fraud-investigation",
  "aggregation_mode": "weighted_average",
  "coupling_factor": 0.0
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | — | Human-readable cluster name |
| `aggregation_mode` | string | `"weighted_average"` | `"weighted_average"` (G2a) or `"attention"` (G2b) |
| `coupling_factor` | float | `0.0` | G4 feedback strength — 0.0 = disabled, 1.0 = full replacement |

```json
// Response 201
{
  "cluster_id":       "c1a2b3c4-...",
  "name":             "fraud-investigation",
  "aggregation_mode": "weighted_average",
  "coupling_factor":  0.0
}
```

> **Note:** `cluster_id` is also the PSS `session_id`. Use it directly with `/run`, `/store`, or the dedicated `/cluster/{cluster_id}/run` endpoint.

#### `GET /api/v1/cluster/`

List all clusters owned by the current API key.

```json
[
  {
    "cluster_id":   "c1a2b3c4-...",
    "name":         "fraud-investigation",
    "member_count": 3
  }
]
```

#### `GET /api/v1/cluster/{cluster_id}`

Get the current state of a cluster's shared session, including the G3 aggregate vector.

```json
{
  "cluster_id":        "c1a2b3c4-...",
  "name":              "fraud-investigation",
  "member_count":      3,
  "phase":             "exploration",
  "interaction_count": 12,
  "total_memories":    8,
  "aggregation_mode":  "weighted_average",
  "coupling_factor":   0.1,
  "aggregate_vector":  [0.12, -0.04, ...]
}
```

#### `DELETE /api/v1/cluster/{cluster_id}`

Delete a cluster and its shared session. Returns `{"deleted": true}`.

#### `POST /api/v1/cluster/{cluster_id}/feedback`

Apply G4 coupling feedback — blend the cluster session vector back into every member session.

Uses the cluster's configured `coupling_factor`:
- `0.0` — no effect (returns `sessions_updated: 0` immediately)
- `1.0` — fully replaces each member's vector with the cluster vector

```json
// Response
{ "cluster_id": "c1a2b3c4-...", "sessions_updated": 3 }
```

Sessions with uninitialised vectors are skipped. Call this after significant cluster-level findings have been stored.

#### `POST /api/v1/cluster/{cluster_id}/store`

Store a Q&A pair directly into the cluster session (e.g. for seeding known typologies).

```json
// Request
{
  "message":  "Find all transactions linked to shell companies",
  "response": "47 transactions above $10k linked to 12 BVI/Cayman shell companies."
}
```

#### `POST /api/v1/cluster/{cluster_id}/run`

Query the cluster's PSS session. Identical to `POST /api/v1/run` but `session_id` is
resolved automatically from `cluster_id` — no need to pass it separately.

```json
// Request
{
  "message":                 "List transactions linked to shell companies",
  "response":                "Previous LLM answer (optional — stores last turn inline)",
  "short_circuit_threshold": 0.85,
  "reset_context":           false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | **yes** | — | The query to run against the cluster session |
| `response` | string | no | — | Inline-store the previous LLM answer (saves a separate `/store` call) |
| `short_circuit_threshold` | float | no | `0.85` | Similarity threshold for short-circuit flag |
| `reset_context` | bool | no | `false` | Wipe cluster session memories and restart |

Response is identical to `POST /api/v1/run` — see that section for field descriptions.

#### `POST /api/v1/cluster/{cluster_id}/members`

Register a personal agent session as a member of this cluster.

```json
{ "session_id": "agent-session-id" }
```

#### `DELETE /api/v1/cluster/{cluster_id}/members/{session_id}`

Remove a member session from the cluster. Returns `{"removed": true}`.

---

### Layer 3 — Regional Consensus

A **region** groups multiple clusters. When a configurable fraction of clusters in
a region all detect drift, the region triggers a realignment (consensus protocol).
Drift events are async — no agent turn is blocked.

#### `POST /api/v1/region/`

Create a region.

```json
// Request
{
  "name":                  "emea-fraud",
  "consensus_threshold":   0.5,
  "vote_window_seconds":   60.0
}

// Response 201
{
  "region_id":       "r1a2b3c4-...",
  "name":            "emea-fraud",
  "meta_session_id": "m9f8e7d6-..."
}
```

| Field | Default | Description |
|---|---|---|
| `consensus_threshold` | `0.5` | Fraction of clusters that must report drift to trigger realignment |
| `vote_window_seconds` | `60.0` | How long a drift vote stays valid |

#### `GET /api/v1/region/`

List all regions owned by the current API key.

#### `GET /api/v1/region/{region_id}`

Get region state: cluster count, meta session, recent drift events, last realignment timestamp.

```json
{
  "region_id":            "r1a2b3c4-...",
  "name":                 "emea-fraud",
  "cluster_count":        3,
  "meta_session_id":      "m9f8e7d6-...",
  "consensus_threshold":  0.5,
  "recent_drift_events":  [...],
  "last_realignment":     1743000000.0
}
```

#### `DELETE /api/v1/region/{region_id}`

Delete a region and its meta session. Returns `{"deleted": true}`.

#### `POST /api/v1/region/{region_id}/clusters`

Add a cluster to this region. The region will now listen for drift events from that cluster.

```json
{ "cluster_id": "c1a2b3c4-..." }
```

#### `DELETE /api/v1/region/{region_id}/clusters/{cluster_id}`

Remove a cluster from the region.

#### `GET /api/v1/region/{region_id}/events?limit=20`

Return the most recent drift events received by this region.

```json
[
  {
    "cluster_id":  "c1a2b3c4-...",
    "region_id":   "r1a2b3c4-...",
    "drift_score": 0.72,
    "drift_phase": "drifted",
    "timestamp":   1743000000.0
  }
]
```

---

### Layer 4 — Global Observer (read-only)

The **observer** samples all registered regions periodically and detects cross-cluster
anomalies. Each API key gets exactly one observer — calling `POST /v1/observer/` again
returns the same observer (idempotent). The observer never writes to any cluster or member
session; it maintains only its own PSS meta-session (G8 pattern vector).

#### `POST /api/v1/observer/`

Create (or retrieve) the observer for this API key and register regions.

```json
// Request
{
  "sample_interval_seconds": 30.0,
  "region_ids": ["r1a2b3c4-...", "r2b3c4d5-..."]
}

// Response 201
{
  "observer_id":        "o1a2b3c4-...",
  "registered_regions": 2,
  "meta_session_id":    "m0a1b2c3-..."
}
```

#### `GET /api/v1/observer/summary`

Snapshot of all observed clusters and current anomaly counts.

```json
{
  "observer_id":              "o1a2b3c4-...",
  "registered_regions":       2,
  "total_clusters_observed":  6,
  "last_sample_at":           1743000000.0,
  "anomaly_count":            1,
  "recent_anomalies":         [...]
}
```

#### `POST /api/v1/observer/sample`

Trigger a manual sample immediately. Returns anomalies found in this sample.

#### `GET /api/v1/observer/anomalies?limit=20`

Return the most recent anomaly events (most recent first).

```json
[
  {
    "anomaly_id":           "a1b2c3d4-...",
    "timestamp":            1743000000.0,
    "anomaly_type":         "cross_cluster_convergence",
    "affected_cluster_ids": ["c1a2b3c4-...", "c2b3c4d5-...", "c3c4d5e6-..."],
    "description":          "3+ clusters across different regions in drifted state",
    "severity":             0.8
  }
]
```

| `anomaly_type` | Condition | Severity |
|---|---|---|
| `systemic_drift` | >50% of all observed clusters in `drifted` phase | proportional |
| `cross_cluster_convergence` | ≥3 clusters in different regions all drifting | 0.8 |
| `cluster_divergence` | A cluster has 0 interactions (inactive) | 0.3 |

#### `DELETE /api/v1/observer/anomalies`

Clear the anomaly log. Returns `{"cleared": N}`.

#### `POST /api/v1/observer/regions`

Register an additional region with the observer.

```json
{ "region_id": "r3c4d5e6-..." }
```

#### `DELETE /api/v1/observer/regions/{region_id}`

Unregister a region from the observer.

---

### Layer 1b — Session Control (Resonance Triggers, State Serialization, Drift Anchors, Template Pre-Initialization, Input Isolation, Policy Vectors, Synthetic Memory Injection)

These endpoints expose advanced per-session features. All require an API key.

#### `POST /api/v1/session/create`

Create a PSS session with optional template pre-initialization and policy vectors.

```json
// Request
{
  "dimension": 384,
  "model_name": "all-MiniLM-L6-v2",
  "use_attention": false,
  "use_cache": false,
  "use_meta_pss": false,
  "enable_topic_switch": true,
  "template_embedding": [0.12, -0.04, ...],
  "policy_vectors": [
    {"embedding": [0.5, 0.3, ...], "weight": 1.5, "promote": true}
  ]
}

// Response
{"session_id": "abc-123", "created": true}
```

#### `POST /api/v1/session/{id}/trigger` — Resonance Triggers

Add a resonance trigger. When matched, forces `β = β_min` for that turn and sets `importance = 1.0`.

```json
// Request (keyword)
{"keyword": "CRITICAL"}

// Request (embedding)
{"embedding": [0.1, -0.3, ...], "threshold": 0.75}

// Response
{"session_id": "abc-123", "trigger_count": 1}
```

#### `DELETE /api/v1/session/{id}/trigger` — Resonance Triggers

Remove all resonance triggers. Returns `{"session_id": "...", "trigger_count": 0}`.

#### `GET /api/v1/session/{id}/export` — State Serialization & Consistency Verification

Export the full PSS state as a JSON-serializable dict with a SHA-256 checksum over all numeric state (semantic vector, memory embeddings, adaptive parameters).

```json
{
  "session_id": "abc-123",
  "state_dict": { ... },
  "checksum": "a3f8b2c1..."
}
```

#### `POST /api/v1/session/{id}/import` — State Serialization & Consistency Verification

Restore a session from a previously exported state dict. Validates the checksum — returns `404` if tampered.

```json
{"state_dict": { ... }}
```

#### `POST /api/v1/session/{id}/verify` — State Serialization & Consistency Verification

Verify behavioral consistency after export/import by running test embeddings through both the original and restored session. Returns `{"consistent": true}` if all similarity scores match within tolerance.

```json
// Request
{"test_embeddings": [[0.1, -0.3, ...], ...]}
```

#### `POST /api/v1/session/{id}/anchor` — Drift Anchors

Add a semantic anchor for drift detection. The anchor score measures cosine similarity between the current semantic state and all stored anchors.

```json
{"embedding": [0.1, -0.3, ...]}
// Response: {"session_id": "...", "anchor_count": 1}
```

#### `GET /api/v1/session/{id}/anchor_score` — Drift Anchors

Return the current anchor score and drift status.

```json
{
  "session_id": "abc-123",
  "anchor_score": 0.87,
  "anchor_count": 2,
  "drift_threshold": 0.6,
  "realignment_remaining": 0
}
```

#### `PUT /api/v1/session/{id}/isolation` — Input Isolation Filter

Set the input isolation level. In `QUARANTINE` mode, blocked inputs go to a shadow buffer instead of being discarded.

```json
// Request
{
  "level": "QUARANTINE",
  "exclusion_embeddings": [[0.1, -0.3, ...]],
  "allowlist_embeddings": [],
  "similarity_threshold": 0.8
}
```

| Level | Behavior |
|-------|----------|
| `OPEN` | All inputs pass through |
| `FILTER` | Inputs matching exclusion criteria are discarded |
| `QUARANTINE` | Blocked inputs go to shadow buffer |
| `LOCKDOWN` | All inputs blocked |

#### `POST /api/v1/session/{id}/isolation/release` — Input Isolation Filter

Release shadow-buffered items from quarantine. Returns `{"released_count": N}`.

#### `POST /api/v1/session/{id}/memory` — Synthetic Memory Injection

Inject a memory entry directly into a tier, bypassing the normal state update. Useful for seeding domain knowledge.

```json
// Request
{
  "embedding": [0.1, -0.3, ...],
  "text": "Kubernetes scales horizontally via pod replicas",
  "tier": "long_term",
  "importance": 1.0,
  "access_count": 0
}
// Response: {"session_id": "...", "tier": "long_term", "total_memories": 5}
```

| `tier` | Options |
|--------|---------|
| `short_term` | Fast-access recent memories |
| `medium_term` | Consolidated medium-importance items |
| `long_term` | Persistent domain knowledge |

---

### Layer 5 — Network (Semantic Delta-Vector Transfer, User-Isolated Instance Partitioning, Trust-Based Consensus)

Cross-session and cross-instance operations: delta transfer, user partitioning, trust-based consensus.

#### `POST /api/v1/network/transfer` — Semantic Delta-Vector Transfer

Transfer semantic delta from a source session to a target session. The delta is the difference between their normalized semantic states, scaled by `max_weight`.

```json
// Request
{
  "source_session_id": "abc-123",
  "target_session_id": "def-456",
  "max_weight": 0.15
}
// Response
{"source_session_id": "abc-123", "target_session_id": "def-456", "delta_norm": 0.423}
```

#### `POST /api/v1/network/users/switch` — User-Isolated Instance Partitioning

Switch the active user partition. The current partition's state is saved; the target partition is loaded (or created fresh if new).

```json
// Request
{"user_id": "alice"}
// Response
{"active_user": "alice", "previous_user": "default"}
```

#### `GET /api/v1/network/users/active` — User-Isolated Instance Partitioning

Return the currently active user ID: `{"active_user": "alice"}`.

#### `POST /api/v1/network/users/{user_id}/serialize` — User-Isolated Instance Partitioning

Serialize a user partition's PSS state to JSON.

```json
{"user_id": "alice", "state": { ... }}
```

#### `POST /api/v1/network/consensus` — Trust-Based Consensus

Propose a consensus vector from a session to all network instances. Each instance votes based on cosine similarity to its own state, weighted by the proposer's trust score.

```json
// Request
{
  "proposer_session_id": "abc-123",
  "target_embedding": [0.1, -0.3, ...]
}
// Response
{
  "accepted": true,
  "trust_scores": {"abc-123": 0.62, "def-456": 0.55}
}
```

#### `GET /api/v1/network/consensus/trust` — Trust-Based Consensus

Return current trust scores for all registered network instances.

```json
{"trust_scores": {"abc-123": 0.62, "def-456": 0.55}}
```

---

### Literal Cache — Verbatim Code-Entity Memory

The literal cache stores exact code artifacts (variable names, function signatures, file paths, error messages) verbatim — no semantic compression. Retrieval is token-overlap based: the LLM context block automatically includes matching entries for the current query turn.

This makes PSS suitable for coding assistants where exact identifier recall matters.

#### `POST /api/v1/session/{id}/entities`

Store a code entity. If the key already exists it is overwritten (last-write-wins).

```json
// Request
{
  "key": "PSS_State_V4._anchor_embeddings",
  "kind": "variable",
  "value": "list[np.ndarray]",
  "context": "in src/pss/core.py line 345",
  "importance": 1.5
}
// Response 201
{"session_id": "abc-123", "key": "PSS_State_V4._anchor_embeddings", "size": 3}
```

| `kind` | Examples |
|--------|---------|
| `variable` | `list[np.ndarray]`, `float = 0.5` |
| `function` | `(x: int) -> str` |
| `class` | `PSS_State_V4` |
| `path` | `src/pss/core.py` |
| `error` | `AttributeError: '_anchors'` |
| `signature` | `def update(self, embedding, text)` |
| `constant` | `MAX_CAPACITY = 200` |
| `type` | `np.ndarray` |

#### `GET /api/v1/session/{id}/entities?q=anchor`

Query entities whose key or value tokens match the query string. Omit `q` to list all.

```json
{
  "session_id": "abc-123",
  "entities": [
    {
      "key": "PSS_State_V4._anchor_embeddings",
      "kind": "variable",
      "value": "list[np.ndarray]",
      "context": "in src/pss/core.py line 345",
      "importance": 1.5,
      "access_count": 2
    }
  ],
  "total": 3
}
```

#### `DELETE /api/v1/session/{id}/entities/{key}`

Remove a code entity by its qualified key. Returns `404` if not found.

```json
{"session_id": "abc-123", "key": "PSS_State_V4._anchor_embeddings", "removed": true}
```

---

### Multi-Agent Quick Reference

```
Layer 2 — Cluster (cross-agent short-circuit):
  POST   /v1/cluster/                          create cluster (aggregation_mode, coupling_factor)
  GET    /v1/cluster/                          list clusters
  GET    /v1/cluster/{id}                      cluster state + aggregate_vector (G3)
  DELETE /v1/cluster/{id}                      delete cluster
  POST   /v1/cluster/{id}/run                  query cluster session (cluster_id == session_id)
  POST   /v1/cluster/{id}/feedback             apply G4 coupling feedback to members
  POST   /v1/cluster/{id}/store                seed Q&A into cluster session
  POST   /v1/cluster/{id}/members              register agent session
  DELETE /v1/cluster/{id}/members/{sid}        remove agent session

Layer 3 — Region (async drift consensus):
  POST   /v1/region/                           create region (consensus_threshold, vote_window_seconds)
  GET    /v1/region/                           list regions
  GET    /v1/region/{id}                       region state + realignment history
  DELETE /v1/region/{id}                       delete region
  POST   /v1/region/{id}/clusters              add cluster to region
  DELETE /v1/region/{id}/clusters/{cid}        remove cluster from region
  GET    /v1/region/{id}/events?limit=20       recent drift events

Layer 4 — Observer (read-only anomaly detection, per API key):
  POST   /v1/observer/                         create/retrieve observer, register regions
  GET    /v1/observer/summary                  full snapshot (includes meta_session_id)
  POST   /v1/observer/sample                   trigger manual sample
  GET    /v1/observer/anomalies?limit=20       list detected anomalies
  DELETE /v1/observer/anomalies                clear anomaly log
  POST   /v1/observer/regions                  register additional region
  DELETE /v1/observer/regions/{region_id}      unregister region

Layer 1b — Session Control (resonance triggers, state serialization, drift anchors, template pre-initialization, input isolation, policy vectors, synthetic memory injection):
  POST   /v1/session/create                            create session with template/policy options
  POST   /v1/session/{id}/trigger                      add resonance trigger (keyword or embedding)
  DELETE /v1/session/{id}/trigger                      clear all triggers
  GET    /v1/session/{id}/export                       export state (JSON + SHA-256 checksum)
  POST   /v1/session/{id}/import                       restore state from export
  POST   /v1/session/{id}/verify                       verify behavioral consistency after import
  POST   /v1/session/{id}/anchor                       add semantic drift anchor
  GET    /v1/session/{id}/anchor_score                 current drift score vs anchors
  PUT    /v1/session/{id}/isolation                    set input isolation level
  POST   /v1/session/{id}/isolation/release            release quarantine buffer
  POST   /v1/session/{id}/memory                       inject memory directly into a tier

Layer 5 — Network (semantic delta-vector transfer, user-isolated instance partitioning, trust-based consensus):
  POST   /v1/network/transfer                          transfer semantic delta between sessions
  POST   /v1/network/users/switch                      switch active user partition
  GET    /v1/network/users/active                      current active user
  POST   /v1/network/users/{user_id}/serialize         serialize a user partition's state
  POST   /v1/network/consensus                         propose trust-based consensus vector
  GET    /v1/network/consensus/trust                   current trust scores for all instances

Literal Cache — verbatim code-entity memory tier:
  POST   /v1/session/{id}/entities                     store code entity (key, kind, value)
  GET    /v1/session/{id}/entities?q=...               query entities by keyword (omit q for all)
  DELETE /v1/session/{id}/entities/{key}               remove entity by qualified key
```
