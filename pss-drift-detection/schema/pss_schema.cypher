// PSS Drift Detection - Neo4j Graph Schema
// Requires Neo4j 5.11+ for vector index support

// === Uniqueness Constraints ===

CREATE CONSTRAINT agent_session_unique IF NOT EXISTS
  FOR (s:AgentSession) REQUIRE s.session_id IS UNIQUE;

CREATE CONSTRAINT agent_unique IF NOT EXISTS
  FOR (a:Agent) REQUIRE a.agent_id IS UNIQUE;

CREATE CONSTRAINT semantic_state_unique IF NOT EXISTS
  FOR (s:SemanticState) REQUIRE s.state_id IS UNIQUE;

CREATE CONSTRAINT drift_event_unique IF NOT EXISTS
  FOR (d:DriftEvent) REQUIRE d.event_id IS UNIQUE;

CREATE CONSTRAINT phase_unique IF NOT EXISTS
  FOR (p:Phase) REQUIRE p.phase_id IS UNIQUE;

CREATE CONSTRAINT memory_unique IF NOT EXISTS
  FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE;

CREATE CONSTRAINT memory_cluster_unique IF NOT EXISTS
  FOR (mc:MemoryCluster) REQUIRE mc.cluster_id IS UNIQUE;

CREATE CONSTRAINT cluster_unique IF NOT EXISTS
  FOR (c:Cluster) REQUIRE c.cluster_id IS UNIQUE;

CREATE CONSTRAINT region_unique IF NOT EXISTS
  FOR (r:Region) REQUIRE r.region_id IS UNIQUE;

CREATE CONSTRAINT global_observer_unique IF NOT EXISTS
  FOR (go:GlobalObserver) REQUIRE go.observer_id IS UNIQUE;

CREATE CONSTRAINT consensus_event_unique IF NOT EXISTS
  FOR (ce:ConsensusEvent) REQUIRE ce.event_id IS UNIQUE;

CREATE CONSTRAINT anomaly_event_unique IF NOT EXISTS
  FOR (ae:AnomalyEvent) REQUIRE ae.event_id IS UNIQUE;

// === Vector Indexes ===

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

// === Temporal Indexes ===

CREATE INDEX drift_event_timestamp IF NOT EXISTS
  FOR (d:DriftEvent) ON (d.timestamp);

CREATE INDEX phase_entered IF NOT EXISTS
  FOR (p:Phase) ON (p.entered_at);

CREATE INDEX semantic_state_timestamp IF NOT EXISTS
  FOR (s:SemanticState) ON (s.timestamp);

CREATE INDEX consensus_event_timestamp IF NOT EXISTS
  FOR (ce:ConsensusEvent) ON (ce.timestamp);

CREATE INDEX anomaly_event_timestamp IF NOT EXISTS
  FOR (ae:AnomalyEvent) ON (ae.timestamp);

// === Composite Indexes ===

CREATE INDEX drift_severity_time IF NOT EXISTS
  FOR (d:DriftEvent) ON (d.severity, d.timestamp);

CREATE INDEX memory_tier_importance IF NOT EXISTS
  FOR (m:Memory) ON (m.tier, m.importance);

CREATE INDEX agent_session_status IF NOT EXISTS
  FOR (s:AgentSession) ON (s.status, s.last_active);
