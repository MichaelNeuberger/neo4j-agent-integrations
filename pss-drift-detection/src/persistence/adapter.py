"""Facade adapter over all Neo4j persistence stores."""

from __future__ import annotations

from neo4j import Driver

from src.persistence.neo4j_session_store import Neo4jSessionStore
from src.persistence.neo4j_state_store import Neo4jStateStore
from src.persistence.neo4j_phase_store import Neo4jPhaseStore
from src.persistence.neo4j_drift_event_store import Neo4jDriftEventStore
from src.persistence.neo4j_memory_store import Neo4jMemoryStore


class Neo4jPSSAdapter:
    """Unified access to all PSS persistence stores.

    Provides a single entry point for the PSS core to interact with
    all Neo4j-backed stores (sessions, states, phases, drift events, memory).
    """

    def __init__(self, driver: Driver, database: str = "neo4j"):
        self.driver = driver
        self.database = database

        self.sessions = Neo4jSessionStore(driver, database)
        self.states = Neo4jStateStore(driver, database)
        self.phases = Neo4jPhaseStore(driver, database)
        self.drift_events = Neo4jDriftEventStore(driver, database)
        self.memories = Neo4jMemoryStore(driver, database)

    def apply_schema(self, schema_path: str) -> None:
        """Apply the PSS graph schema from a .cypher file."""
        with open(schema_path, "r") as f:
            schema_cypher = f.read()

        statements = []
        current: list[str] = []
        for line in schema_cypher.split("\n"):
            stripped = line.strip()
            if stripped.startswith("//") or not stripped:
                continue
            current.append(line)
            if stripped.endswith(";"):
                statements.append("\n".join(current).rstrip(";"))
                current = []

        with self.driver.session(database=self.database) as session:
            for stmt in statements:
                stmt = stmt.strip()
                if stmt:
                    try:
                        session.run(stmt).consume()
                    except Exception:
                        pass  # Constraints/indexes may already exist
