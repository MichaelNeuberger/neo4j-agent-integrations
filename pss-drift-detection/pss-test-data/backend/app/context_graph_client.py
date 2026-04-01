"""Neo4j context graph client."""

from __future__ import annotations

import asyncio
import json as _json
import logging
import threading

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.graph import Node, Relationship, Path


logger = logging.getLogger(__name__)

_driver: AsyncDriver | None = None
_memory_client = None  # MemoryClient | None — lazy typed to avoid import errors
_connected: bool = False


# ---------------------------------------------------------------------------
# Result collector — captures Cypher results from agent tool calls so the
# chat endpoint can attach them as graph_data without modifying agent templates.
# ---------------------------------------------------------------------------

class CypherResultCollector:
    """Collects Cypher query results for downstream graph visualization.

    When an event queue is attached (via ``set_event_queue``), the collector
    also pushes SSE-ready event dicts to the queue so the streaming endpoint
    can forward them to the frontend in real time.
    """

    def __init__(self):
        self.results: list[dict] = []
        self.tool_calls: list[dict] = []
        self._event_queue: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- event queue management ------------------------------------------------

    def set_event_queue(self, queue: asyncio.Queue) -> None:
        """Attach an asyncio.Queue for SSE streaming."""
        self._event_queue = queue
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def clear_event_queue(self) -> None:
        """Detach the event queue."""
        self._event_queue = None
        self._loop = None

    def _push_event(self, event: str, data: dict) -> None:
        """Push an SSE event to the queue if one is attached.

        Thread-safe: if called from a worker thread (e.g. CrewAI/Strands tools
        running via ``asyncio.to_thread``), uses ``call_soon_threadsafe`` to
        schedule the put on the event loop's thread.
        """
        if self._event_queue is None:
            return
        payload = {"event": event, "data": data}
        try:
            loop = self._loop
            if loop is not None and loop.is_running():
                # Detect whether we are on the event-loop thread or a worker.
                loop_thread = getattr(loop, "_thread_id", None)
                current_tid = threading.current_thread().ident
                if loop_thread is not None and current_tid != loop_thread:
                    loop.call_soon_threadsafe(self._event_queue.put_nowait, payload)
                    return
            self._event_queue.put_nowait(payload)
        except RuntimeError:
            pass  # loop closed or queue full — best effort

    # -- tool call events ------------------------------------------------------

    def emit_tool_start(self, name: str, inputs: dict) -> None:
        """Emit a tool_start SSE event."""
        self._push_event("tool_start", {"name": name, "inputs": inputs})

    def emit_text_delta(self, text: str) -> None:
        """Emit a text_delta SSE event."""
        self._push_event("text_delta", {"text": text})

    def emit_done(self, response_text: str, session_id: str = "") -> None:
        """Emit a done SSE event."""
        self._push_event("done", {"response": response_text, "session_id": session_id})

    # -- existing collection methods -------------------------------------------

    def collect(self, records: list[dict]) -> None:
        self.results.extend(records)

    def collect_tool_call(self, name: str, inputs: dict, output_preview: str = "") -> None:
        self.tool_calls.append({
            "name": name,
            "inputs": inputs,
            "output_preview": output_preview[:500],
        })
        # Emit tool_end SSE event with the graph data collected so far
        self._push_event("tool_end", {
            "name": name,
            "inputs": inputs,
            "output_preview": output_preview[:500],
            "graph_data": {"results": list(self.results)},
        })

    def drain(self) -> list[dict]:
        results = list(self.results)
        self.results.clear()
        return results

    def drain_tool_calls(self) -> list[dict]:
        calls = list(self.tool_calls)
        self.tool_calls.clear()
        return calls


_collector = CypherResultCollector()


def get_collector() -> CypherResultCollector:
    """Get the global result collector."""
    return _collector


def _serialize(value):
    """Serialize Neo4j graph types to JSON-friendly dicts."""
    if isinstance(value, Node):
        props = dict(value)
        props["elementId"] = value.element_id
        props["labels"] = list(value.labels)
        return props
    if isinstance(value, Relationship):
        props = dict(value)
        props["elementId"] = value.element_id
        props["type"] = value.type
        props["startNodeElementId"] = value.start_node.element_id
        props["endNodeElementId"] = value.end_node.element_id
        return props
    if isinstance(value, Path):
        return {
            "nodes": [_serialize(n) for n in value.nodes],
            "relationships": [_serialize(r) for r in value.relationships],
        }
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    return value


def is_connected() -> bool:
    """Check if Neo4j was successfully connected at startup."""
    return _connected


async def connect_neo4j() -> None:
    """Connect to Neo4j and optionally initialize the memory client."""
    global _driver, _memory_client, _connected
    from app.config import settings
    _driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    await _driver.verify_connectivity()
    _connected = True

    # Initialize neo4j-agent-memory MemoryClient for conversation persistence
    try:
        import os
        from pydantic import SecretStr
        from neo4j_agent_memory import MemoryClient, MemorySettings

        # Use OpenAI embeddings if API key is available, otherwise local sentence-transformers
        if os.getenv("OPENAI_API_KEY"):
            embedding_config = {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimensions": 1536,
            }
        else:
            embedding_config = {
                "provider": "sentence_transformers",
                "model": "all-MiniLM-L6-v2",
                "dimensions": 384,
                "device": "cpu",
            }
            logger.info("Using local sentence-transformers embeddings (set OPENAI_API_KEY for OpenAI embeddings)")

        memory_settings = MemorySettings(
            neo4j={
                "uri": settings.neo4j_uri,
                "username": settings.neo4j_username,
                "password": SecretStr(settings.neo4j_password),
            },
            embedding=embedding_config,
        )
        _memory_client = MemoryClient(memory_settings)
        await _memory_client.__aenter__()
        logger.info("neo4j-agent-memory MemoryClient initialized")
    except ImportError:
        logger.info("neo4j-agent-memory not installed — conversation memory disabled")
        _memory_client = None
    except Exception as e:
        logger.warning("Failed to initialize MemoryClient: %s", e)
        _memory_client = None

    # Ensure spacy model is available for entity extraction
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spacy model en_core_web_sm...")
            spacy.cli.download("en_core_web_sm")
    except ImportError:
        pass


async def close_neo4j() -> None:
    """Close Neo4j connection and memory client."""
    global _driver, _memory_client, _connected
    _connected = False
    if _memory_client:
        try:
            await _memory_client.__aexit__(None, None, None)
        except Exception:
            pass
        _memory_client = None
    if _driver:
        await _driver.close()
        _driver = None


# ---------------------------------------------------------------------------
# Conversation memory — powered by neo4j-agent-memory short-term store
# ---------------------------------------------------------------------------


async def get_conversation_history(session_id: str, limit: int = 20) -> list[dict]:
    """Retrieve conversation history for a session.

    Returns a list of {"role": str, "content": str} dicts.
    Falls back to an empty list if memory client is unavailable.
    """
    if _memory_client is None:
        return []
    try:
        conversation = await _memory_client.short_term.get_conversation(
            session_id=session_id,
            limit=limit,
        )
        return [{"role": m.role, "content": m.content} for m in conversation.messages]
    except Exception as e:
        logger.warning("Failed to retrieve conversation history: %s", e)
        return []


async def store_message(session_id: str, role: str, content: str) -> None:
    """Store a message in conversation history.

    Silently no-ops if memory client is unavailable.
    """
    if _memory_client is None:
        return
    try:
        await _memory_client.short_term.add_message(
            session_id=session_id,
            role=role,
            content=content,
        )
    except Exception as e:
        logger.warning("Failed to store message: %s", e)


def get_driver() -> AsyncDriver:
    """Get the Neo4j driver instance."""
    if _driver is None:
        raise RuntimeError("Neo4j not connected. Call connect_neo4j() first.")
    return _driver


async def execute_cypher(
    query: str,
    parameters: dict | None = None,
    *,
    collect: bool = True,
    tool_name: str | None = None,
    timeout: float | None = 30.0,
) -> list[dict]:
    """Execute a Cypher query and return results as dicts with graph metadata."""
    driver = get_driver()
    if tool_name and collect:
        _collector.emit_tool_start(tool_name, parameters or {})
    async with driver.session() as session:
        result = await session.run(query, parameters or {}, timeout=timeout)
        records = []
        async for record in result:
            records.append({k: _serialize(v) for k, v in record.items()})
        if collect:
            _collector.collect(records)
            if tool_name:
                _collector.collect_tool_call(
                    tool_name, parameters or {}, str(records[:2])
                )
        return records


async def search_entities(query: str, label: str | None = None, limit: int = 20) -> list[dict]:
    """Full-text search across entities."""
    from app.config import settings
    cypher = """
    MATCH (n)
    WHERE ($label IS NULL OR $label IN labels(n))
      AND (n.domain IS NULL OR n.domain = $domain)
      AND (toLower(n.name) CONTAINS toLower($query)
           OR toLower(coalesce(n.description, '')) CONTAINS toLower($query))
    OPTIONAL MATCH (n)-[r]-(related)
    WHERE related.domain IS NULL OR related.domain = $domain
    RETURN n, type(r) AS rel_type, labels(related) AS related_labels,
           related.name AS related_name
    LIMIT $limit
    """
    return await execute_cypher(cypher, {"query": query, "label": label, "limit": limit, "domain": settings.domain_id})


async def get_entity_graph(entity_name: str, depth: int = 2) -> dict:
    """Get the subgraph around an entity."""
    from app.config import settings
    cypher = """
    MATCH (n)
    WHERE toLower(n.name) = toLower($name)
      AND (n.domain IS NULL OR n.domain = $domain)
    OPTIONAL MATCH path = (n)-[*1..2]-(connected)
    WHERE connected.domain IS NULL OR connected.domain = $domain
    RETURN n, collect(DISTINCT connected) AS connected_nodes
    """
    results = await execute_cypher(cypher, {"name": entity_name, "domain": settings.domain_id})
    if results:
        return results[0]
    return {"nodes": [], "relationships": []}


async def get_schema() -> dict:
    """Get the graph database schema, scoped to the current domain."""
    from app.config import settings
    # Get labels that actually exist in domain data
    labels_result = await execute_cypher(
        """
        MATCH (n) WHERE n.domain IS NULL OR n.domain = $domain
        WITH DISTINCT labels(n) AS ls UNWIND ls AS l
        RETURN DISTINCT l AS label ORDER BY label
        """,
        {"domain": settings.domain_id},
        collect=False,
    )
    rel_types_result = await execute_cypher(
        """
        MATCH (a)-[r]->(b)
        WHERE (a.domain IS NULL OR a.domain = $domain)
          AND (b.domain IS NULL OR b.domain = $domain)
        RETURN DISTINCT type(r) AS relationshipType ORDER BY relationshipType
        """,
        {"domain": settings.domain_id},
        collect=False,
    )
    return {
        "labels": [r["label"] for r in labels_result],
        "relationship_types": [r["relationshipType"] for r in rel_types_result],
    }


async def get_schema_visualization() -> dict:
    """Get the graph schema as nodes and relationships for visualization.

    Uses db.schema.visualization() filtered to only labels that exist
    in the current domain's data.
    """
    from app.config import settings
    try:
        results = await execute_cypher(
            "CALL db.schema.visualization() YIELD nodes, relationships RETURN nodes, relationships",
            collect=False,
        )
        if results:
            raw = results[0]
            # Get labels that exist in domain data to filter schema
            domain_labels_result = await execute_cypher(
                """
                MATCH (n) WHERE n.domain IS NULL OR n.domain = $domain
                WITH DISTINCT labels(n) AS ls UNWIND ls AS l
                RETURN collect(DISTINCT l) AS domain_labels
                """,
                {"domain": settings.domain_id},
                collect=False,
            )
            domain_labels = set(domain_labels_result[0]["domain_labels"]) if domain_labels_result else set()
            # Also include infrastructure labels
            domain_labels |= {"Document", "DecisionTrace", "TraceStep"}

            # Filter schema nodes to only domain-relevant labels
            filtered_nodes = []
            for node in raw.get("nodes", []):
                node_labels = set()
                if isinstance(node, dict):
                    node_labels = set(node.get("labels", []))
                elif hasattr(node, "labels"):
                    node_labels = set(node.labels)
                if node_labels & domain_labels:
                    filtered_nodes.append(node)

            # Filter relationships to only those between domain labels
            filtered_rels = []
            for rel in raw.get("relationships", []):
                start_id = None
                end_id = None
                if isinstance(rel, dict):
                    start_id = rel.get("startNodeElementId")
                    end_id = rel.get("endNodeElementId")
                elif hasattr(rel, "start_node"):
                    start_id = rel.start_node.element_id
                    end_id = rel.end_node.element_id
                # Keep relationship if both endpoints are in filtered nodes
                node_ids = set()
                for n in filtered_nodes:
                    if isinstance(n, dict):
                        node_ids.add(n.get("elementId"))
                    elif hasattr(n, "element_id"):
                        node_ids.add(n.element_id)
                if start_id in node_ids and end_id in node_ids:
                    filtered_rels.append(rel)

            return {"nodes": filtered_nodes, "relationships": filtered_rels}
    except Exception:
        pass
    # Fallback: return basic label/rel lists
    return await get_schema()


async def expand_node(element_id: str) -> dict:
    """Get immediate neighbors of a node for graph expansion."""
    from app.config import settings
    cypher = """
    MATCH (n) WHERE elementId(n) = $elementId
    OPTIONAL MATCH (n)-[r]-(m)
    WHERE m.domain IS NULL OR m.domain = $domain
    RETURN collect(DISTINCT n) + collect(DISTINCT m) AS nodes,
           collect(DISTINCT r) AS relationships
    """
    results = await execute_cypher(cypher, {"elementId": element_id, "domain": settings.domain_id}, collect=False)
    if results:
        return results[0]
    return {"nodes": [], "relationships": []}


async def reset_database() -> None:
    """Drop all data and recreate schema."""
    await execute_cypher("MATCH (n) DETACH DELETE n", collect=False)
