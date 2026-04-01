"""Healthcare AI Agent — PydanticAI implementation."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass

from app.config import settings


# Ensure ANTHROPIC_API_KEY env var is set before PydanticAI creates the provider.
# pydantic-settings may load an empty value from the shell env, overriding .env.
if not os.environ.get("ANTHROPIC_API_KEY"):
    if settings.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
    else:
        from dotenv import dotenv_values
        _key = dotenv_values("../.env").get("ANTHROPIC_API_KEY", "")
        if _key:
            os.environ["ANTHROPIC_API_KEY"] = _key

from pydantic_ai import Agent, RunContext

from app.context_graph_client import (
    execute_cypher, get_schema,
    get_conversation_history, store_message,
)


SYSTEM_PROMPT = """You are an AI clinical intelligence assistant with access to a comprehensive
knowledge graph of healthcare data. You help clinicians, care coordinators,
and medical staff analyze patient records, diagnoses, treatments, and
provider networks.

Your capabilities include:
- Searching patient records and clinical history
- Checking medication contraindications and interactions
- Finding similar past cases for clinical decision support
- Analyzing provider referral networks
- Tracing treatment decisions and outcomes

Always prioritize patient safety. Flag potential contraindications immediately.
Provide evidence-based insights grounded in the clinical data available.


IMPORTANT: You MUST use the available tools to query the knowledge graph before answering any question about the data. Never guess or make up information — always use tools to look up actual data from the graph. If a user asks a question, identify which tool(s) can help answer it and call them.

CRITICAL: Call tools DIRECTLY without any introductory text. Do NOT say "I'll search for..." or "Let me look up..." before calling a tool — just call the tool immediately. Only generate text AFTER you have received the tool results and are ready to provide your final answer.

When writing Cypher queries with run_cypher:
- Never combine ORDER BY with DISTINCT or aggregation in the same RETURN clause — use a WITH clause first
- Always LIMIT results (default LIMIT 25) to avoid overwhelming responses
- Use toLower() for case-insensitive matching
- If a query fails, try a simpler approach rather than repeating the same pattern"""



@dataclass
class AgentDeps:
    """Dependencies injected into the agent."""
    session_id: str


agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    system_prompt=SYSTEM_PROMPT,
    deps_type=AgentDeps,
    retries=2,
)

# ---------------------------------------------------------------------------
# Agent tools — domain-specific for Healthcare
# ---------------------------------------------------------------------------

@agent.tool
async def search_patient(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search for patients by name or ID"""
    cypher = """MATCH (p:Patient)
    WHERE toLower(p.name) CONTAINS toLower($query)
       OR p.patient_id = $query
    OPTIONAL MATCH (p)-[r]-(related)
    RETURN p, type(r) AS rel_type, related
    LIMIT 20
"""
    params = {
        "query": query,
    }
    result = await execute_cypher(cypher, params, tool_name="search_patient")
    return json.dumps(result, default=str)

@agent.tool
async def get_patient_history(ctx: RunContext[AgentDeps], patient_id: str) -> str:
    """Get comprehensive history for a patient"""
    cypher = """MATCH (p:Patient {patient_id: $patient_id})
    OPTIONAL MATCH (p)-[:HAD_ENCOUNTER]->(e:Encounter)-[:OCCURRED_AT]->(f:Facility)
    OPTIONAL MATCH (p)-[:DIAGNOSED_WITH]->(d:Diagnosis)
    OPTIONAL MATCH (e)-[:INCLUDES]->(t:Treatment)
    RETURN p, collect(DISTINCT e) AS encounters,
           collect(DISTINCT d) AS diagnoses,
           collect(DISTINCT t) AS treatments,
           collect(DISTINCT f) AS facilities
"""
    params = {
        "patient_id": patient_id,
    }
    result = await execute_cypher(cypher, params, tool_name="get_patient_history")
    return json.dumps(result, default=str)

@agent.tool
async def check_contraindications(ctx: RunContext[AgentDeps], patient_id: str) -> str:
    """Check for medication contraindications for a patient"""
    cypher = """MATCH (p:Patient {patient_id: $patient_id})-[:DIAGNOSED_WITH]->(d:Diagnosis)
    MATCH (t:Treatment)-[:TREATS]->(d)
    MATCH (t)-[:USES]->(m:Medication)
    OPTIONAL MATCH (m)-[:CONTRAINDICATED_WITH]->(contra:Medication)
    RETURN p, d, m, collect(contra) AS contraindications
"""
    params = {
        "patient_id": patient_id,
    }
    result = await execute_cypher(cypher, params, tool_name="check_contraindications")
    return json.dumps(result, default=str)

@agent.tool
async def find_similar_cases(ctx: RunContext[AgentDeps], patient_id: str) -> str:
    """Find patients with similar diagnoses and demographics"""
    cypher = """MATCH (p:Patient {patient_id: $patient_id})-[:DIAGNOSED_WITH]->(d:Diagnosis)
    MATCH (similar:Patient)-[:DIAGNOSED_WITH]->(d)
    WHERE similar.patient_id <> $patient_id
    OPTIONAL MATCH (similar)-[:HAD_ENCOUNTER]->(e:Encounter)-[:INCLUDES]->(t:Treatment)
    RETURN similar, d, collect(DISTINCT t) AS treatments
    LIMIT 10
"""
    params = {
        "patient_id": patient_id,
    }
    result = await execute_cypher(cypher, params, tool_name="find_similar_cases")
    return json.dumps(result, default=str)

@agent.tool
async def get_referral_network(ctx: RunContext[AgentDeps], provider_id: str) -> str:
    """Get the referral network for a provider"""
    cypher = """MATCH (p:Provider {provider_id: $provider_id})
    OPTIONAL MATCH (p)-[:REFERRED_TO]->(referred:Provider)
    OPTIONAL MATCH (referrer:Provider)-[:REFERRED_TO]->(p)
    OPTIONAL MATCH (p)-[:AFFILIATED_WITH]->(f:Facility)
    RETURN p, collect(DISTINCT referred) AS referred_to,
           collect(DISTINCT referrer) AS referred_by,
           collect(DISTINCT f) AS facilities
"""
    params = {
        "provider_id": provider_id,
    }
    result = await execute_cypher(cypher, params, tool_name="get_referral_network")
    return json.dumps(result, default=str)

@agent.tool
async def list_patients(ctx: RunContext[AgentDeps], limit: str) -> str:
    """List Patient records with optional limit"""
    cypher = """MATCH (n:Patient)
    RETURN n
    ORDER BY n.name
    LIMIT toInteger($limit)
"""
    params = {
        "limit": limit,
    }
    result = await execute_cypher(cypher, params, tool_name="list_patients")
    return json.dumps(result, default=str)

@agent.tool
async def get_patient_by_id(ctx: RunContext[AgentDeps], id: str) -> str:
    """Get a specific Patient by ID with all connections"""
    cypher = """MATCH (n:Patient {patient_id: $id})
    OPTIONAL MATCH (n)-[r]-(related)
    RETURN n, type(r) AS relationship, labels(related) AS related_labels, related.name AS related_name
    LIMIT 50
"""
    params = {
        "id": id,
    }
    result = await execute_cypher(cypher, params, tool_name="get_patient_by_id")
    return json.dumps(result, default=str)



@agent.tool
async def run_cypher(ctx: RunContext[AgentDeps], query: str, parameters: str = "{}") -> str:
    """Execute a read-only Cypher query against the knowledge graph."""
    try:
        params = json.loads(parameters) if parameters else {}
    except json.JSONDecodeError:
        return json.dumps([{"error": "Invalid JSON parameters"}])
    params.setdefault("domain", settings.domain_id)
    try:
        result = await execute_cypher(query, params, tool_name="run_cypher")
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps([{"error": f"Cypher query failed: {e}"}])


@agent.tool
async def get_graph_schema(ctx: RunContext[AgentDeps]) -> str:
    """Get the knowledge graph schema (node labels and relationship types)."""
    result = await get_schema()
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------


async def handle_message(message: str, session_id: str | None = None) -> dict:
    """Handle an incoming chat message."""
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Retrieve conversation history and store the new user message
    history = await get_conversation_history(session_id)
    await store_message(session_id, "user", message)

    # Convert history to PydanticAI message format
    from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
    message_history = []
    for msg in history:
        if msg["role"] == "user":
            message_history.append(
                ModelRequest(parts=[UserPromptPart(content=msg["content"])])
            )
        elif msg["role"] == "assistant":
            message_history.append(
                ModelResponse(parts=[TextPart(content=msg["content"])])
            )

    deps = AgentDeps(session_id=session_id)
    result = await agent.run(
        message, deps=deps, message_history=message_history
    )

    response_text = result.output or ""
    if not response_text.strip():
        response_text = "I searched the knowledge graph but couldn't find relevant results for your query. Could you try rephrasing your question?"
    await store_message(session_id, "assistant", response_text)

    return {
        "response": response_text,
        "session_id": session_id,
        "graph_data": None,
    }


async def handle_message_stream(message: str, session_id: str | None = None) -> dict:
    """Handle a chat message with streaming text deltas via the collector event queue."""
    from app.context_graph_client import get_collector

    if session_id is None:
        session_id = str(uuid.uuid4())

    collector = get_collector()
    history = await get_conversation_history(session_id)
    await store_message(session_id, "user", message)

    # Convert history to PydanticAI message format
    from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
    message_history = []
    for msg in history:
        if msg["role"] == "user":
            message_history.append(
                ModelRequest(parts=[UserPromptPart(content=msg["content"])])
            )
        elif msg["role"] == "assistant":
            message_history.append(
                ModelResponse(parts=[TextPart(content=msg["content"])])
            )

    deps = AgentDeps(session_id=session_id)
    # Use agent.run() (not run_stream) so the full agent loop completes —
    # including all tool calls — before we emit the final text.
    # run_stream stops at the first text part, so it cuts off before tool
    # results are incorporated when Claude generates "I'll search..." + a tool
    # call in the same response.  Tool events (tool_start / tool_end) are still
    # pushed to the SSE queue by execute_cypher during the run.
    result = await agent.run(
        message, deps=deps, message_history=message_history
    )

    response_text = result.output or ""
    if not response_text.strip():
        response_text = "I searched the knowledge graph but couldn't find relevant results for your query. Could you try rephrasing your question?"

    collector.emit_text_delta(response_text)
    await store_message(session_id, "assistant", response_text)
    collector.emit_done(response_text, session_id)

    return {
        "response": response_text,
        "session_id": session_id,
        "graph_data": None,
    }
