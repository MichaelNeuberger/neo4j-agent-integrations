"""Generate sample data for Healthcare context graph."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.config import settings
from app.context_graph_client import connect_neo4j, close_neo4j, execute_cypher


DATA_DIR = Path(__file__).parent.parent.parent / "data"


async def apply_schema():
    """Apply the Cypher schema constraints and indexes."""
    schema_path = Path(__file__).parent.parent.parent / "cypher" / "schema.cypher"
    if schema_path.exists():
        schema = schema_path.read_text()
        for statement in schema.split(";"):
            stmt = statement.strip()
            if stmt and not stmt.startswith("//"):
                try:
                    await execute_cypher(stmt)
                    print(f"  Applied: {stmt[:60]}...")
                except Exception as e:
                    print(f"  Warning: {e}")


async def load_fixture_data(data: dict):
    """Load entities and relationships from fixture data."""
    # Create entities
    entities = data.get("entities", {})
    for label, items in entities.items():
        for item in items:
            enriched = {**item, "domain": settings.domain_id}
            set_clause = ", ".join(f"n.{k} = ${k}" for k in enriched.keys())
            cypher = f"MERGE (n:{label} {{name: $name, domain: $domain}}) ON CREATE SET {set_clause} ON MATCH SET {set_clause}"
            try:
                await execute_cypher(cypher, enriched)
            except Exception as e:
                print(f"  Warning creating {label}: {e}")
        print(f"  Created {len(items)} {label} nodes")

    # Create relationships
    relationships = data.get("relationships", [])
    for rel in relationships:
        cypher = f"""
        MATCH (a:{rel['source_label']} {{name: $source_name}})
        MATCH (b:{rel['target_label']} {{name: $target_name}})
        MERGE (a)-[r:{rel['type']}]->(b)
        """
        try:
            await execute_cypher(cypher, {
                "source_name": rel["source_name"],
                "target_name": rel["target_name"],
            })
        except Exception as e:
            print(f"  Warning creating relationship: {e}")
    print(f"  Created {len(relationships)} relationships")


async def load_documents(data: dict):
    """Load documents and link them to mentioned entities."""
    documents = data.get("documents", [])
    if not documents:
        print("  No documents to load")
        return

    for doc in documents:
        cypher = """
        MERGE (d:Document {title: $title})
        SET d.content = $content,
            d.template_id = $template_id,
            d.template_name = $template_name,
            d.domain = $domain
        """
        try:
            await execute_cypher(cypher, {
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "template_id": doc.get("template_id", ""),
                "template_name": doc.get("template_name", ""),
                "domain": settings.domain_id,
            })
        except Exception as e:
            print(f"  Warning creating document: {e}")

    print(f"  Created {len(documents)} Document nodes")

    # Link documents to mentioned entities
    try:
        link_cypher = """
        MATCH (d:Document) WHERE d.domain = $domain
        MATCH (e) WHERE e.name IS NOT NULL
          AND NOT 'Document' IN labels(e)
          AND NOT 'DecisionTrace' IN labels(e)
          AND NOT 'TraceStep' IN labels(e)
          AND (e.domain IS NULL OR e.domain = $domain)
          AND d.content CONTAINS e.name
        MERGE (d)-[:MENTIONS]->(e)
        RETURN count(*) AS links_created
        """
        result = await execute_cypher(link_cypher, {"domain": settings.domain_id})
        link_count = result[0]["links_created"] if result else 0
        print(f"  Created {link_count} MENTIONS relationships")
    except Exception as e:
        print(f"  Warning linking documents: {e}")


async def load_decision_traces(data: dict):
    """Load decision traces with their reasoning steps."""
    traces = data.get("traces", [])
    if not traces:
        print("  No decision traces to load")
        return

    for trace in traces:
        # Create trace node
        try:
            await execute_cypher(
                "MERGE (t:DecisionTrace {id: $id}) SET t.task = $task, t.outcome = $outcome, t.domain = $domain",
                {
                    "id": trace.get("id", ""),
                    "task": trace.get("task", ""),
                    "outcome": trace.get("outcome", ""),
                    "domain": settings.domain_id,
                },
            )
        except Exception as e:
            print(f"  Warning creating trace: {e}")
            continue

        # Create steps and link to trace
        for i, step in enumerate(trace.get("steps", [])):
            try:
                await execute_cypher(
                    """
                    MATCH (t:DecisionTrace {id: $trace_id})
                    MERGE (s:TraceStep {trace_id: $trace_id, step_number: $step_number})
                    SET s.thought = $thought, s.action = $action, s.observation = $observation
                    MERGE (t)-[:HAS_STEP]->(s)
                    """,
                    {
                        "trace_id": trace.get("id", ""),
                        "step_number": i + 1,
                        "thought": step.get("thought", ""),
                        "action": step.get("action", ""),
                        "observation": step.get("observation", ""),
                    },
                )
            except Exception as e:
                print(f"  Warning creating trace step: {e}")

    print(f"  Created {len(traces)} DecisionTrace nodes with steps")

async def main():
    print("Seeding Healthcare context graph...")
    await connect_neo4j()


    # Load fixture data
    fixture_path = DATA_DIR / "fixtures.json"
    if not fixture_path.exists():
        print("No fixture data found. Create data/fixtures.json to seed data.")
        await close_neo4j()
        return

    data = json.loads(fixture_path.read_text())

    print("\n[1/4] Applying schema...")
    await apply_schema()

    print("\n[2/4] Loading entities and relationships...")

    await load_fixture_data(data)

    print("\n[3/4] Loading documents...")

    await load_documents(data)

    print("\n[4/4] Loading decision traces...")

    await load_decision_traces(data)

    await close_neo4j()
    print("\nDone! Your Healthcare context graph is ready.")


if __name__ == "__main__":
    asyncio.run(main())
