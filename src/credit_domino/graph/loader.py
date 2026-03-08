"""Load customer nodes and relationship edges into Neo4j."""

import pandas as pd

from credit_domino.config import Settings


def get_neo4j_driver(settings: Settings | None = None):
    """Create a Neo4j driver from settings."""
    from neo4j import GraphDatabase

    if settings is None:
        settings = Settings()
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def load_graph_to_neo4j(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    driver,
    batch_size: int = 25000,
) -> dict:
    """Batch-load customer nodes and relationship edges into Neo4j.

    Uses MERGE for idempotent node creation and CREATE for edges.
    Returns counts of nodes and edges loaded.
    """
    node_count = 0
    edge_count = 0

    with driver.session() as session:
        # Clear existing data for clean reload
        session.run("MATCH (n:Customer) DETACH DELETE n")

        # Create index for fast lookups
        session.run("CREATE INDEX IF NOT EXISTS FOR (c:Customer) ON (c.customer_id)")

        # Batch-insert nodes
        node_records = nodes_df.to_dict("records")
        for i in range(0, len(node_records), batch_size):
            batch = node_records[i : i + batch_size]
            session.run(
                "UNWIND $batch AS row "
                "MERGE (c:Customer {customer_id: row.customer_id}) "
                "SET c.person_age = row.person_age, "
                "    c.person_income = row.person_income, "
                "    c.loan_status = row.loan_status, "
                "    c.is_recent_default = row.is_recent_default, "
                "    c.cb_person_default_on_file = row.cb_person_default_on_file, "
                "    c.loan_grade = row.loan_grade, "
                "    c.person_home_ownership = row.person_home_ownership",
                batch=batch,
            )
            node_count += len(batch)

        # Batch-insert edges
        edge_records = edges_df.to_dict("records")
        for i in range(0, len(edge_records), batch_size):
            batch = edge_records[i : i + batch_size]
            session.run(
                "UNWIND $batch AS row "
                "MATCH (a:Customer {customer_id: row.src_customer_id}) "
                "MATCH (b:Customer {customer_id: row.dst_customer_id}) "
                "CREATE (a)-[:RELATES_TO {type: row.edge_type}]->(b)",
                batch=batch,
            )
            edge_count += len(batch)

    return {"nodes_loaded": node_count, "edges_loaded": edge_count}


if __name__ == "__main__":
    from pathlib import Path

    from credit_domino.data.loaders import load_data

    print("Loading data...")
    customers, relationships = load_data(Path("data"))

    print("Connecting to Neo4j...")
    driver = get_neo4j_driver()

    print("Loading graph...")
    result = load_graph_to_neo4j(customers, relationships, driver)
    print(f"  Loaded {result['nodes_loaded']:,} nodes, {result['edges_loaded']:,} edges")

    driver.close()
    print("Done!")
