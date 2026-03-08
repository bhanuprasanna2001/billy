"""Domino contagion simulator: BFS cascade with edge-weighted stress propagation."""

from collections import deque
from pathlib import Path

import networkx as nx

from credit_domino.data.loaders import load_data

# How much stress each relationship type transmits (relative to base decay).
# Co-borrowers share loan liability → full propagation.
# Guarantors back the loan → high propagation.
# Employer links are weaker → low propagation.
_EDGE_STRESS: dict[str, float] = {
    "co-borrower": 1.0,
    "guarantor": 0.7,
    "employer": 0.4,
    "loan": 1.0,  # Prosper P2P direct lending — full stress propagation
}


def _node_vulnerability(G: nx.Graph, node: str) -> float:
    """Financial vulnerability of a node (range 0.5–1.5).

    High debt-to-income → more vulnerable (amplifies incoming stress).
    No financial data → neutral (1.0), preserving backward compat with plain graphs.
    """
    data = G.nodes[node]
    income = data.get("person_income")
    loan = data.get("loan_amnt")
    if income and loan and income > 0:
        dti = min(loan / income, 1.0)
        return 0.5 + dti  # 0.5 (low debt) … 1.5 (maxed out)
    return 1.0


def build_graph(data_dir: Path = Path("data"), seed: int = 42, n_customers: int | None = None):
    """Build NetworkX graph with risk scores as node attributes.

    Returns (G, df) where G has node attributes: risk_score, customer_id, etc.
    """
    df, edges = load_data(data_dir, seed=seed, n_customers=n_customers)

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(
            row["customer_id"],
            **{
                "person_income": row["person_income"],
                "loan_amnt": row["loan_amnt"],
                "loan_status": int(row["loan_status"]),
                "is_recent_default": bool(row["is_recent_default"]),
            },
        )

    for src, dst, etype in zip(
        edges["src_customer_id"], edges["dst_customer_id"], edges["edge_type"]
    ):
        G.add_edge(src, dst, edge_type=etype)

    return G, df


def simulate_domino(
    G: nx.Graph,
    trigger_node: str,
    initial_shock: float = 1.0,
    decay: float = 0.6,
    threshold: float = 0.3,
    max_hops: int = 5,
) -> list[dict]:
    """BFS contagion cascade from a trigger node.

    Args:
        G: NetworkX graph with customer nodes
        trigger_node: customer_id to start the cascade
        initial_shock: stress level applied to the trigger (0–1)
        decay: base multiplier per hop (further modulated by edge type + node vulnerability)
        threshold: minimum stress to "fall" (trigger further cascade)
        max_hops: max BFS depth

    Stress formula per hop:
        next_stress = current_stress × decay × edge_weight × node_vulnerability
    where edge_weight depends on relationship type (co-borrower=1.0, guarantor=0.7,
    employer=0.4) and node_vulnerability on debt-to-income ratio (0.5–1.5).
    Plain graphs without these attributes get uniform decay (backward compatible).

    Returns:
        List of dicts: [{customer_id, hop, stress, fallen}, ...]
        sorted by hop then stress descending.
    """
    if trigger_node not in G:
        return []

    cascade: list[dict] = []
    visited: set[str] = set()
    queue: deque[tuple[str, int, float, str | None]] = deque()

    queue.append((trigger_node, 0, initial_shock, None))
    visited.add(trigger_node)

    while queue:
        node, hop, stress, parent = queue.popleft()
        fallen = stress >= threshold

        # Get edge type from parent (if any)
        edge_type = None
        if parent is not None and G.has_edge(parent, node):
            edge_type = G.edges[parent, node].get("edge_type", "unknown")

        cascade.append(
            {
                "customer_id": node,
                "hop": hop,
                "stress": round(stress, 4),
                "fallen": fallen,
                "parent": parent,
                "edge_type": edge_type,
            }
        )

        if fallen and hop < max_hops:
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    edge_data = G.edges[node, neighbor]
                    edge_mult = _EDGE_STRESS.get(edge_data.get("edge_type", ""), 1.0)
                    vuln = _node_vulnerability(G, neighbor)
                    next_stress = stress * decay * edge_mult * vuln
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1, next_stress, node))

    cascade.sort(key=lambda x: (x["hop"], -x["stress"]))
    return cascade


def cascade_summary(cascade: list[dict]) -> dict:
    """Summarize a cascade result."""
    fallen = [c for c in cascade if c["fallen"]]
    return {
        "total_affected": len(cascade),
        "total_fallen": len(fallen),
        "max_hop": max(c["hop"] for c in cascade) if cascade else 0,
        "avg_stress": sum(c["stress"] for c in cascade) / len(cascade) if cascade else 0,
        "cascade_by_hop": {
            hop: sum(1 for c in cascade if c["hop"] == hop)
            for hop in range(max(c["hop"] for c in cascade) + 1)
            if cascade
        },
    }
