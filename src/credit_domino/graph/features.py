"""Graph feature engineering: NetworkX (local/CI) and Neo4j (production) backends."""

from collections import deque
from pathlib import Path

import networkx as nx
import pandas as pd

from credit_domino.data.loaders import load_data


def _multi_source_bfs(G: nx.Graph, sources: set) -> dict:
    """BFS from multiple source nodes simultaneously. Returns {node: min_distance}."""
    distances: dict = {}
    queue: deque = deque()
    for s in sources:
        if s in G:
            distances[s] = 0
            queue.append(s)
    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def compute_graph_features(
    customers_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute graph features from pre-loaded DataFrames (no I/O).

    Features computed:
      - degree: total connections (in + out)
      - in_degree: edges pointing TO this node (borrowing intensity)
      - out_degree: edges pointing FROM this node (lending activity)
      - pagerank: influence centrality on the directed graph
      - distance_to_prior_default: BFS shortest path to any customer
        with prior default history (-1 if unreachable)
      - clustering_coefficient: local clustering on undirected projection
      - neighbor_default_frac: 1-hop neighbor prior default rate
      - neighbor_default_frac_2hop: 2-hop neighbor prior default rate
    """
    # Directed graph for pagerank (deduplicates multi-edges, which is fine
    # for pagerank but NOT for degree computation).
    G = nx.DiGraph()
    G.add_nodes_from(customers_df["customer_id"].values)
    G.add_edges_from(zip(edges_df["src_customer_id"], edges_df["dst_customer_id"]))

    # Compute degree counts directly from the edges DataFrame to preserve
    # multi-edges.  The Prosper dataset has multiple loans between the same
    # pair; nx.DiGraph silently deduplicates them, causing the model's
    # in/out degree to diverge from the values used in target generation.
    in_deg_series = edges_df.groupby("dst_customer_id").size()
    out_deg_series = edges_df.groupby("src_customer_id").size()
    all_cust = customers_df["customer_id"].values
    in_degree_dict = {c: int(in_deg_series.get(c, 0)) for c in all_cust}
    out_degree_dict = {c: int(out_deg_series.get(c, 0)) for c in all_cust}
    degree_dict = {c: in_degree_dict[c] + out_degree_dict[c] for c in all_cust}

    # Log-normalize in/out degree to [0, 1] — compresses the power-law tail
    # so that the signal is spread across all nodes.  Uses log(1+d)/log(1+max)
    # which matches the normalization in the Prosper target generation logit.
    max_in = max(in_degree_dict.values()) if in_degree_dict else 1
    max_out = max(out_degree_dict.values()) if out_degree_dict else 1
    max_in = max(max_in, 1)  # guard against all-zero
    max_out = max(max_out, 1)
    import math

    log1p_max_in = math.log1p(max_in)
    log1p_max_out = math.log1p(max_out)
    norm_in_dict = {k: math.log1p(v) / log1p_max_in for k, v in in_degree_dict.items()}
    norm_out_dict = {k: math.log1p(v) / log1p_max_out for k, v in out_degree_dict.items()}

    pagerank_dict = nx.pagerank(G)

    # BFS and clustering operate on the undirected projection
    G_undir = G.to_undirected()

    # Uses cb_person_default_on_file (credit bureau flag) — NOT loan_status (target)
    default_nodes = set(
        customers_df.loc[customers_df["cb_person_default_on_file"] == 1, "customer_id"].values
    )
    dist_from_defaults = _multi_source_bfs(G_undir, default_nodes) if default_nodes else {}

    clustering_dict = nx.clustering(G_undir)

    # Neighbor default fraction: fraction of neighbor edges where the
    # neighbor has cb_person_default_on_file == 1.  Uses the edges DataFrame
    # directly (not NetworkX) to preserve multi-edges, matching the
    # adjacency structure used in target generation.
    default_set = set(
        customers_df.loc[customers_df["cb_person_default_on_file"] == 1, "customer_id"].values
    )
    # Build deduplicated adjacency for consistent contagion computation
    adj_sets: dict[str, set[str]] = {c: set() for c in customers_df["customer_id"].values}
    for src, dst in zip(edges_df["src_customer_id"], edges_df["dst_customer_id"]):
        if src in adj_sets:
            adj_sets[src].add(dst)
        if dst in adj_sets:
            adj_sets[dst].add(src)

    # 1-hop neighbor default fraction
    neighbor_default_frac_dict: dict[str, float] = {}
    for cust_id, nbs in adj_sets.items():
        if nbs:
            neighbor_default_frac_dict[cust_id] = sum(1 for nb in nbs if nb in default_set) / len(
                nbs
            )
        else:
            neighbor_default_frac_dict[cust_id] = 0.0

    # 2-hop neighbor default fraction: fraction of 2-hop neighbors
    # (neighbors of neighbors, excluding self and direct neighbors)
    # who have prior default.  Captures structural contagion that
    # GNNs learn via message passing but flat models need explicitly.
    neighbor_default_frac_2hop_dict: dict[str, float] = {}
    for cust_id, hop1_nbs in adj_sets.items():
        if not hop1_nbs:
            neighbor_default_frac_2hop_dict[cust_id] = 0.0
            continue
        hop2: set[str] = set()
        for nb in hop1_nbs:
            hop2.update(adj_sets.get(nb, set()))
        hop2.discard(cust_id)
        hop2 -= hop1_nbs
        if hop2:
            neighbor_default_frac_2hop_dict[cust_id] = sum(
                1 for nb in hop2 if nb in default_set
            ) / len(hop2)
        else:
            neighbor_default_frac_2hop_dict[cust_id] = 0.0

    rows = []
    for cust_id in customers_df["customer_id"].values:
        rows.append(
            {
                "customer_id": cust_id,
                "degree": degree_dict.get(cust_id, 0),
                "in_degree": in_degree_dict.get(cust_id, 0),
                "out_degree": out_degree_dict.get(cust_id, 0),
                "norm_in_degree": norm_in_dict.get(cust_id, 0.0),
                "norm_out_degree": norm_out_dict.get(cust_id, 0.0),
                "pagerank": pagerank_dict.get(cust_id, 0.0),
                "distance_to_prior_default": dist_from_defaults.get(cust_id, -1),
                "clustering_coefficient": clustering_dict.get(cust_id, 0.0),
                "neighbor_default_frac": neighbor_default_frac_dict.get(cust_id, 0.0),
                "neighbor_default_frac_2hop": neighbor_default_frac_2hop_dict.get(cust_id, 0.0),
            }
        )

    return pd.DataFrame(rows)


def compute_graph_features_local(
    data_dir: Path = Path("data"),
    seed: int = 42,
    n_customers: int | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: loads data then computes graph features."""
    df, edges = load_data(data_dir, seed=seed, n_customers=n_customers)
    return compute_graph_features(df, edges)


def compute_graph_features_neo4j(
    driver,
    customers_df: pd.DataFrame | None = None,
    edges_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute graph features: Neo4j GDS pagerank + NetworkX multi-source BFS.

    Neo4j GDS provides scalable pagerank on the real graph topology.
    BFS distance uses NetworkX multi-source BFS — O(N+M) total — because
    Cypher shortestPath is O(N × (N+M)) and unusable at 89K nodes / 3.4M edges.

    Parameters
    ----------
    driver : neo4j.Driver
        Active Neo4j driver with Customer nodes and RELATES_TO edges loaded.
    customers_df : DataFrame, optional
        Customer data with customer_id and cb_person_default_on_file columns.
        Required for BFS distance computation.
    edges_df : DataFrame, optional
        Edge data with src_customer_id and dst_customer_id columns.
        Required for BFS distance computation.
    """
    with driver.session() as session:
        # In-degree (edges pointing TO node — borrowing intensity)
        in_deg_result = session.run(
            "MATCH (c:Customer) "
            "OPTIONAL MATCH ()-[r]->(c) "
            "RETURN c.customer_id AS customer_id, count(r) AS in_degree"
        )
        in_degree_data = {r["customer_id"]: r["in_degree"] for r in in_deg_result}

        # Out-degree (edges FROM node — lending activity)
        out_deg_result = session.run(
            "MATCH (c:Customer) "
            "OPTIONAL MATCH (c)-[r]->() "
            "RETURN c.customer_id AS customer_id, count(r) AS out_degree"
        )
        out_degree_data = {r["customer_id"]: r["out_degree"] for r in out_deg_result}

        # Total degree
        degree_data = {
            k: in_degree_data.get(k, 0) + out_degree_data.get(k, 0)
            for k in set(in_degree_data) | set(out_degree_data)
        }

        # Log-normalized in/out degree to [0, 1]
        max_in = max(in_degree_data.values()) if in_degree_data else 1
        max_out = max(out_degree_data.values()) if out_degree_data else 1
        max_in = max(max_in, 1)
        max_out = max(max_out, 1)
        import math

        log1p_max_in = math.log1p(max_in)
        log1p_max_out = math.log1p(max_out)
        norm_in_data = {k: math.log1p(v) / log1p_max_in for k, v in in_degree_data.items()}
        norm_out_data = {k: math.log1p(v) / log1p_max_out for k, v in out_degree_data.items()}

        # Pagerank via GDS (falls back to degree-based approximation if GDS unavailable)
        try:
            # Drop stale projection if it exists (e.g. from a failed prior run)
            try:
                session.run("CALL gds.graph.drop('risk_graph', false)")
            except Exception:  # noqa: BLE001
                pass
            session.run("CALL gds.graph.project('risk_graph', 'Customer', 'RELATES_TO')")
            pr_result = session.run(
                "CALL gds.pageRank.stream('risk_graph') "
                "YIELD nodeId, score "
                "RETURN gds.util.asNode(nodeId).customer_id AS customer_id, score AS pagerank"
            )
            pagerank_data = {r["customer_id"]: r["pagerank"] for r in pr_result}
            session.run("CALL gds.graph.drop('risk_graph')")
        except (Exception,) as _:  # noqa: BLE001 — GDS plugin may not be installed
            # GDS not available — use degree as proxy
            total = sum(degree_data.values()) or 1
            pagerank_data = {k: v / total for k, v in degree_data.items()}

    # BFS distance + clustering: NetworkX multi-source BFS is O(N+M) total.
    # Cypher shortestPath((c)-[*]-(d)) runs a SEPARATE BFS for each of 89K nodes,
    # giving O(N × (N+M)) ≈ 311 billion traversals — causes OOM / timeout / infinite retry.
    dist_from_defaults: dict = {}
    clustering_dict: dict = {}
    if customers_df is not None and edges_df is not None:
        G = nx.Graph()
        G.add_nodes_from(customers_df["customer_id"].values)
        G.add_edges_from(zip(edges_df["src_customer_id"], edges_df["dst_customer_id"]))
        default_nodes = set(
            customers_df.loc[customers_df["cb_person_default_on_file"] == 1, "customer_id"].values
        )
        if default_nodes:
            dist_from_defaults = _multi_source_bfs(G, default_nodes)
        clustering_dict = nx.clustering(G)

        # Neighbor default fraction (1-hop and 2-hop)
        neighbor_default_frac_dict: dict[str, float] = {}
        neighbor_default_frac_2hop_dict: dict[str, float] = {}
        for cust_id in G.nodes():
            nbs = set(G.neighbors(cust_id))
            if nbs:
                neighbor_default_frac_dict[cust_id] = sum(
                    1 for nb in nbs if nb in default_nodes
                ) / len(nbs)
                # 2-hop neighbors
                hop2: set[str] = set()
                for nb in nbs:
                    hop2.update(G.neighbors(nb))
                hop2.discard(cust_id)
                hop2 -= nbs
                if hop2:
                    neighbor_default_frac_2hop_dict[cust_id] = sum(
                        1 for nb in hop2 if nb in default_nodes
                    ) / len(hop2)
                else:
                    neighbor_default_frac_2hop_dict[cust_id] = 0.0
            else:
                neighbor_default_frac_dict[cust_id] = 0.0
                neighbor_default_frac_2hop_dict[cust_id] = 0.0

    # Merge into DataFrame
    all_ids = set(degree_data) | set(pagerank_data)
    rows = []
    for cust_id in sorted(all_ids):
        rows.append(
            {
                "customer_id": cust_id,
                "degree": degree_data.get(cust_id, 0),
                "in_degree": in_degree_data.get(cust_id, 0),
                "out_degree": out_degree_data.get(cust_id, 0),
                "norm_in_degree": norm_in_data.get(cust_id, 0.0),
                "norm_out_degree": norm_out_data.get(cust_id, 0.0),
                "pagerank": pagerank_data.get(cust_id, 0.0),
                "distance_to_prior_default": dist_from_defaults.get(cust_id, -1),
                "clustering_coefficient": clustering_dict.get(cust_id, 0.0),
                "neighbor_default_frac": neighbor_default_frac_dict.get(cust_id, 0.0),
                "neighbor_default_frac_2hop": neighbor_default_frac_2hop_dict.get(cust_id, 0.0),
            }
        )

    return pd.DataFrame(rows)
