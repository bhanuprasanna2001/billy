"""Tests for domino contagion simulation."""

import networkx as nx
import pytest

from credit_domino.simulation.domino import cascade_summary, simulate_domino


@pytest.fixture
def simple_chain():
    """Linear chain: A - B - C - D - E"""
    G = nx.path_graph(5)
    mapping = {i: f"CUST_{i}" for i in range(5)}
    return nx.relabel_nodes(G, mapping)


@pytest.fixture
def star_graph():
    """Hub-and-spoke: CUST_0 connected to CUST_1..4"""
    G = nx.star_graph(4)
    mapping = {i: f"CUST_{i}" for i in range(5)}
    return nx.relabel_nodes(G, mapping)


def test_cascade_trigger_only(simple_chain):
    """With high threshold, only trigger falls."""
    result = simulate_domino(simple_chain, "CUST_0", initial_shock=0.5, decay=0.6, threshold=0.4)
    fallen = [c for c in result if c["fallen"]]
    assert len(fallen) == 1
    assert fallen[0]["customer_id"] == "CUST_0"


def test_cascade_propagates_chain(simple_chain):
    """With low threshold, cascade propagates through chain."""
    result = simulate_domino(
        simple_chain,
        "CUST_0",
        initial_shock=1.0,
        decay=0.8,
        threshold=0.1,
        max_hops=10,
    )
    cust_ids = [c["customer_id"] for c in result]
    assert "CUST_0" in cust_ids
    assert "CUST_1" in cust_ids
    # With decay=0.8, stress at hop 2 = 0.64, hop 3 = 0.512, hop 4 = 0.4096
    assert len(result) == 5  # All 5 nodes reached


def test_cascade_decay(simple_chain):
    """Stress decreases geometrically with each hop."""
    result = simulate_domino(
        simple_chain,
        "CUST_0",
        initial_shock=1.0,
        decay=0.6,
        threshold=0.1,
        max_hops=10,
    )
    stresses = {c["customer_id"]: c["stress"] for c in result}
    assert stresses["CUST_0"] == 1.0
    assert stresses["CUST_1"] == pytest.approx(0.6, abs=0.01)
    assert stresses["CUST_2"] == pytest.approx(0.36, abs=0.01)


def test_cascade_stops_at_threshold(simple_chain):
    """Cascade stops when stress drops below threshold."""
    result = simulate_domino(
        simple_chain,
        "CUST_0",
        initial_shock=1.0,
        decay=0.5,
        threshold=0.3,
        max_hops=10,
    )
    # Stress: 1.0 → 0.5 → 0.25 (below 0.3, stops propagating)
    fallen = [c for c in result if c["fallen"]]
    assert len(fallen) == 2  # CUST_0 (1.0) and CUST_1 (0.5)


def test_cascade_max_hops(simple_chain):
    result = simulate_domino(
        simple_chain,
        "CUST_0",
        initial_shock=1.0,
        decay=0.9,
        threshold=0.01,
        max_hops=2,
    )
    max_hop = max(c["hop"] for c in result)
    assert max_hop <= 2


def test_star_graph_spread(star_graph):
    """Hub trigger reaches all spokes in 1 hop."""
    result = simulate_domino(
        star_graph,
        "CUST_0",
        initial_shock=1.0,
        decay=0.7,
        threshold=0.1,
    )
    assert len(result) == 5
    hop1 = [c for c in result if c["hop"] == 1]
    assert len(hop1) == 4  # All spokes reached


def test_cascade_nonexistent_node(simple_chain):
    result = simulate_domino(simple_chain, "CUST_999")
    assert result == []


def test_cascade_summary_structure(simple_chain):
    result = simulate_domino(
        simple_chain,
        "CUST_0",
        initial_shock=1.0,
        decay=0.7,
        threshold=0.1,
        max_hops=10,
    )
    summary = cascade_summary(result)
    assert "total_affected" in summary
    assert "total_fallen" in summary
    assert "max_hop" in summary
    assert "avg_stress" in summary
    assert summary["total_affected"] > 0
