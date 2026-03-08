"""Graph embedding module for credit risk prediction.

Provides lightweight graph embedding approaches for the Prosper P2P lending
graph that are feasible on a laptop:

1. **Spectral embeddings** (default): Truncated SVD on the normalized
   adjacency matrix. Extremely fast — uses scipy.sparse, no GPU needed.
   Captures global structural patterns.

2. **Node2Vec** (optional, slower): Random walk-based embeddings with
   reduced walks (2 walks × 10 steps) to fit in memory.

3. **GraphSAGE** (supervised): Mini-batch training with manual neighbor
   sampling, bypassing the pyg-lib/torch-sparse requirement.

Embeddings are concatenated with tabular features and fed into XGBoost
(hybrid model).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from credit_domino.data.loaders import load_data
from credit_domino.modeling.train import TARGET

# Node feature columns used as GNN input (numeric only, no categoricals)
_NODE_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
    "crisis_exposure",
]


# ═══════════════════════════════════════════════════════════════════════════
# Adjacency helpers
# ═══════════════════════════════════════════════════════════════════════════


def _build_adjacency(edges_df: pd.DataFrame, node_id_map: dict[str, int]) -> dict[int, np.ndarray]:
    """Build adjacency list from edges DataFrame (undirected, deduplicated)."""
    adj: dict[int, set[int]] = {i: set() for i in range(len(node_id_map))}
    for src, dst in zip(edges_df["src_customer_id"], edges_df["dst_customer_id"]):
        si = node_id_map.get(src)
        di = node_id_map.get(dst)
        if si is not None and di is not None:
            adj[si].add(di)
            adj[di].add(si)
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in adj.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Spectral embeddings (truncated SVD on adjacency — fast & laptop-safe)
# ═══════════════════════════════════════════════════════════════════════════


def compute_spectral_embeddings(
    customers_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    embedding_dim: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """Compute spectral node embeddings via Laplacian Eigenmaps.

    Uses the normalized Laplacian L = I - D^{-1/2} A D^{-1/2} and extracts
    the smallest non-trivial eigenvectors (Fiedler vectors), which capture
    community structure and graph topology.  Much more discriminative than
    SVD on the normalized adjacency (which produces degenerate all-1 singular
    values for graphs with many near-components).

    Returns array of shape (num_nodes, embedding_dim).
    """
    from scipy.sparse import csr_matrix, diags, eye
    from scipy.sparse.linalg import eigsh

    node_ids = customers_df["customer_id"].values
    node_id_map = {cid: i for i, cid in enumerate(node_ids)}
    n = len(node_ids)

    # Build sparse adjacency (undirected, deduplicated)
    src_mapped = edges_df["src_customer_id"].map(node_id_map)
    dst_mapped = edges_df["dst_customer_id"].map(node_id_map)
    valid = src_mapped.notna() & dst_mapped.notna()
    src_idx = src_mapped[valid].values.astype(np.int32)
    dst_idx = dst_mapped[valid].values.astype(np.int32)

    # Undirected: both directions
    rows = np.concatenate([src_idx, dst_idx])
    cols = np.concatenate([dst_idx, src_idx])
    data = np.ones(len(rows), dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Deduplicate by clipping
    A.data = np.minimum(A.data, 1.0)

    print(f"  Adjacency: {n} nodes, {A.nnz} non-zeros")

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    degrees = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros(n)
    nonzero = degrees > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

    D_inv_sqrt = diags(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    L_norm = eye(n, format="csr") - A_norm

    # Find smallest eigenvalues (skip trivial λ=0 for connected components)
    k = min(embedding_dim + 1, n - 2)
    print(f"  Computing Laplacian Eigenmaps (k={k})...")
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n).astype(np.float64)
    eigenvalues, eigenvectors = eigsh(L_norm.astype(np.float64), k=k, which="SM", v0=v0)

    # Sort by eigenvalue (ascending — smallest first)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Skip trivial eigenvectors (eigenvalue ≈ 0, one per connected component)
    # Use eigenvectors corresponding to the smallest NON-trivial eigenvalues
    nontrivial_mask = eigenvalues > 1e-6
    eigenvectors = eigenvectors[:, nontrivial_mask][:, :embedding_dim]
    eigenvalues = eigenvalues[nontrivial_mask][:embedding_dim]

    # Scale by inverse sqrt of eigenvalue — emphasizes low-frequency structure
    scale = 1.0 / np.sqrt(np.maximum(eigenvalues, 1e-8))
    embeddings = eigenvectors * scale[np.newaxis, :]

    print(
        f"  Laplacian embeddings: shape={embeddings.shape}, "
        f"smallest eigenvalues={eigenvalues[:5].round(4)}"
    )

    return embeddings.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Node2Vec (lightweight version for laptops)
# ═══════════════════════════════════════════════════════════════════════════


def _random_walks(
    adj: dict[int, np.ndarray],
    start_nodes: np.ndarray,
    walk_length: int = 10,
    p: float = 1.0,
    q: float = 0.5,
    rng: np.random.Generator | None = None,
) -> list[list[int]]:
    """Generate biased random walks (Node2Vec style)."""
    if rng is None:
        rng = np.random.default_rng(42)

    walks = []
    for start in start_nodes:
        walk = [start]
        if len(adj.get(start, [])) == 0:
            walks.append(walk)
            continue
        cur = rng.choice(adj[start])
        walk.append(cur)
        for _ in range(walk_length - 2):
            neighbors = adj.get(cur, np.array([], dtype=np.int64))
            if len(neighbors) == 0:
                break
            prev = walk[-2]
            prev_neighbors = set(adj.get(prev, []))
            probs = np.ones(len(neighbors), dtype=np.float64)
            for i, nb in enumerate(neighbors):
                if nb == prev:
                    probs[i] = 1.0 / p
                elif nb in prev_neighbors:
                    probs[i] = 1.0
                else:
                    probs[i] = 1.0 / q
            probs /= probs.sum()
            cur = rng.choice(neighbors, p=probs)
            walk.append(cur)
        walks.append(walk)
    return walks


def train_node2vec(
    customers_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    embedding_dim: int = 32,
    walk_length: int = 10,
    walks_per_node: int = 2,
    window_size: int = 3,
    p: float = 1.0,
    q: float = 0.5,
    epochs: int = 3,
    batch_size: int = 8192,
    lr: float = 0.01,
    num_negatives: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Train Node2Vec embeddings using Skip-Gram with negative sampling.

    Uses reduced walk parameters (2 walks x 10 steps x window 3) to keep
    training pairs under ~15M for laptop feasibility.

    Returns array of shape (num_nodes, embedding_dim).
    """
    import torch
    import torch.nn.functional as fn
    from torch import nn

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    node_ids = customers_df["customer_id"].values
    node_id_map = {cid: i for i, cid in enumerate(node_ids)}
    n = len(node_ids)

    print(f"  Building adjacency list ({n} nodes)...")
    adj = _build_adjacency(edges_df, node_id_map)

    all_nodes = np.arange(n)
    print(f"  Generating random walks ({walks_per_node} walks x {walk_length} steps)...")
    all_walks = []
    for walk_iter in range(walks_per_node):
        perm = rng.permutation(all_nodes)
        walks = _random_walks(adj, perm, walk_length=walk_length, p=p, q=q, rng=rng)
        all_walks.extend(walks)
        print(f"    Walk iteration {walk_iter + 1}/{walks_per_node}")

    print(f"  Building training pairs (window={window_size})...")
    centers = []
    contexts = []
    for walk in all_walks:
        for i, center in enumerate(walk):
            left = max(0, i - window_size)
            right = min(len(walk), i + window_size + 1)
            for j in range(left, right):
                if j != i:
                    centers.append(center)
                    contexts.append(walk[j])
    centers = np.array(centers, dtype=np.int64)
    contexts = np.array(contexts, dtype=np.int64)
    print(f"  Training pairs: {len(centers):,}")

    degrees = np.array([len(adj.get(i, [])) for i in range(n)], dtype=np.float64)
    degrees = np.power(degrees + 1, 0.75)
    neg_probs = degrees / degrees.sum()

    class SkipGram(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n, embedding_dim)
            nn.init.xavier_uniform_(self.emb.weight)

        def forward(self, c, ctx):
            return (self.emb(c) * self.emb(ctx)).sum(dim=-1)

    model = SkipGram()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_pairs = len(centers)
    for epoch in range(epochs):
        perm_idx = rng.permutation(num_pairs)
        total_loss = 0.0
        num_batches = 0
        for start in range(0, num_pairs, batch_size):
            end = min(start + batch_size, num_pairs)
            idx = perm_idx[start:end]
            bs = len(idx)

            c = torch.tensor(centers[idx], dtype=torch.long)
            pos = torch.tensor(contexts[idx], dtype=torch.long)
            neg_idx = rng.choice(n, size=(bs, num_negatives), p=neg_probs)
            neg = torch.tensor(neg_idx, dtype=torch.long)

            pos_score = model(c, pos)
            pos_loss = -fn.logsigmoid(pos_score).mean()

            c_exp = c.unsqueeze(1).expand(-1, num_negatives)
            neg_score = model(c_exp.reshape(-1), neg.reshape(-1))
            neg_loss = -fn.logsigmoid(-neg_score).mean()

            loss = pos_loss + neg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

    embeddings = model.emb.weight.detach().numpy()
    return embeddings


# ═══════════════════════════════════════════════════════════════════════════
# Supervised GraphSAGE with manual neighbor sampling
# ═══════════════════════════════════════════════════════════════════════════


def build_pyg_data(
    customers_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_features: list[str] | None = None,
):
    """Convert customer + edge DataFrames to a PyG Data object."""
    import torch
    from torch_geometric.data import Data

    if node_features is None:
        node_features = _NODE_FEATURES

    node_ids = customers_df["customer_id"].values
    node_id_map = {cid: i for i, cid in enumerate(node_ids)}

    X = customers_df[node_features].values.astype(np.float32)
    means = X.mean(axis=0)
    stds = X.std(axis=0) + 1e-8
    X = (X - means) / stds

    src_ids = edges_df["src_customer_id"].map(node_id_map)
    dst_ids = edges_df["dst_customer_id"].map(node_id_map)
    valid = src_ids.notna() & dst_ids.notna()
    src_idx = src_ids[valid].values.astype(np.int64)
    dst_idx = dst_ids[valid].values.astype(np.int64)
    all_src = np.concatenate([src_idx, dst_idx])
    all_dst = np.concatenate([dst_idx, src_idx])
    edge_index = torch.tensor(np.stack([all_src, all_dst]), dtype=torch.long)

    y = torch.tensor(customers_df[TARGET].values, dtype=torch.float32)
    data = Data(x=torch.tensor(X, dtype=torch.float32), edge_index=edge_index, y=y)
    data.means = means
    data.stds = stds
    return data, node_id_map


def train_graphsage(
    data_dir: Path = Path("data"),
    seed: int = 42,
    n_customers: int | None = None,
    hidden_channels: int = 64,
    out_channels: int = 32,
    lr: float = 0.01,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 2048,
    num_neighbors: list[int] | None = None,
) -> dict:
    """Train GraphSAGE with manual mini-batch neighbor sampling."""
    import torch
    import torch.nn.functional as fn
    from torch_geometric.nn import SAGEConv

    if num_neighbors is None:
        num_neighbors = [15, 10]

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    customers_df, edges_df = load_data(data_dir, seed=seed, n_customers=n_customers)
    data, node_id_map = build_pyg_data(customers_df, edges_df)
    adj = _build_adjacency(edges_df, node_id_map)

    n = data.num_nodes
    y_np = data.y.numpy()
    indices = np.arange(n)

    train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=seed, stratify=y_np)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=y_np[temp_idx]
    )

    pos_count = y_np[train_idx].sum()
    neg_count = len(train_idx) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32)

    class GraphSAGEModel(torch.nn.Module):
        def __init__(self, in_ch, hid_ch, out_ch):
            super().__init__()
            self.conv1 = SAGEConv(in_ch, hid_ch)
            self.conv2 = SAGEConv(hid_ch, out_ch)
            self.classifier = torch.nn.Linear(out_ch, 1)

        def forward(self, x, edge_index):
            h = self.conv1(x, edge_index)
            h = fn.relu(h)
            h = fn.dropout(h, p=0.3, training=self.training)
            h = self.conv2(h, edge_index)
            h = fn.relu(h)
            return h

        def logits(self, x, edge_index):
            return self.classifier(self.forward(x, edge_index)).squeeze(-1)

        def predict(self, x, edge_index):
            return torch.sigmoid(self.logits(x, edge_index))

    def _sample_subgraph(seed_nodes, adj, num_neighbors, rng):
        current_nodes = set(seed_nodes.tolist())
        frontier = set(seed_nodes.tolist())
        sampled_edges = []
        for num_nb in reversed(num_neighbors):
            new_frontier = set()
            for node in frontier:
                nbs = adj.get(node, np.array([], dtype=np.int64))
                if len(nbs) == 0:
                    continue
                if len(nbs) > num_nb:
                    sampled = rng.choice(nbs, size=num_nb, replace=False)
                else:
                    sampled = nbs
                for nb in sampled:
                    sampled_edges.append((int(nb), node))
                    new_frontier.add(int(nb))
            current_nodes |= new_frontier
            frontier = new_frontier
        all_nodes = np.array(sorted(current_nodes), dtype=np.int64)
        remap = {g: idx for idx, g in enumerate(all_nodes)}
        if sampled_edges:
            src = [remap[e[0]] for e in sampled_edges]
            dst = [remap[e[1]] for e in sampled_edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        return all_nodes, edge_index, remap

    model = GraphSAGEModel(data.x.shape[1], hidden_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(perm), batch_size):
            batch_seeds = perm[start : start + batch_size]
            all_nodes, local_edges, remap = _sample_subgraph(batch_seeds, adj, num_neighbors, rng)

            x_sub = data.x[all_nodes]
            y_sub = data.y[all_nodes]
            seed_local = torch.tensor([remap[int(s)] for s in batch_seeds], dtype=torch.long)

            optimizer.zero_grad()
            logit = model.logits(x_sub, local_edges)
            loss = fn.binary_cross_entropy_with_logits(
                logit[seed_local], y_sub[seed_local], pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_probas = []
        with torch.no_grad():
            for vstart in range(0, len(val_idx), batch_size * 2):
                vbatch = val_idx[vstart : vstart + batch_size * 2]
                v_nodes, v_edges, v_remap = _sample_subgraph(vbatch, adj, num_neighbors, rng)
                x_sub = data.x[v_nodes]
                v_local = torch.tensor([v_remap[int(v)] for v in vbatch], dtype=torch.long)
                proba = model.predict(x_sub, v_edges)[v_local].numpy()
                val_probas.append(proba)

        val_proba = np.concatenate(val_probas)
        val_auc = roc_auc_score(y_np[val_idx], val_proba)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1:3d}  loss={avg_loss:.4f}  val_auc={val_auc:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_embeddings = np.zeros((n, out_channels), dtype=np.float32)
    test_probas = []
    with torch.no_grad():
        for tstart in range(0, n, batch_size * 2):
            tend = min(tstart + batch_size * 2, n)
            tbatch = indices[tstart:tend]
            t_nodes, t_edges, t_remap = _sample_subgraph(tbatch, adj, num_neighbors, rng)
            x_sub = data.x[t_nodes]
            t_local = torch.tensor([t_remap[int(t)] for t in tbatch], dtype=torch.long)
            emb = model.forward(x_sub, t_edges)[t_local].numpy()
            all_embeddings[tbatch] = emb
            test_mask_batch = np.isin(tbatch, test_idx)
            if test_mask_batch.any():
                proba = model.predict(x_sub, t_edges)[t_local].numpy()
                test_probas.append(proba[test_mask_batch])

    test_proba = np.concatenate(test_probas)
    test_auc = roc_auc_score(y_np[test_idx], test_proba)
    test_pred = (test_proba >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = {
        "accuracy": accuracy_score(y_np[test_idx], test_pred),
        "precision": precision_score(y_np[test_idx], test_pred, zero_division=0),
        "recall": recall_score(y_np[test_idx], test_pred, zero_division=0),
        "f1": f1_score(y_np[test_idx], test_pred, zero_division=0),
        "roc_auc": test_auc,
    }
    print("\n  GraphSAGE test metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    return {
        "model": model,
        "data": data,
        "node_id_map": node_id_map,
        "metrics": metrics,
        "embeddings": all_embeddings,
        "customers_df": customers_df,
        "edges_df": edges_df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid pipeline: Spectral/Node2Vec/GraphSAGE + XGBoost
# ═══════════════════════════════════════════════════════════════════════════


def train_hybrid(
    data_dir: Path = Path("data"),
    seed: int = 42,
    n_customers: int | None = None,
    method: str = "spectral",
    gnn_epochs: int = 100,
    embedding_dim: int = 32,
) -> dict:
    """Train hybrid Graph-Embedding + XGBoost model.

    Parameters
    ----------
    method : str
        One of "spectral" (default, fast), "node2vec", or "graphsage".
    """
    import xgboost as xgb

    from credit_domino.modeling.evaluate import evaluate_model
    from credit_domino.modeling.train import assemble_features

    customers_df, edges_df = load_data(data_dir, seed=seed, n_customers=n_customers)
    gnn_metrics = None
    gnn_model = None

    if method == "spectral":
        print("=== Phase 1: Computing Spectral Embeddings ===")
        embeddings = compute_spectral_embeddings(
            customers_df,
            edges_df,
            embedding_dim=embedding_dim,
            seed=seed,
        )
    elif method == "node2vec":
        print("=== Phase 1: Training Node2Vec Embeddings ===")
        embeddings = train_node2vec(
            customers_df,
            edges_df,
            embedding_dim=embedding_dim,
            seed=seed,
        )
    elif method == "graphsage":
        print("=== Phase 1: Training GraphSAGE ===")
        result = train_graphsage(
            data_dir=data_dir,
            seed=seed,
            n_customers=n_customers,
            out_channels=embedding_dim,
            epochs=gnn_epochs,
        )
        embeddings = result["embeddings"]
        gnn_metrics = result["metrics"]
        gnn_model = result["model"]
    else:
        raise ValueError(f"Unknown method: {method!r}")

    print("\n=== Phase 2: Training Hybrid XGBoost ===")
    X, y, encoders = assemble_features(data_dir, seed=seed, n_customers=n_customers)

    emb_cols = [f"graph_emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    X_hybrid = pd.concat([X.reset_index(drop=True), emb_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_hybrid, y, test_size=0.2, random_state=seed, stratify=y
    )

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    spw = neg_count / pos_count if pos_count > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=seed,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    metrics_hybrid, threshold = evaluate_model(model, X_test, y_test)
    print(f"\n  Hybrid XGBoost + {method} metrics:")
    for k, v in metrics_hybrid.items():
        print(f"    {k}: {v:.4f}")

    # Vanilla comparison
    X_vanilla = X_hybrid[list(X.columns)]
    X_tr_v, X_te_v, y_tr_v, y_te_v = train_test_split(
        X_vanilla, y, test_size=0.2, random_state=seed, stratify=y
    )
    model_v = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=seed,
    )
    model_v.fit(X_tr_v, y_tr_v, eval_set=[(X_te_v, y_te_v)], verbose=False)
    metrics_vanilla, _ = evaluate_model(model_v, X_te_v, y_te_v)

    print("\n  Vanilla XGBoost metrics (for comparison):")
    for k, v in metrics_vanilla.items():
        print(f"    {k}: {v:.4f}")

    auc_boost = metrics_hybrid["roc_auc"] - metrics_vanilla["roc_auc"]
    print(f"\n  AUC improvement from graph embeddings: {auc_boost:+.4f}")

    return {
        "gnn_model": gnn_model,
        "gnn_metrics": gnn_metrics,
        "hybrid_model": model,
        "hybrid_metrics": metrics_hybrid,
        "vanilla_metrics": metrics_vanilla,
        "encoders": encoders,
        "feature_names": list(X_hybrid.columns),
        "embeddings": embeddings,
    }


if __name__ == "__main__":
    result = train_hybrid(method="spectral")
