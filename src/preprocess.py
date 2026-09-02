"""
preprocess.py
=============
Load the Elliptic Bitcoin dataset from raw CSVs and build a
PyTorch Geometric (PyG) Data object ready for GNN training.

Elliptic dataset files expected in data/raw/:
    elliptic_txs_features.csv   — 203,769 rows x 167 cols (txId + 166 features)
    elliptic_txs_edgelist.csv   — 234,355 rows x 2 cols   (txId1, txId2)
    elliptic_txs_classes.csv    — 203,769 rows x 2 cols   (txId, class)

Class mapping:
    "1"       -> 1  (illicit)
    "2"       -> 0  (licit)
    "unknown" -> -1 (excluded from supervised training)

Usage:
    python src/preprocess.py                    # uses config defaults
    python src/preprocess.py --config configs/config.yaml
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(raw_dir: str):
    """Load raw Elliptic CSVs into dataframes."""
    print("[preprocess] Loading raw CSVs...")

    features_path = os.path.join(raw_dir, "elliptic_txs_features.csv")
    edges_path    = os.path.join(raw_dir, "elliptic_txs_edgelist.csv")
    classes_path  = os.path.join(raw_dir, "elliptic_txs_classes.csv")

    for p in [features_path, edges_path, classes_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing file: {p}\n"
                "Download the Elliptic dataset from Kaggle and place CSVs in data/raw/"
            )

    # Features: first column is txId, no header in original file
    feat_cols = ["txId"] + [f"f{i}" for i in range(1, 167)]
    df_feats  = pd.read_csv(features_path, header=None, names=feat_cols)

    # Edges: no header
    df_edges  = pd.read_csv(edges_path, header=None, names=["txId1", "txId2"])

    # Classes: has header "txId,class"
    df_cls    = pd.read_csv(classes_path)

    print(f"[preprocess]   Nodes: {len(df_feats):,}")
    print(f"[preprocess]   Edges: {len(df_edges):,}")
    return df_feats, df_edges, df_cls


def build_pyg_graph(df_feats, df_edges, df_cls) -> Data:
    """
    Merge features + labels and build a PyG Data object.

    Returns:
        data (Data): PyG graph with attributes:
            x           — node feature matrix  [N, 166]
            edge_index  — edge index            [2, E]
            y           — labels (0/1/-1)       [N]
            train_mask  — boolean mask for training nodes
            val_mask    — boolean mask for validation nodes
            test_mask   — boolean mask for test nodes
    """
    print("[preprocess] Building PyG graph...")

    # --- Merge class labels ---
    df = df_feats.merge(df_cls, on="txId", how="left")
    df["class"] = df["class"].fillna("unknown")

    # --- Encode labels ---
    label_map = {"1": 1, "2": 0, "unknown": -1}
    df["y"] = df["class"].map(label_map)

    # --- Build node id -> index mapping ---
    node_ids = df["txId"].values
    id_to_idx = {txid: i for i, txid in enumerate(node_ids)}

    # --- Node features (166 dims), exclude txId and class cols ---
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values.astype(np.float32)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # --- Edge index ---
    valid_edges = df_edges[
        df_edges["txId1"].isin(id_to_idx) & df_edges["txId2"].isin(id_to_idx)
    ]
    src = valid_edges["txId1"].map(id_to_idx).values
    dst = valid_edges["txId2"].map(id_to_idx).values
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # --- Labels ---
    y = torch.tensor(df["y"].values, dtype=torch.long)

    # --- Masks: supervised nodes only (licit=0 or illicit=1) ---
    labeled_mask = (y >= 0)
    labeled_idx  = labeled_mask.nonzero(as_tuple=True)[0]
    n_labeled    = labeled_idx.shape[0]

    # 70 / 10 / 20 split (temporal split mirrors actual Elliptic setup)
    n_train = int(0.70 * n_labeled)
    n_val   = int(0.10 * n_labeled)

    train_nodes = labeled_idx[:n_train]
    val_nodes   = labeled_idx[n_train:n_train + n_val]
    test_nodes  = labeled_idx[n_train + n_val:]

    train_mask = torch.zeros(len(df), dtype=torch.bool)
    val_mask   = torch.zeros(len(df), dtype=torch.bool)
    test_mask  = torch.zeros(len(df), dtype=torch.bool)

    train_mask[train_nodes] = True
    val_mask[val_nodes]     = True
    test_mask[test_nodes]   = True

    data = Data(
        x          = torch.tensor(X, dtype=torch.float32),
        edge_index = edge_index,
        y          = y,
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
    )

    print(f"[preprocess]   Graph built: {data.num_nodes:,} nodes, "
          f"{data.num_edges:,} edges, {data.num_features} features")
    print(f"[preprocess]   Train / Val / Test: "
          f"{train_mask.sum()} / {val_mask.sum()} / {test_mask.sum()}")
    return data, scaler


def save_processed(data: Data, scaler, out_dir: str):
    """Save processed PyG graph and scaler to disk."""
    os.makedirs(out_dir, exist_ok=True)
    graph_path  = os.path.join(out_dir, "elliptic_graph.pt")
    scaler_path = os.path.join(out_dir, "scaler.pkl")

    torch.save(data, graph_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[preprocess] Saved graph  -> {graph_path}")
    print(f"[preprocess] Saved scaler -> {scaler_path}")


def load_processed(processed_dir: str):
    """Load previously saved PyG graph and scaler."""
    graph_path  = os.path.join(processed_dir, "elliptic_graph.pt")
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    data   = torch.load(graph_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return data, scaler


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess Elliptic dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir  = cfg["paths"]["raw_data"]
    proc_dir = cfg["paths"]["processed_data"]

    df_feats, df_edges, df_cls = load_raw_data(raw_dir)
    data, scaler = build_pyg_graph(df_feats, df_edges, df_cls)
    save_processed(data, scaler, proc_dir)
    print("[preprocess] Done.")


if __name__ == "__main__":
    main()
