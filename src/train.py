"""
train.py
========
Plaintext training loop for the GNN on the Elliptic dataset.

Pipeline:
    1. Load processed PyG graph from data/processed/
    2. Build GCN or GIN model from config
    3. Train with Adam optimizer + early stopping
    4. Save best model weights to results/model.pth
    5. Log metrics (accuracy, F1, loss) per epoch to results/metrics.csv

Usage:
    python src/train.py
    python src/train.py --config configs/config.yaml
"""

import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, accuracy_score

from src.model import build_model
from src.preprocess import load_processed


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    """
    Compute accuracy and macro-F1 for nodes selected by mask.
    Ignores unknown nodes (label == -1).
    """
    preds  = logits[mask].argmax(dim=1).cpu().numpy()
    target = labels[mask].cpu().numpy()

    acc = accuracy_score(target, preds)
    f1  = f1_score(target, preds, average="macro", zero_division=0)
    return acc, f1


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------

def train_epoch(model, data, optimizer, device):
    """Run one training step and return loss."""
    model.train()
    optimizer.zero_grad()

    out  = model(data.x.to(device), data.edge_index.to(device))
    mask = data.train_mask.to(device)
    y    = data.y.to(device)

    # Only compute loss on labeled training nodes
    loss = F.cross_entropy(out[mask], y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, device):
    """Evaluate model on nodes selected by mask."""
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    acc, f1 = compute_metrics(out.cpu(), data.y, mask)
    return acc, f1


# -------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------

def train(cfg: dict):
    # --- Setup ---
    torch.manual_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # --- Data ---
    proc_dir = cfg["paths"]["processed_data"]
    data, _  = load_processed(proc_dir)
    print(f"[train] Graph loaded: {data.num_nodes:,} nodes, {data.num_edges:,} edges")

    # --- Model ---
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model: {cfg['model']['type']}  |  Parameters: {n_params:,}")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # --- Logging setup ---
    results_dir = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.csv")
    model_path   = cfg["paths"]["model_save"]

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "train_acc", "train_f1",
                         "val_acc", "val_f1", "epoch_time_s"])

    # --- Early stopping ---
    patience     = cfg["train"]["early_stopping_patience"]
    best_val_f1  = 0.0
    patience_ctr = 0

    print(f"[train] Starting training for {cfg['train']['epochs']} epochs...")
    print("-" * 65)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0   = time.time()
        loss = train_epoch(model, data, optimizer, device)
        t1   = time.time()

        train_acc, train_f1 = evaluate(model, data, data.train_mask, device)
        val_acc,   val_f1   = evaluate(model, data, data.val_mask,   device)

        epoch_time = t1 - t0

        # Log to CSV
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{loss:.4f}",
                             f"{train_acc:.4f}", f"{train_f1:.4f}",
                             f"{val_acc:.4f}",   f"{val_f1:.4f}",
                             f"{epoch_time:.3f}"])

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
                  f"Time: {epoch_time:.2f}s")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[train] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    print("-" * 65)
    print(f"[train] Best Val F1: {best_val_f1:.4f}")
    print(f"[train] Model saved -> {model_path}")
    print(f"[train] Metrics CSV -> {metrics_path}")

    # Final test evaluation
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    except TypeError:
        model.load_state_dict(torch.load(model_path, map_location=device))
    test_acc, test_f1 = evaluate(model, data, data.test_mask, device)
    print(f"[train] Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    return model


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GNN on Elliptic dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
