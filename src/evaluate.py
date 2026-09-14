"""
evaluate.py
===========
Comprehensive evaluation and comparison across all pipeline stages:

    Stage 1: Plaintext GCN (baseline)
    Stage 2: Quantized GCN in plaintext (isolate accuracy loss from quantization)
    Stage 3: Encrypted GCN inference (end-to-end HE pipeline)

Reports per stage:
    - Accuracy, F1 (macro), Precision, Recall, AUC-ROC
    - Inference latency (wall-clock, per-node and per-graph)
    - Accuracy delta vs. plaintext baseline

Output:
    results/evaluation_report.csv
    results/plots/stage_comparison.png
    results/plots/latency_comparison.png

Usage:
    python src/evaluate.py
    python src/evaluate.py --config configs/config.yaml
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import yaml

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception as _e:
    plt = None
    MATPLOTLIB_AVAILABLE = False
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)

from src.model import build_model
from src.preprocess import load_processed
from src.quantize import quantize_model, swap_activations


# -------------------------------------------------------------------
# Plaintext Inference
# -------------------------------------------------------------------

@torch.no_grad()
def plaintext_inference(model, data, device: str = "cpu"):
    """
    Run plaintext GNN inference and return predictions + latency.

    Returns:
        preds   (np.ndarray): Predicted class labels [N_test]
        probs   (np.ndarray): Class probabilities    [N_test, 2]
        latency (float)     : Total inference wall-clock time (seconds)
    """
    model.eval()
    model = model.to(device)

    t0  = time.time()
    out = model(data.x.to(device), data.edge_index.to(device))
    t1  = time.time()

    probs  = torch.softmax(out, dim=1).cpu().numpy()
    preds  = out.argmax(dim=1).cpu().numpy()
    latency = t1 - t0

    return preds, probs, latency


# -------------------------------------------------------------------
# Metric Computation
# -------------------------------------------------------------------

def compute_all_metrics(preds: np.ndarray, probs: np.ndarray,
                        labels: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute all evaluation metrics on masked nodes.

    Args:
        preds  : Predicted class indices [N]
        probs  : Predicted probabilities [N, 2]
        labels : Ground-truth labels     [N]
        mask   : Boolean mask for subset [N]  (e.g. test_mask)

    Returns:
        dict of metric name -> value
    """
    y_pred  = preds[mask]
    y_true  = labels[mask]
    y_prob  = probs[mask, 1]   # probability of class 1 (illicit)

    return {
        "accuracy"  : round(accuracy_score(y_true, y_pred), 4),
        "f1_macro"  : round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "f1_illicit": round(f1_score(y_true, y_pred, average=None,       zero_division=0)[1], 4),
        "precision" : round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall"    : round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "auc_roc"   : round(roc_auc_score(y_true, y_prob), 4),
    }


# -------------------------------------------------------------------
# Stage Evaluation
# -------------------------------------------------------------------

def evaluate_all_stages(cfg: dict):
    """
    Evaluate all three pipeline stages and save results.
    """
    device   = "cpu"    # Use CPU for consistent latency measurement
    proc_dir = cfg["paths"]["processed_data"]
    data, _  = load_processed(proc_dir)

    test_mask  = data.test_mask.numpy()
    true_labels = data.y.numpy()

    model_path  = cfg["paths"]["model_save"]
    results_dir = cfg["paths"]["results"]
    plots_dir   = cfg["paths"]["plots"]
    os.makedirs(plots_dir, exist_ok=True)

    report_path = os.path.join(results_dir, "evaluation_report.csv")

    stages   = []
    metrics  = []
    latencies = []

    # -- Stage 1: Plaintext GCN -------------------------------------
    print("\n[evaluate] -- Stage 1: Plaintext GCN ------------------")
    model_pt = build_model(cfg)
    try:
        model_pt.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    except TypeError:
        model_pt.load_state_dict(torch.load(model_path, map_location=device))

    preds_pt, probs_pt, lat_pt = plaintext_inference(model_pt, data, device)
    m1 = compute_all_metrics(preds_pt, probs_pt, true_labels, test_mask)
    print(f"  Accuracy: {m1['accuracy']}  F1-macro: {m1['f1_macro']}  "
          f"F1-illicit: {m1['f1_illicit']}  AUC: {m1['auc_roc']}")
    print(f"  Latency : {lat_pt:.4f}s  ({lat_pt*1000:.1f}ms)")

    stages.append("1. Plaintext GCN")
    metrics.append(m1)
    latencies.append(lat_pt)

    # -- Stage 2: Quantized + Poly-activation GCN ------------------
    print("\n[evaluate] -- Stage 2: Quantized GCN (plaintext) -----")
    model_q = quantize_model(model_pt, bits=cfg["quantize"]["bits"])
    model_q = swap_activations(model_q)

    preds_q, probs_q, lat_q = plaintext_inference(model_q, data, device)
    m2 = compute_all_metrics(preds_q, probs_q, true_labels, test_mask)
    delta_f1 = round(m1["f1_macro"] - m2["f1_macro"], 4)
    print(f"  Accuracy: {m2['accuracy']}  F1-macro: {m2['f1_macro']}  "
          f"F1-illicit: {m2['f1_illicit']}  AUC: {m2['auc_roc']}")
    print(f"  Latency : {lat_q:.4f}s  ({lat_q*1000:.1f}ms)")
    print(f"  Accuracy delta vs Stage 1: F1 -{delta_f1}")

    stages.append("2. Quantized GCN (plaintext)")
    metrics.append(m2)
    latencies.append(lat_q)

    # -- Stage 3: Encrypted GCN -------------------------------------
    print("\n[evaluate] -- Stage 3: HE Encrypted GCN --------------")
    print("  [!] Full encrypted inference requires TenSEAL.")
    print("  [!] Run src/encrypt_infer.py for end-to-end HE benchmark.")
    print("  [!] Placeholder metrics shown — replace after HE run.")

    # Placeholder for encrypted stage (filled after HE benchmark run)
    m3 = {k: "N/A (run encrypt_infer.py)" for k in m2}
    m3["latency_note"] = "Typically 100x-1000x slower than plaintext"
    stages.append("3. HE Encrypted GCN")
    metrics.append(m3)
    latencies.append(None)

    # -- Save report CSV --------------------------------------------
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "accuracy", "f1_macro", "f1_illicit",
                         "precision", "recall", "auc_roc", "latency_s"])
        for stage, m, lat in zip(stages, metrics, latencies):
            writer.writerow([
                stage,
                m.get("accuracy",   "N/A"),
                m.get("f1_macro",   "N/A"),
                m.get("f1_illicit", "N/A"),
                m.get("precision",  "N/A"),
                m.get("recall",     "N/A"),
                m.get("auc_roc",    "N/A"),
                f"{lat:.4f}" if lat else "N/A",
            ])
    print(f"\n[evaluate] Report saved -> {report_path}")

    # -- Plots ------------------------------------------------------
    plot_metric_comparison(stages[:2], metrics[:2], plots_dir)
    plot_latency_comparison(stages[:2], latencies[:2], plots_dir)

    # -- Compliance Report & Alerts ---------------------------------
    try:
        from src.alert import ThresholdAlerter, AuditLogger, ComplianceReporter
        test_indices = data.test_mask.nonzero(as_tuple=True)[0][:100].numpy()
        sample_probs = probs_pt[data.test_mask.numpy()][:100, 1]
        sample_ids   = [f"txn_{int(i):06d}" for i in test_indices]
        alerter      = ThresholdAlerter(threshold=cfg["eval"]["threshold"], institution_id="INSTITUTION_A")
        alerts       = alerter.check_batch(sample_ids, sample_probs)
        log_path     = os.path.join(results_dir, "audit_log.jsonl")
        AuditLogger(log_path).log_batch(alerts)
        reporter     = ComplianceReporter(results_dir, log_path)
        reporter.generate()
    except Exception as e:
        print(f"[evaluate] Notice: Alert/Compliance report generation skipped ({e})")

    return stages, metrics, latencies


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def plot_metric_comparison(stages: list, metrics: list, plots_dir: str):
    """Bar chart comparing F1 scores across stages."""
    if not MATPLOTLIB_AVAILABLE or plt is None:
        print("[evaluate] Note: Matplotlib unavailable - skipping offline PNG generation (metrics saved to CSV).")
        return
    metric_names = ["accuracy", "f1_macro", "f1_illicit", "auc_roc"]
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (stage, m) in enumerate(zip(stages, metrics)):
        vals = [float(m[k]) for k in metric_names]
        ax.bar(x + i * width, vals, width, label=stage)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("GNN Performance: Plaintext vs Quantized")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(plots_dir, "stage_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[evaluate] Plot saved -> {path}")


def plot_latency_comparison(stages: list, latencies: list, plots_dir: str):
    """Bar chart of inference latency per stage."""
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["steelblue", "darkorange"]
    ax.bar(stages, [l * 1000 for l in latencies], color=colors)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Comparison")
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(plots_dir, "latency_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[evaluate] Plot saved -> {path}")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate all pipeline stages")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate_all_stages(cfg)


if __name__ == "__main__":
    main()
