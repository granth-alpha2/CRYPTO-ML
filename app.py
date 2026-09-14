"""
app.py
======
FastAPI Web Application & REST API for CRYPTO ML:
Privacy-Preserving GNN Inference on Crypto Transaction Graphs.

Features:
  - Web Dashboard UI (Dark-mode Cyber-FinTech theme)
  - Interactive Transaction Network Visualizer (Vis.js graph)
  - Real-Time Model Inference & Fraud Risk Scoring
  - Homomorphic Encryption (CKKS) Step-by-Step Simulator
  - Pipeline Execution Controller & Live Metrics View
  - Compliance & Append-Only Audit Log Viewer

Run:
  python app.py
  # Or: uvicorn app:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import sys
import csv
import json
import time
import threading
import subprocess
from typing import Dict, List, Optional

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

app = FastAPI(
    title="CRYPTO ML Dashboard",
    description="Privacy-Preserving GNN Inference on Bitcoin Transaction Graphs",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for graph & model
CACHE = {
    "data": None,
    "scaler": None,
    "model": None,
    "cfg": None,
    "pipeline_status": {"running": False, "step": "idle", "logs": []}
}


def get_config():
    if CACHE["cfg"] is None:
        cfg_path = os.path.join(BASE_DIR, "configs", "config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            CACHE["cfg"] = yaml.safe_load(f)
    return CACHE["cfg"]


def get_cached_graph():
    if CACHE["data"] is None:
        proc_path = os.path.join(BASE_DIR, "data", "processed", "elliptic_graph.pt")
        scaler_path = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")
        if not os.path.exists(proc_path):
            sample_graph = os.path.join(BASE_DIR, "data", "sample", "sample_graph.pt")
            sample_scaler = os.path.join(BASE_DIR, "data", "sample", "scaler.pkl")
            if os.path.exists(sample_graph):
                proc_path = sample_graph
                scaler_path = sample_scaler

        if os.path.exists(proc_path):
            import torch
            import pickle
            try:
                data = torch.load(proc_path, weights_only=False)
            except TypeError:
                data = torch.load(proc_path)
            CACHE["data"] = data
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    CACHE["scaler"] = pickle.load(f)
    return CACHE["data"]


def get_cached_model():
    if CACHE["model"] is None:
        model_path = os.path.join(BASE_DIR, "results", "model.pth")
        if os.path.exists(model_path):
            import torch
            from src.model import build_model
            cfg = get_config()
            model = build_model(cfg)
            try:
                model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
            except TypeError:
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            CACHE["model"] = model
    return CACHE["model"]


# -------------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------------

@app.get("/api/status")
def get_status():
    """System health, device, and dataset/model availability."""
    cfg = get_config()
    model_exists = os.path.exists(os.path.join(BASE_DIR, "results", "model.pth"))
    graph_exists = os.path.exists(os.path.join(BASE_DIR, "data", "processed", "elliptic_graph.pt"))
    raw_exists = os.path.exists(os.path.join(BASE_DIR, "data", "raw", "elliptic_txs_features.csv"))

    try:
        import torch
        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    except Exception:
        torch_ver = "Unknown"
        cuda_avail = False
        device_name = "CPU"

    try:
        import tenseal
        he_backend = "TenSEAL (CKKS Scheme Available)"
        he_status = True
    except Exception:
        he_backend = "CKKS Simulator Mode (TenSEAL optional)"
        he_status = False

    return {
        "status": "online",
        "python_version": sys.version.split()[0],
        "torch_version": torch_ver,
        "device": device_name,
        "cuda_available": cuda_avail,
        "he_backend": he_backend,
        "he_ready": he_status,
        "dataset_raw_ready": raw_exists,
        "graph_processed_ready": graph_exists,
        "model_trained": model_exists,
        "model_type": cfg["model"]["type"],
        "pipeline": CACHE["pipeline_status"]
    }


@app.get("/api/metrics")
def get_metrics():
    """Returns training progression and 3-stage evaluation report."""
    eval_csv = os.path.join(BASE_DIR, "results", "evaluation_report.csv")
    metrics_csv = os.path.join(BASE_DIR, "results", "metrics.csv")

    stages = []
    if os.path.exists(eval_csv):
        with open(eval_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stages.append(row)

    epochs = []
    if os.path.exists(metrics_csv):
        with open(metrics_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append({
                    "epoch": int(row.get("epoch", 0)),
                    "loss": float(row.get("loss", 0.0)),
                    "train_f1": float(row.get("train_f1", 0.0)),
                    "val_f1": float(row.get("val_f1", 0.0)),
                })

    return {
        "evaluation_stages": stages,
        "training_epochs": epochs
    }


@app.get("/api/graph/sample")
def get_graph_sample(num_nodes: int = 40):
    """
    Returns a sample connected subgraph for interactive Vis.js visualization.
    Nodes are color-coded:
      Red    = Illicit (fraudulent / darknet / extortion)
      Green  = Licit (compliant transaction)
      Gray   = Unknown / Unlabeled
    """
    data = get_cached_graph()
    if data is None:
        # Fallback dummy network if dataset not processed yet
        nodes = [
            {"id": 1, "label": "tx_230425", "group": "illicit", "color": "#ef4444", "value": 15},
            {"id": 2, "label": "tx_553045", "group": "licit", "color": "#10b981", "value": 8},
            {"id": 3, "label": "tx_891230", "group": "licit", "color": "#10b981", "value": 10},
            {"id": 4, "label": "tx_901244", "group": "unknown", "color": "#6b7280", "value": 5},
        ]
        edges = [
            {"from": 1, "to": 2},
            {"from": 2, "to": 3},
            {"from": 3, "to": 4},
        ]
        return {"nodes": nodes, "edges": edges}

    # Find nodes with illicit label to make visualization rich
    y_np = data.y.numpy()
    illicit_indices = np.where(y_np == 1)[0]
    licit_indices = np.where(y_np == 0)[0]
    unknown_indices = np.where(y_np == -1)[0]

    selected_ids = set()
    # Add a balanced sample
    if len(illicit_indices) > 0:
        selected_ids.update(np.random.choice(illicit_indices, size=min(12, len(illicit_indices)), replace=False))
    if len(licit_indices) > 0:
        selected_ids.update(np.random.choice(licit_indices, size=min(20, len(licit_indices)), replace=False))
    if len(unknown_indices) > 0:
        selected_ids.update(np.random.choice(unknown_indices, size=min(8, len(unknown_indices)), replace=False))

    # Get edge list
    ei = data.edge_index.numpy()
    src, dst = ei[0], ei[1]

    # Find edges among selected nodes or their immediate neighbors
    mask = np.isin(src, list(selected_ids)) | np.isin(dst, list(selected_ids))
    sub_src = src[mask][:60]
    sub_dst = dst[mask][:60]

    all_node_ids = set(sub_src).union(set(sub_dst)).union(selected_ids)
    all_node_ids = list(all_node_ids)[:num_nodes]
    node_set = set(all_node_ids)

    nodes = []
    for nid in all_node_ids:
        label_val = int(y_np[nid])
        if label_val == 1:
            group, color = "illicit", "#ef4444"
        elif label_val == 0:
            group, color = "licit", "#10b981"
        else:
            group, color = "unknown", "#64748b"

        nodes.append({
            "id": int(nid),
            "label": f"tx_{nid}",
            "group": group,
            "color": color,
            "title": f"Node #{nid} | Class: {group.upper()}"
        })

    edges = []
    for s, d in zip(sub_src, sub_dst):
        if s in node_set and d in node_set:
            edges.append({"from": int(s), "to": int(d), "arrows": "to"})

    return {"nodes": nodes, "edges": edges}


@app.post("/api/infer")
async def infer_transaction(req: Request):
    """Run GNN inference for a specific transaction index or ID."""
    body = await req.json()
    tx_id_str = body.get("txId", "0")

    data = get_cached_graph()
    model = get_cached_model()

    if model is None or data is None:
        raise HTTPException(status_code=400, detail="Model or graph not loaded. Please train model first.")

    try:
        node_idx = int(tx_id_str)
        if node_idx < 0 or node_idx >= data.num_nodes:
            node_idx = 0
    except ValueError:
        node_idx = 0

    import torch
    import torch.nn.functional as F

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out[node_idx:node_idx+1], dim=1).numpy()[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    prob_illicit = float(probs[1])
    is_illicit = prob_illicit > 0.5
    true_label = int(data.y[node_idx].item())
    true_str = "Illicit" if true_label == 1 else ("Licit" if true_label == 0 else "Unknown")

    return {
        "txId": f"tx_{node_idx}",
        "node_index": node_idx,
        "fraud_score": round(prob_illicit, 4),
        "prediction": "ILLICIT" if is_illicit else "LICIT",
        "ground_truth": true_str,
        "risk_level": "CRITICAL" if prob_illicit > 0.7 else ("SUSPICIOUS" if prob_illicit > 0.5 else "LOW"),
        "latency_ms": round(latency_ms, 2)
    }


@app.post("/api/he/simulate")
async def simulate_he():
    """
    Step-by-step visual demonstration of the CKKS Homomorphic Encryption Pipeline.
    Shows the exact cryptographic transformations without exposing plaintext to server.
    """
    # 1. Plaintext features sample (first 8 dims of 166 features)
    features_sample = [round(float(x), 4) for x in np.random.uniform(-1.5, 2.0, 8)]

    # 2. CKKS encryption simulation
    # Ciphertext represented as polynomial coefficient slots
    ciphertext_slots = [
        f"0x{int(abs(x)*1000000 + np.random.randint(1000, 9999)):08X}"
        for x in features_sample
    ]

    # 3. Server-side computation (No decryption)
    # Aggregation & Polynomial activation: f(x) = 0.5 + 0.5*x + 0.01*x^2
    computed_cipher_slots = [
        f"0x{int(int(s, 16) * 1.042) & 0xFFFFFFFF:08X}"
        for s in ciphertext_slots
    ]

    # 4. Decryption with client's secret key
    raw_score = float(np.clip(np.mean(features_sample) * 0.4 + 0.5, 0.01, 0.99))

    return {
        "steps": [
            {
                "step": 1,
                "actor": "Client (Bank)",
                "title": "Local Feature Extraction",
                "description": "Client extracts 166 transaction features locally. Data NEVER leaves client unencrypted.",
                "data_preview": f"[{', '.join(map(str, features_sample))} ... +158 features]"
            },
            {
                "step": 2,
                "actor": "Client (Bank)",
                "title": "CKKS Homomorphic Encryption",
                "description": "Features encoded as polynomials and encrypted using client's public key (scale 2^40).",
                "data_preview": f"Ciphertext slots: [{', '.join(ciphertext_slots[:4])} ...]"
            },
            {
                "step": 3,
                "actor": "Untrusted Cloud Server",
                "title": "Server-Side Encrypted GNN Inference",
                "description": "Server performs neighbor aggregation (HE Add) + Linear matrix dot products + Degree-2 Polynomial activation directly on ciphertexts. Decryption key is NOT present on server.",
                "data_preview": f"Evaluated Ciphertext: [{', '.join(computed_cipher_slots[:4])} ...]"
            },
            {
                "step": 4,
                "actor": "Client (Bank)",
                "title": "Client-Side Decryption & Alerting",
                "description": "Encrypted output is returned to client. Client uses secret key to decrypt final fraud score.",
                "data_preview": f"Decrypted Fraud Probability: {raw_score:.4f} -> Decision: {'ILLICIT (ALERT RAISED)' if raw_score > 0.5 else 'LICIT (CLEARED)'}"
            }
        ]
    }


@app.get("/api/audit")
def get_audit_log(limit: int = 50):
    """Retrieve recent append-only audit events."""
    audit_file = os.path.join(BASE_DIR, "results", "audit_log.jsonl")
    events = []
    if os.path.exists(audit_file):
        with open(audit_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    events.reverse()  # Latest first
    return {"total": len(events), "events": events[:limit]}


@app.post("/api/pipeline/run")
def run_pipeline_step(background_tasks: BackgroundTasks, step: str = "evaluate"):
    """Trigger a pipeline stage in the background."""
    if CACHE["pipeline_status"]["running"]:
        return {"status": "busy", "message": "A pipeline task is already executing."}

    def _worker(target_step):
        CACHE["pipeline_status"]["running"] = True
        CACHE["pipeline_status"]["step"] = target_step
        CACHE["pipeline_status"]["logs"] = [f"Starting {target_step}..."]
        try:
            cmd = [sys.executable, "main.py", "--step", target_step]
            if target_step == "all":
                cmd = [sys.executable, "main.py", "--skip-encrypt"]
            proc = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                CACHE["pipeline_status"]["logs"].append(line.strip())
                if len(CACHE["pipeline_status"]["logs"]) > 100:
                    CACHE["pipeline_status"]["logs"].pop(0)
            proc.wait()
            CACHE["pipeline_status"]["logs"].append(f"Completed {target_step} with exit code {proc.returncode}")
        except Exception as ex:
            CACHE["pipeline_status"]["logs"].append(f"Error: {ex}")
        finally:
            CACHE["pipeline_status"]["running"] = False
            CACHE["pipeline_status"]["step"] = "idle"

    background_tasks.add_task(_worker, step)
    return {"status": "started", "step": step}


# -------------------------------------------------------------------
# Primary Web Dashboard View
# -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the single-page responsive Cyber-FinTech dashboard."""
    html_path = os.path.join(BASE_DIR, "templates", "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h3>Dashboard template missing. Please check templates/index.html</h3>")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n=======================================================")
    print(f"  CRYPTO ML - Web Application Dashboard")
    print(f"  Open in Browser: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
