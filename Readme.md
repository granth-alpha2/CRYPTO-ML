# ðŸ” Privacy-Preserving GNN Inference on Crypto Transaction Graphs

> **Machine learning that computes directly on *encrypted* graph data.**  
> The graph is protected by proven cryptography; a GNN performs inference without ever seeing the plaintext.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-latest-red)
![TenSEAL](https://img.shields.io/badge/TenSEAL-CKKS%2FBFV-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

## ðŸ“– Overview

This project separates two concerns that are often wrongly bundled together:

| Concern | Approach |
|---|---|
| **Protecting the data** | Standard, formally-analyzed **Homomorphic Encryption** (CKKS scheme via TenSEAL / Concrete ML) |
| **Computing on protected data** | A **Graph Neural Network (GNN)** that classifies nodes/transactions *without ever decrypting them* |

The goal is a working end-to-end pipeline that:
1. Trains a GNN on the **Elliptic Bitcoin dataset** (real-world crypto transaction graph)
2. Converts the model to run under homomorphic encryption
3. Reports the accuracy/latency tradeoff vs. plaintext inference

---

## ðŸ›ï¸ Architecture

### System Overview

```
+====================================================================================+
|                   PRIVACY-PRESERVING GNN INFERENCE PIPELINE                        |
+====================================================================================+

  +-----------------------------+         +------------------------------------------+
  |         CLIENT SIDE         |         |               SERVER SIDE                |
  |   (holds private key)       |         |         (never sees plaintext)            |
  +-----------------------------+         +------------------------------------------+

         |                                                    |
         v                                                    v
  +-----------------+    CKKS Encrypt    +----------------------------------------+
  |  Raw Graph Data |  --------------->  |          Encrypted Graph               |
  |                 |                    |                                        |
  |  Nodes: txn_i   |                    |  Nodes: Enc(f1), Enc(f2), ..Enc(f166)  |
  |  Edges: payment |                    |  Edges: topology visible (known        |
  |  Feats: 166-dim |                    |         limitation, features hidden)   |
  +-----------------+                    +----------------------------------------+
         |                                                    |
         |  [OFFLINE TRAINING - plaintext]                    |
         v                                                    v
  +-----------------+                    +----------------------------------------+
  |  Train GNN      |                    |     GNN Inference on Ciphertext        |
  |  (plaintext)    |  -- weights -->    |                                        |
  |                 |                    |  Layer 1: Enc(Aij * Wh)   [HE Mult]    |
  |  GCN / GIN      |                    |  Layer 2: Enc(Aij * Wh)   [HE Mult]    |
  |  + Quantize     |                    |  Activ. : poly approx(x)  [HE Mult]    |
  |  + Poly approx  |                    |  Output : Enc(logits)                  |
  +-----------------+                    +----------------------------------------+
                                                              |
                                          Send Enc(result) back to client
                                                              |
                                                              v
                                         +----------------------------------------+
                                         |           CLIENT DECRYPTS              |
                                         |                                        |
                                         |   Enc(logits) --> logits               |
                                         |   argmax(logits) = licit / illicit     |
                                         +----------------------------------------+
```

---

### GNN Message-Passing Layer Detail (under HE)

```
  One Message-Passing Layer on Encrypted Features
  -----------------------------------------------------------------------

   Node v:  Enc(hv) --+
                       |   Aggregate neighbours
   Node u1: Enc(hu1)--+   (topology known to     +----------------------+
   Node u2: Enc(hu2)--+--> server, HE addition -->| Enc(W * sum(hu) + b) |
   Node u3: Enc(hu3)--+   over neighbour feats)   |  Polynomial approx   |
                                                   |  replaces ReLU:      |
                                                   |  f(x)=a0+a1*x+a2*x^2|
                                                   +----------------------+
                                                              |
                                                         Next layer
```

---

### Pipeline Stages & What Is Protected

```
  +----------------------+---------------------+--------------------------+
  |       Stage          |   Feature Values    |      Graph Topology      |
  +----------------------+---------------------+--------------------------+
  | 1. Plaintext GCN     |  [VISIBLE]          |  [VISIBLE]               |
  | 2. Quantized GCN     |  [VISIBLE]          |  [VISIBLE]               |
  | 3. HE Encrypted GCN  |  [HIDDEN - CKKS]    |  [WARNING - visible]     |
  | 4. Full HE + SMPC    |  [HIDDEN]           |  [HIDDEN - future work]  |
  +----------------------+---------------------+--------------------------+
```

> **Key guarantee:** The server never holds the private key. It operates exclusively on ciphertext, and only the client can decrypt the final prediction.

> **Known limitation:** Most HE schemes protect node/edge *feature values*, not the graph *topology*. The server still sees which nodes are connected to route message passing. Fully hiding topology requires SMPC or oblivious graph protocols â€” an active research problem.

---

## ðŸ—‚ï¸ Dataset

**[Elliptic Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)** â€” A public graph of ~200,000 Bitcoin transactions:

| Property | Details |
|---|---|
| Nodes | ~203,769 transactions |
| Edges | ~234,355 payment flows |
| Node features | 166 features per transaction |
| Labels | `licit` / `illicit` / `unknown` |
| Benchmark use | GNN-based fraud & AML detection |

---

## ðŸ› ï¸ Tech Stack

| Purpose | Tool |
|---|---|
| GNN model | PyTorch + PyTorch Geometric (GCN or GIN) |
| Homomorphic Encryption | [TenSEAL](https://github.com/OpenMined/TenSEAL) (CKKS via Microsoft SEAL) or [Concrete ML](https://github.com/zama-ai/concrete-ml) |
| Quantization / Pruning | PyTorch quantization utilities |
| Experiment tracking | CSV/JSON logs â†’ optionally Weights & Biases |
| Data processing | pandas, scikit-learn, NetworkX |

---

## ðŸ“ Project Structure

```
CRYPTO ML/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                    # Elliptic dataset (gitignored)
â”‚   â””â”€â”€ processed/              # PyG-ready graph objects
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ __init__.py             # Package init
â”‚   â”œâ”€â”€ preprocess.py           # CSV â†’ PyG graph (203k nodes, 166 features)
â”‚   â”œâ”€â”€ model.py                # GNN definition (GCN / GIN + PolyActivation)
â”‚   â”œâ”€â”€ train.py                # Plaintext training loop + early stopping
â”‚   â”œâ”€â”€ quantize.py             # 8-bit quantization + ReLUâ†’Poly swap
â”‚   â”œâ”€â”€ encrypt_infer.py        # CKKS HE context + full encrypted GCN forward
â”‚   â”œâ”€â”€ evaluate.py             # 3-stage accuracy / latency comparison + plots
â”‚   â”œâ”€â”€ alert.py                # ThresholdAlerter + AuditLogger + ComplianceReporter
â”‚   â”œâ”€â”€ client_sdk.py           # CryptoMLClient â€” high-level SDK for data owners
â”‚   â””â”€â”€ utils.py                # Shared helpers (seed, logger, timer, metrics)
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 01_eda.ipynb            # Exploratory data analysis
â”‚   â”œâ”€â”€ 02_baseline_gnn.ipynb   # Plaintext GNN training + evaluation
â”‚   â””â”€â”€ 03_encrypted_infer.ipynb# Full HE inference demo + compliance report
â”œâ”€â”€ results/
â”‚   â”œâ”€â”€ metrics.csv             # Per-epoch training metrics
â”‚   â”œâ”€â”€ evaluation_report.csv   # 3-stage comparison
â”‚   â”œâ”€â”€ audit_log.jsonl         # Encrypted audit trail
â”‚   â”œâ”€â”€ compliance_report.html  # Regulator-facing HTML report
â”‚   â””â”€â”€ plots/                  # Comparison charts (PNG)
â”œâ”€â”€ configs/
â”‚   â””â”€â”€ config.yaml             # All hyperparameters in one place
â”œâ”€â”€ templates/
â”‚   â””â”€â”€ index.html              # Cyber-FinTech Dark Mode Web Dashboard
â”œâ”€â”€ app.py                      # FastAPI Web Application & REST Server
â”œâ”€â”€ run_app.bat                 # 1-Click Web Dashboard Launcher (http://127.0.0.1:8000)
â”œâ”€â”€ setup.bat                   # 1-Click Automated Setup & Pipeline Runner
â”œâ”€â”€ main.py                     # Entry point (delegates to run_pipeline.py)
â”œâ”€â”€ run_pipeline.py             # Full pipeline runner with error handling
â”œâ”€â”€ download_dataset.py         # Automated Kaggle dataset downloader
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .gitignore
â”œâ”€â”€ LICENSE
â””â”€â”€ README.md
```

---

## ⚡ Quick Start (1-Click Automated Run)

### On Windows:
Just **double-click `setup.bat`** (or `run_app.bat` for the web dashboard):
- Automatically sets up Python dependencies.
- Downloads the Elliptic Bitcoin dataset.
- Preprocesses the graph and trains the GNN.
- Boots the interactive Web Dashboard at `http://127.0.0.1:8000`.

### On macOS / Linux / Terminal:
```bash
# Clone the repository
git clone https://github.com/your-username/crypto-ml.git
cd crypto-ml

# Install dependencies
pip install -r requirements.txt

# Run full ML pipeline
python main.py --skip-encrypt

# Launch interactive Web UI
python app.py
```
Open your browser at **`http://127.0.0.1:8000`**.

---

## 🖥️ Interactive Web Dashboard Features

The dashboard includes 6 core interactive modules:
1. **Overview & Benchmarks**: Real-time KPI cards (Nodes, Edges, Accuracy, Quantization delta) and training curve charts.
2. **Transaction Network Visualizer**: Interactive force-directed payment graph (Vis.js). Click any node to inspect connected flows.
3. **Fraud Risk Inspector**: Enter any Bitcoin transaction index (0 to 203,768) to get instant GNN risk classification.
4. **Homomorphic Encryption (HE) Simulator**: Step-by-step interactive demonstration of client encryption, server matrix operations, and client decryption.
5. **Compliance & Audit Logs**: Append-only log table compliant with GDPR, RBI data localization, and FATF AML reporting.
6. **Pipeline Controller**: Trigger preprocessing, training, or evaluation directly from the browser with a live console log.

---

## ðŸ—ºï¸  Roadmap

- [x] **Data Pipeline** â€” Load and preprocess the Elliptic dataset into a PyTorch Geometric graph
- [x] **Baseline Model** â€” Train a GCN/GIN in plaintext; record accuracy, F1, and latency
- [x] **Quantization** â€” Quantize the trained model; replace ReLU with a low-degree polynomial approximation
- [x] **Plaintext Approximation Eval** â€” Re-evaluate quantized + approximated model in plaintext (isolate accuracy loss from this step)
- [x] **Encrypted Inference** â€” Full encrypted GCN forward pass with CKKS (TenSEAL)
- [x] **End-to-End Test** â€” Run inference on encrypted test graphs; decrypt only the final prediction
- [x] **Alert & Audit System** â€” Threshold alerts + encrypted audit log + HTML compliance report
- [x] **Client SDK** â€” `CryptoMLClient` API hiding all HE complexity from data owners
- [x] **Notebooks** â€” EDA, training walkthrough, and HE inference demo
- [ ] **Full 3-Stage Benchmark** â€” Compare plaintext vs quantized vs encrypted on accuracy + wall-clock latency (requires TenSEAL + hardware time)
- [ ] **Write-up** â€” Final research findings documentplaintext baseline vs. quantized plaintext vs. encrypted inference on accuracy and wall-clock latency
- [ ] **Write-up** â€” Document findings, including what is and isn't protected (features vs. topology)

---

## ðŸ“Š Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy / F1** | On illicit-transaction classification, at each pipeline stage |
| **Latency** | Plaintext inference vs. encrypted inference (per-graph and per-node) |
| **Accuracy Delta** | Degradation attributable specifically to quantization + polynomial activation approximation |
| **Encryption Overhead** | Ratio of encrypted vs. plaintext inference wall-clock time |

---

## âš ï¸ Limitations

- **Speed:** Encrypted inference is typically **10Ã—â€“10,000Ã— slower** than plaintext, depending on the HE scheme and model size.
- **Training:** Training under full homomorphic encryption is not practical today â€” models are trained in plaintext and only encrypted at inference time.
- **Topology exposure:** Graph topology (which nodes connect) is generally still visible to the server; only feature values are protected.
- **Polynomial approximation:** Replacing ReLU with polynomial approximations introduces some accuracy loss, especially in deeper networks.

---

## ðŸ”€ Alternatives Considered

| Approach | How it works | Tradeoff |
|---|---|---|
| **Federated Learning** | Data never leaves each institution; only model updates are shared | Can be combined with HE on gradients; weaker per-inference privacy |
| **Differential Privacy** | Add calibrated noise to outputs/gradients | Much cheaper computationally, but weaker and probabilistic guarantee |
| **Secure Multi-Party Computation (SMPC)** | Multiple parties jointly compute without revealing their inputs | Requires interactive protocol; good for small models |

---

## ðŸ§  Why This Approach

- **Formal security:** CKKS-based homomorphic encryption has formal, proven security properties. A "learned" encryption scheme does not â€” it is only as strong as the specific adversary it was trained against.
- **GNN + HE synergy:** GNNs are a natural fit for HE because their core operation â€” aggregating neighbor features and applying linear transforms â€” reduces to **additions and multiplications**, which CKKS supports natively.
- **Real-world relevance:** This mirrors active research in privacy-preserving GNN inference for AML (anti-money-laundering), federated healthcare graphs, and financial fraud detection.

---

## ðŸ“š References

| Paper | Link |
|---|---|
| Abadi & Andersen â€” *Learning to Protect Communications with Adversarial Neural Cryptography* (2016) | [arxiv.org/abs/1610.06918](https://arxiv.org/abs/1610.06918) |
| *Privacy-Preserving Graph-Based ML with FHE for Collaborative AML* | [arxiv.org/pdf/2411.02926](https://arxiv.org/pdf/2411.02926) |
| *FedGraphHE: Privacy-Preserving Federated GNN with Dynamic HE* | [NIH PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12768379/) |
| *DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning* | [arxiv.org/pdf/2507.05649](https://arxiv.org/pdf/2507.05649) |
| *EDLaaS: Fully Homomorphic Encryption Over Neural Network Graphs* | [arxiv.org/pdf/2110.13638](https://arxiv.org/pdf/2110.13638) |

---

## ðŸ“„ License

This project is licensed under the **MIT License** â€” see the [LICENSE](LICENSE) file for details.