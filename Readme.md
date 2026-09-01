# 🔐 Privacy-Preserving GNN Inference on Crypto Transaction Graphs

> **Machine learning that computes directly on *encrypted* graph data.**  
> The graph is protected by proven cryptography; a GNN performs inference without ever seeing the plaintext.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-latest-red)
![TenSEAL](https://img.shields.io/badge/TenSEAL-CKKS%2FBFV-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

## 📖 Overview

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

## 🏛️ Architecture

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

> **Known limitation:** Most HE schemes protect node/edge *feature values*, not the graph *topology*. The server still sees which nodes are connected to route message passing. Fully hiding topology requires SMPC or oblivious graph protocols — an active research problem.

---

## 🗂️ Dataset

**[Elliptic Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)** — A public graph of ~200,000 Bitcoin transactions:

| Property | Details |
|---|---|
| Nodes | ~203,769 transactions |
| Edges | ~234,355 payment flows |
| Node features | 166 features per transaction |
| Labels | `licit` / `illicit` / `unknown` |
| Benchmark use | GNN-based fraud & AML detection |

---

## 🛠️ Tech Stack

| Purpose | Tool |
|---|---|
| GNN model | PyTorch + PyTorch Geometric (GCN or GIN) |
| Homomorphic Encryption | [TenSEAL](https://github.com/OpenMined/TenSEAL) (CKKS via Microsoft SEAL) or [Concrete ML](https://github.com/zama-ai/concrete-ml) |
| Quantization / Pruning | PyTorch quantization utilities |
| Experiment tracking | CSV/JSON logs → optionally Weights & Biases |
| Data processing | pandas, scikit-learn, NetworkX |

---

## 📁 Project Structure

```
CRYPTO ML/
├── data/
│   ├── raw/                    # Elliptic dataset (gitignored)
│   └── processed/              # PyG-ready graph objects
├── src/
│   ├── model.py                # GNN definition (GCN / GIN, plaintext)
│   ├── train.py                # Plaintext training loop
│   ├── quantize.py             # Quantization + polynomial activation swap
│   ├── encrypt_infer.py        # HE encryption + encrypted inference pipeline
│   └── evaluate.py             # Accuracy + latency comparison (plaintext vs encrypted)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_baseline_gnn.ipynb   # Plaintext GNN experiments
│   └── 03_encrypted_infer.ipynb# Encrypted inference walkthrough
├── results/
│   ├── metrics.csv             # Accuracy / F1 / latency at each stage
│   └── plots/                  # Comparison charts
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (optional but recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/crypto-ml.git
cd crypto-ml

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install torch torch-geometric tenseal pandas scikit-learn networkx matplotlib
```

### Dataset Setup

1. Download the [Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) from Kaggle
2. Place the CSV files in `data/raw/`
3. Run the preprocessing script:
   ```bash
   python src/preprocess.py
   ```

---

## 🗺️ Roadmap

- [ ] **Data Pipeline** — Load and preprocess the Elliptic dataset into a PyTorch Geometric graph
- [ ] **Baseline Model** — Train a GCN/GIN in plaintext; record accuracy, F1, and latency
- [ ] **Quantization** — Quantize the trained model; replace ReLU with a low-degree polynomial approximation
- [ ] **Plaintext Approximation Eval** — Re-evaluate quantized + approximated model in plaintext (isolate accuracy loss from this step)
- [ ] **Encrypted Inference** — Implement encrypted inference with TenSEAL or Concrete ML
- [ ] **End-to-End Test** — Run inference on encrypted test graphs; decrypt only the final prediction
- [ ] **Comparison Report** — Compare plaintext baseline vs. quantized plaintext vs. encrypted inference on accuracy and wall-clock latency
- [ ] **Write-up** — Document findings, including what is and isn't protected (features vs. topology)

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy / F1** | On illicit-transaction classification, at each pipeline stage |
| **Latency** | Plaintext inference vs. encrypted inference (per-graph and per-node) |
| **Accuracy Delta** | Degradation attributable specifically to quantization + polynomial activation approximation |
| **Encryption Overhead** | Ratio of encrypted vs. plaintext inference wall-clock time |

---

## ⚠️ Limitations

- **Speed:** Encrypted inference is typically **10×–10,000× slower** than plaintext, depending on the HE scheme and model size.
- **Training:** Training under full homomorphic encryption is not practical today — models are trained in plaintext and only encrypted at inference time.
- **Topology exposure:** Graph topology (which nodes connect) is generally still visible to the server; only feature values are protected.
- **Polynomial approximation:** Replacing ReLU with polynomial approximations introduces some accuracy loss, especially in deeper networks.

---

## 🔀 Alternatives Considered

| Approach | How it works | Tradeoff |
|---|---|---|
| **Federated Learning** | Data never leaves each institution; only model updates are shared | Can be combined with HE on gradients; weaker per-inference privacy |
| **Differential Privacy** | Add calibrated noise to outputs/gradients | Much cheaper computationally, but weaker and probabilistic guarantee |
| **Secure Multi-Party Computation (SMPC)** | Multiple parties jointly compute without revealing their inputs | Requires interactive protocol; good for small models |

---

## 🧠 Why This Approach

- **Formal security:** CKKS-based homomorphic encryption has formal, proven security properties. A "learned" encryption scheme does not — it is only as strong as the specific adversary it was trained against.
- **GNN + HE synergy:** GNNs are a natural fit for HE because their core operation — aggregating neighbor features and applying linear transforms — reduces to **additions and multiplications**, which CKKS supports natively.
- **Real-world relevance:** This mirrors active research in privacy-preserving GNN inference for AML (anti-money-laundering), federated healthcare graphs, and financial fraud detection.

---

## 📚 References

| Paper | Link |
|---|---|
| Abadi & Andersen — *Learning to Protect Communications with Adversarial Neural Cryptography* (2016) | [arxiv.org/abs/1610.06918](https://arxiv.org/abs/1610.06918) |
| *Privacy-Preserving Graph-Based ML with FHE for Collaborative AML* | [arxiv.org/pdf/2411.02926](https://arxiv.org/pdf/2411.02926) |
| *FedGraphHE: Privacy-Preserving Federated GNN with Dynamic HE* | [NIH PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12768379/) |
| *DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning* | [arxiv.org/pdf/2507.05649](https://arxiv.org/pdf/2507.05649) |
| *EDLaaS: Fully Homomorphic Encryption Over Neural Network Graphs* | [arxiv.org/pdf/2110.13638](https://arxiv.org/pdf/2110.13638) |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.