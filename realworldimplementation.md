# 🌍 Real World Implementation Guide
### Privacy-Preserving GNN Inference on Crypto Transaction Graphs

> This document explains how the system described in [README.md](./README.md) would be
> deployed in a real production environment — who uses it, how it works end-to-end,
> and what it takes to build it.

---

## 🏦 The Core Real World Problem: AML Across Multiple Banks

Imagine **3 banks** — HDFC, SBI, Citibank. Each has millions of crypto transactions.
A fraudster moves money **across all 3 banks**.

| The Challenge | Why It Is Hard |
|---|---|
| Each bank sees only part of the picture | Fraud spans multiple institutions |
| Banks cannot share raw customer data | RBI rules, GDPR, legal liability |
| Central databases are a honeypot | Single breach exposes everyone |
| Manual review is too slow | Millions of transactions per day |

**This system solves it:**
All 3 banks contribute encrypted transaction graphs to a central AI server.
The server runs a GNN on the encrypted data.
Each bank gets an encrypted fraud score and decrypts only their own result.

**Nobody saw anyone else raw data. Ever.**

---

## 🏗️ Production System Architecture

```
+======================================================================+
|                    PRODUCTION SYSTEM DESIGN                          |
+======================================================================+
|                                                                      |
|  [DATA OWNERS]          [SECURE COMPUTE]          [REGULATORS]       |
|                                                                      |
|  Bank A --encrypt--+                         +-- Audit logs         |
|  Bank B --encrypt--+--> AI Server            |   (encrypted)        |
|  Bank C --encrypt--+   (GNN inference)  -----+                      |
|                         |                    +-- Compliance reports  |
|                         v                                            |
|                  Alert Service                                       |
|                  "Flag txn #X as illicit"                            |
|                         |                                            |
|                         v                                            |
|                 [Law Enforcement / FIU]                              |
+======================================================================+
```

### How the 3-Bank Collaboration Works

```
  HDFC Bank                SBI Bank              Citibank
  [txn data]               [txn data]             [txn data]
      |                        |                       |
   Encrypt                  Encrypt                 Encrypt
      |                        |                       |
      +------------------------+-----------------------+
                               |
                        Central AI Server
                     (sees only ciphertext)
                               |
                    GNN runs on encrypted graph
                    detects money flow patterns
                               |
                        Encrypted result
                               |
              Each bank decrypts only their portion
                               |
                   "Transaction X is ILLICIT"
```

---

## 👣 Step-by-Step: How One Transaction Gets Checked

```
Step 1: Customer sends Bitcoin
        Wallet A --0.5 BTC--> Wallet B --1 BTC--> Wallet C
                                    (suspicious pattern?)

Step 2: Bank extracts 166 features per transaction
        [amount, time, num_inputs, num_outputs, fee_rate ...]

Step 3: Bank encrypts features using CKKS homomorphic encryption
        [0.5, 1623, 3, 1, 0.001] --> [Enc1, Enc2, Enc3, Enc4, Enc5]

Step 4: Encrypted features sent to AI server
        Server CANNOT read: sees only [Enc1, Enc2, Enc3, Enc4, Enc5]

Step 5: GNN runs on encrypted data
        - Looks at this transaction + its neighbours in the graph
        - Multiplies encrypted features by model weights
        - Aggregates encrypted neighbour info
        - Outputs: Enc(score)

Step 6: Encrypted score sent back to bank
        Bank decrypts: score = 0.94

Step 7: Threshold check
        0.94 > 0.5 threshold --> FLAG as ILLICIT --> Alert sent to compliance team
```

---

## 🏢 Who Would Build and Use This?

| Player | Role | Example |
|---|---|---|
| **Banks / Exchanges** | Data owners — encrypt and send transaction graphs | HDFC, SBI, Binance, CoinDCX |
| **RegTech Companies** | Build and host the AI server | ComplyAdvantage, Chainalysis |
| **Cloud Providers** | Host HE compute infrastructure | AWS, Azure, GCP |
| **Regulators** | Receive alerts and audit logs — but never raw data | FIU-India, FinCEN (US), EBA (EU) |
| **Blockchain Analytics** | Train and maintain the GNN model | Elliptic, TRM Labs |

---

## 💰 Real Companies Already Doing This

| Company | What They Do | Link |
|---|---|---|
| **Zama.ai** | Building FHE chips + Concrete ML library | https://zama.ai |
| **Chainalysis** | Crypto transaction monitoring | https://chainalysis.com |
| **Inpher** | Privacy-preserving ML for financial institutions | https://inpher.io |
| **Duality Technologies** | HE-based fraud detection for banks | https://dualitytech.com |
| **OpenMined** | Open-source tools — TenSEAL used in this project | https://openmined.org |
| **Elliptic** | GNN-based crypto AML — the dataset source | https://elliptic.co |

---

## 📋 Infrastructure Requirements

```
Infrastructure:
+-- HE Key Management System
|     Like AWS KMS but specifically for homomorphic encryption keys
|     Each bank holds its own private key — never shared
|
+-- High-Memory Servers
|     HE ciphertexts are 100x-1000x larger than plaintext data
|     A 1 KB transaction becomes a 1 MB ciphertext
|     Need 512 GB+ RAM servers for large graph inference
|
+-- GPU Cluster (for speed)
|     HE arithmetic is parallelizable on GPUs
|     10x-100x speedup over CPU-only inference
|
+-- Encrypted Audit Trail
      Every inference logged in encrypted form
      Regulators can audit without seeing raw data
```

---

## ✅ Compliance and Legal Coverage

| Regulation | Requirement | How This System Handles It |
|---|---|---|
| **GDPR (EU)** | Data minimization, right to privacy | Raw data never leaves client — fully compliant |
| **RBI Guidelines (India)** | Data localization | Each bank encrypts locally — data stays in India |
| **FATF AML Rules** | Suspicious transaction reporting | Alerts generated without exposing raw data |
| **DPDP Act 2023 (India)** | Personal data protection | No plaintext personal data transmitted |
| **SOC 2 / ISO 27001** | Security certification | Required for the AI server operator |

---

## 🔐 Security Model — What Is and Is Not Protected

```
+----------------------+---------------------+---------------------------+
|       Stage          |   Feature Values    |      Graph Topology       |
+----------------------+---------------------+---------------------------+
| 1. Plaintext GCN     |  [VISIBLE]          |  [VISIBLE]                |
| 2. Quantized GCN     |  [VISIBLE]          |  [VISIBLE]                |
| 3. HE Encrypted GCN  |  [HIDDEN - CKKS]    |  [WARNING - still visible]|
| 4. Full HE + SMPC    |  [HIDDEN]           |  [HIDDEN - future work]   |
+----------------------+---------------------+---------------------------+
```

> **Note:** In the current design, graph topology (which wallets connect to which)
> is visible to the server. Only the feature values (amounts, timestamps, etc.)
> are encrypted. Hiding topology requires Secure Multi-Party Computation (SMPC)
> — an active area of research.

---

## ⚡ Speed in Production

The biggest challenge is latency. Here is how production systems handle it:

### Technique 1: CKKS Batching

Pack thousands of node features into a single ciphertext.
One HE operation processes 8,192 nodes simultaneously.

```
Without batching:  8,192 separate HE ops  (very slow)
With batching:     1 HE op               (8,192x faster)
```

### Technique 2: Hybrid HE + Trusted Execution Environment (TEE)

```
Client                Trusted Enclave (TEE)        Server
  |                          |                        |
  |-- encrypted features --> |                        |
  |                    Decrypt inside                 |
  |                    Intel SGX chip  --> GNN runs   |
  |                          |          in plaintext  |
  |<-- encrypted result -----|                        |
```

Intel SGX / AMD SEV runs code in a hardware-protected vault.
Data is decrypted inside the chip — the server admin physically cannot access it.
Much faster than full HE, with hardware-level security guarantees.

### Realistic Speedup Breakdown

| Technique | Speedup |
|---|---|
| CKKS batching (8192 nodes per ciphertext) | 100 to 1000x |
| Quantization (8-bit weights) | 2 to 4x |
| Pruning (80% sparsity) | 3 to 5x |
| Fewer GNN layers (6 to 2) | 3x |
| GPU-accelerated HE | 10 to 100x |
| Hybrid HE + TEE | 1000x+ |

> Combined: Minutes per inference becomes Seconds per inference

---

## 🛣️ Realistic Deployment Timeline

```
Month 1-2   --> Plaintext GNN baseline on Elliptic dataset
                Target: F1 >= 0.85 on illicit class

Month 3     --> Quantize + polynomial activation approximation
                Measure accuracy drop vs plaintext baseline

Month 4-5   --> Encrypted inference with TenSEAL or Concrete ML
                Measure latency: how slow is it?

Month 6     --> Accuracy/latency benchmarks, research report or paper
-----------------------------------------------------------------------
Month 7-9   --> Multi-party setup (2-3 institutions with dummy data)
                Build key management + encrypted audit trail

Month 10-12 --> Compliance review + pilot with 1 real institution
                Legal agreements between participating banks

Year 2      --> Production deployment
                GPU-accelerated HE, real transaction volumes
                Regulatory sign-off from FIU or RBI
```

---

## 🎯 What This Project Proves (The Research Value)

This project is the foundational proof-of-concept that answers:

1. Can a GNN run on encrypted data? — Yes, using CKKS HE
2. How much accuracy is lost? — Measured by comparing plaintext vs encrypted F1
3. How slow is it? — Measured in wall-clock latency per inference
4. Is the privacy guarantee real? — Yes, CKKS has formal security proofs

> These 4 answers are exactly what a bank, regulator, or investor
> needs before committing to build a production system.

---

## 📚 Further Reading

| Topic | Resource |
|---|---|
| CKKS homomorphic encryption | https://eprint.iacr.org/2016/421.pdf |
| TenSEAL library | https://github.com/OpenMined/TenSEAL |
| Concrete ML (faster alternative) | https://github.com/zama-ai/concrete-ml |
| Privacy-preserving GNN for AML | https://arxiv.org/pdf/2411.02926 |
| FedGraphHE paper | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12768379/ |
| Elliptic Bitcoin Dataset | https://www.kaggle.com/datasets/ellipticco/elliptic-data-set |

---

*See [README.md](./README.md) for project setup, codebase structure, and development roadmap.*
