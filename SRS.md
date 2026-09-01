# Software Requirements Specification (SRS)

### Privacy-Preserving GNN Inference on Crypto Transaction Graphs

| | |
|---|---|
| **Document version** | 1.0 |
| **Status** | Draft |
| **Related documents** | [README.md](./README.md), Real World Implementation Guide |

---

## 1. Introduction

### 1.1 Purpose

This document specifies the functional and non-functional requirements, and
the full system architecture, for a system that runs Graph Neural Network
(GNN) inference directly on **encrypted** cryptocurrency transaction graphs.
It is intended for developers, reviewers, and any institution evaluating the
system before a pilot deployment.

### 1.2 Scope

The system:
- Accepts a transaction graph (nodes = transactions/wallets, edges = payment
  flows) from one or more data owners (e.g. banks, exchanges).
- Encrypts node/edge features client-side using homomorphic encryption
  (CKKS scheme).
- Runs a GNN model on the encrypted graph, on a server that never holds the
  decryption key.
- Returns an encrypted prediction (e.g. `licit` / `illicit`) that only the
  originating data owner can decrypt.
- Optionally aggregates encrypted signals across multiple data owners for
  cross-institution fraud detection, and produces compliance-ready alerts.

Out of scope for v1: full topology-hiding (Secure Multi-Party Computation),
on-chain settlement or wallet custody, and training under encryption.

### 1.3 Definitions, Acronyms, Abbreviations

| Term | Meaning |
|---|---|
| GNN | Graph Neural Network |
| HE | Homomorphic Encryption |
| CKKS | Homomorphic encryption scheme supporting approximate arithmetic on encrypted real numbers |
| AML | Anti-Money Laundering |
| FIU | Financial Intelligence Unit (India) |
| TEE | Trusted Execution Environment (e.g. Intel SGX) |
| SMPC | Secure Multi-Party Computation |
| KMS | Key Management System |
| SRS | Software Requirements Specification |

### 1.4 References

- Abadi & Andersen, *Learning to Protect Communications with Adversarial Neural Cryptography* — https://arxiv.org/abs/1610.06918
- *Privacy-Preserving Graph-Based ML with FHE for Collaborative AML* — https://arxiv.org/pdf/2411.02926
- *FedGraphHE: Privacy-Preserving Federated GNN with Dynamic HE* — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12768379/
- *DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning* — https://arxiv.org/pdf/2507.05649
- *EDLaaS: Fully Homomorphic Encryption Over Neural Network Graphs* — https://arxiv.org/pdf/2110.13638
- Elliptic Dataset — https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- TenSEAL — https://github.com/OpenMined/TenSEAL
- Concrete ML — https://github.com/zama-ai/concrete-ml

### 1.5 Overview

Section 2 gives the overall product context. Section 3 lists functional
requirements. Section 4 covers external interfaces. **Section 5 is the full
system architecture** — layers, components, data flow, and a step-by-step
sequence for a single transaction. Section 6 covers non-functional
requirements. Section 7 covers system models and data design. Section 8
covers compliance. Section 9 is the appendix.

---

## 2. Overall Description

### 2.1 Product Perspective

The system is a new, standalone privacy layer that sits between data owners
(banks/exchanges) and a shared inference service. It does not replace an
institution's existing transaction-monitoring stack; it replaces the *data
sharing* step in cross-institution fraud detection with an encrypted
computation step.

### 2.2 Product Functions (summary)

1. Client-side feature extraction and encryption
2. Encrypted transmission of graph data to a shared inference server
3. Plaintext GNN training (offline, on historical/labeled data)
4. Model quantization and polynomial activation approximation for HE-compatibility
5. Encrypted GNN inference on the server
6. Encrypted result return and client-side decryption
7. Optional multi-party aggregation across data owners
8. Alerting, encrypted audit logging, and compliance reporting

### 2.3 User Classes and Characteristics

| User class | Description | Technical level |
|---|---|---|
| Data Owner (Bank/Exchange) | Owns raw transaction data, holds its own private key, sends encrypted graphs, receives encrypted results | Integrates via SDK/API; not required to understand HE internals |
| ML/Platform Team | Trains and maintains the GNN model, manages quantization and the encrypted-inference pipeline | Data science / ML engineering |
| Compliance Officer | Reviews alerts and audit logs | Non-technical; uses a dashboard |
| Regulator (FIU/FinCEN/EBA) | Receives compliance reports and can audit encrypted logs | Non-technical |
| Platform/Infra Operator | Runs and scales the secure compute cluster; never has access to any private key | DevOps / infrastructure |

### 2.4 Operating Environment

- Client SDK: Python 3.10+, runs on-premises at each data owner (data never leaves this boundary unencrypted).
- Server: Linux-based cloud or on-prem cluster, GPU-accelerated, high-memory nodes (ciphertexts are 100–1000× larger than plaintext).
- Network: TLS-encrypted transport between clients and server, in addition to the HE layer itself.

### 2.5 Design and Implementation Constraints

- Only additions and multiplications are natively supported under CKKS; all nonlinear activations must be replaced with polynomial approximations.
- Model training cannot happen under encryption; training is plaintext-only, and only inference runs on ciphertext.
- Graph topology (which nodes/wallets connect to which) is visible to the server in v1; only feature values are encrypted.

### 2.6 Assumptions and Dependencies

- Each data owner can reliably generate and safeguard its own HE key pair.
- The Elliptic dataset (or equivalent labeled transaction graph) is available for offline model training.
- TenSEAL or Concrete ML is stable enough for the target model size and batching strategy.

---

## 3. System Features / Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | The system shall extract a fixed-length feature vector (e.g. 166 features) per transaction node from raw transaction data. | High |
| FR-2 | The system shall encrypt node/edge features client-side using CKKS before any data leaves the data owner's environment. | High |
| FR-3 | The system shall transmit only ciphertext (plus non-sensitive graph topology) to the inference server over a TLS-secured channel. | High |
| FR-4 | The system shall support offline, plaintext training of a GCN or GIN model on labeled historical data. | High |
| FR-5 | The system shall quantize the trained model and replace nonlinear activations (e.g. ReLU) with a low-degree polynomial approximation. | High |
| FR-6 | The system shall run GNN inference (aggregation + linear layers + approximated activation) entirely on ciphertext, without decrypting on the server at any point. | High |
| FR-7 | The system shall return an encrypted prediction to the originating data owner, decryptable only with that owner's private key. | High |
| FR-8 | The system shall support combining encrypted signals from multiple data owners into a single encrypted graph for cross-institution inference, without any owner seeing another's raw data. | Medium |
| FR-9 | The system shall apply a configurable decision threshold to the decrypted score and generate an alert when a transaction is flagged. | Medium |
| FR-10 | The system shall maintain an encrypted, append-only audit log of inference requests and outcomes for regulator review. | Medium |
| FR-11 | The system shall provide a key management component so each data owner generates, stores, and rotates its own private key, with no key ever transmitted to the server. | High |
| FR-12 | The system shall support CKKS ciphertext batching (packing many nodes' features into one ciphertext) to reduce the number of HE operations. | Medium |

---

## 4. External Interface Requirements

### 4.1 User Interfaces
- Compliance dashboard: view alerts, decrypted only for the reviewing institution's own data.
- Admin console: model version, quantization config, threshold configuration.

### 4.2 Hardware Interfaces
- GPU-accelerated compute nodes for the inference server (HE arithmetic is parallelizable).
- High-memory nodes (512 GB+ RAM recommended) to hold large ciphertext batches.
- Optional Intel SGX / AMD SEV–capable hardware if the hybrid HE+TEE mode (Section 5.6) is enabled.

### 4.3 Software Interfaces
- PyTorch + PyTorch Geometric for model definition and plaintext training.
- TenSEAL (CKKS via Microsoft SEAL) or Concrete ML for the encrypted-inference runtime.
- REST/gRPC API between client SDK and inference server.

### 4.4 Communication Interfaces
- TLS 1.2+ for all network transport, in addition to the HE ciphertext layer itself (defense in depth).
- Encrypted audit log export interface for regulators (read-only, append-only).

---

## 5. System Architecture

### 5.1 Architecture Layers

```
+--------------------------------------------------------------------+
| LAYER 1 — Data Owner / Client Layer                                |
|   Banks, exchanges. Holds raw data + private HE key. Encrypts      |
|   locally. Nothing unencrypted leaves this layer.                  |
+--------------------------------------------------------------------+
| LAYER 2 — Key Management Layer                                     |
|   Each data owner's own key store. Server never has a key.         |
+--------------------------------------------------------------------+
| LAYER 3 — Secure Compute / Inference Layer                         |
|   GNN runs on ciphertext only. GPU + high-memory cluster.          |
+--------------------------------------------------------------------+
| LAYER 4 — Alerting & Compliance Layer                               |
|   Threshold checks, encrypted audit trail, compliance reports.     |
+--------------------------------------------------------------------+
| LAYER 5 — Regulator / Audit Layer                                   |
|   Reads encrypted logs and compliance reports. No raw data access. |
+--------------------------------------------------------------------+
```

### 5.2 High-Level Component Diagram

```
+======================================================================+
|                    FULL SYSTEM ARCHITECTURE                          |
+======================================================================+
|                                                                      |
|  [DATA OWNERS]              [SECURE COMPUTE]         [REGULATORS]    |
|                                                                      |
|  Bank A --encrypt--+                          +-- Encrypted audit    |
|  Bank B --encrypt--+--> Ingestion API          |   logs               |
|  Bank C --encrypt--+   (TLS + CKKS ciphertext) |                     |
|                          |                     +-- Compliance         |
|                          v                          reports          |
|                   GNN Inference Engine                                |
|                (runs on ciphertext only)                              |
|                          |                                            |
|                          v                                            |
|                   Alert Service                                       |
|                   "Flag txn #X" (threshold check on decrypted score)  |
|                          |                                            |
|                          v                                            |
|               Law Enforcement / FIU notification                     |
+======================================================================+
```

### 5.3 Component Table

| Component | Responsibility | Technology |
|---|---|---|
| Client SDK | Feature extraction, CKKS encryption, decryption of results | Python, TenSEAL / Concrete ML |
| Key Management System | Per-owner key generation, storage, rotation | HSM-backed or cloud KMS equivalent, HE-aware |
| Ingestion API | Receives encrypted graphs over TLS | REST/gRPC service |
| GNN Inference Engine | Runs quantized, HE-compatible GNN on ciphertext | PyTorch-trained model compiled to TenSEAL/Concrete ML |
| Alert Service | Applies threshold to decrypted score, raises alerts | Python microservice |
| Encrypted Audit Log Store | Append-only log of requests/outcomes | Encrypted database |
| Compliance Reporting Module | Generates regulator-facing reports | Python + report templates |

### 5.4 Data Flow — Single Transaction (Sequence)

```
1. Client extracts features
   [amount, time, num_inputs, num_outputs, fee_rate, ...]  (166-dim)

2. Client encrypts features (CKKS)
   [0.5, 1623, 3, 1, 0.001] --> [Enc1, Enc2, Enc3, Enc4, Enc5]

3. Encrypted features sent to Ingestion API
   Server receives ciphertext only; cannot read values.

4. GNN Inference Engine processes the encrypted graph
   - Aggregates encrypted neighbour features (HE addition)
   - Applies encrypted linear layer (HE multiplication by weights)
   - Applies polynomial-approximated activation
   - Produces Enc(score)

5. Enc(score) returned to the originating client

6. Client decrypts locally
   score = 0.94

7. Threshold check (client or alert service, per deployment)
   score > 0.5  -->  flagged as illicit  -->  alert raised
```

### 5.5 Multi-Party Aggregation (Cross-Institution Mode)

```
  Bank A --encrypt--+
  Bank B --encrypt--+--> Central GNN Inference Engine
  Bank C --encrypt--+    (sees only ciphertext from each party)
                          |
                    Encrypted per-owner result
                          |
        Each bank decrypts ONLY its own portion of the result
```

Each data owner's ciphertext is processed against the shared graph, but
results are partitioned and encrypted per-owner so no institution can
decrypt another institution's outcome.

### 5.6 Optional Speed-Path: Hybrid HE + TEE

```
Client                Trusted Enclave (TEE)         Server
  |                          |                         |
  |-- encrypted features --> |                         |
  |                    Decrypt inside hardware-         |
  |                    protected enclave (SGX/SEV)      |
  |                          |--> GNN runs in plaintext |
  |                              inside the enclave      |
  |<-- encrypted result -----|                         |
```

This path trades a hardware trust assumption (the enclave vendor and its
attestation guarantees) for substantially lower latency than pure HE. It is
optional and should be evaluated against each deployment's threat model.

### 5.7 Deployment / Infrastructure Architecture

```
Infrastructure
+-- HE Key Management System
|     Each data owner holds its own private key — never shared
+-- High-Memory Inference Nodes
|     Ciphertexts are 100x-1000x larger than plaintext
|     512 GB+ RAM recommended for large-graph inference
+-- GPU Cluster
|     HE arithmetic is parallelizable; 10x-100x speedup over CPU-only
+-- Encrypted Audit Trail Store
      Every inference logged in encrypted form for regulator review
```

### 5.8 Security Architecture — What Is and Is Not Protected

| Stage | Feature values | Graph topology |
|---|---|---|
| 1. Plaintext GNN (dev/testing only) | Visible | Visible |
| 2. Quantized GNN (dev/testing only) | Visible | Visible |
| 3. HE-encrypted GNN (v1 target) | Hidden (CKKS) | Still visible — known limitation |
| 4. Full HE + SMPC (future work) | Hidden | Hidden |

---

## 6. Non-Functional Requirements

### 6.1 Performance
- NFR-1: Baseline (unoptimized) encrypted inference latency shall be measured and reported against the plaintext baseline; expect 10×–10,000× overhead depending on model size and batching.
- NFR-2: CKKS ciphertext batching shall be used to pack up to ~8,192 nodes per ciphertext, targeting up to a 100–1000× reduction in HE operation count versus unbatched inference.
- NFR-3: Quantization (e.g. 8-bit weights) and pruning (e.g. 80% sparsity) shall be evaluated as additional latency-reduction techniques, each targeting a 2–5× speedup.

### 6.2 Security
- NFR-4: No private key shall ever be transmitted to or stored on the inference server.
- NFR-5: All network transport shall use TLS 1.2+ in addition to the HE ciphertext layer.
- NFR-6: Audit logs shall be encrypted at rest and append-only.

### 6.3 Reliability / Availability
- NFR-7: The inference service shall degrade gracefully (queue/retry) rather than drop requests under load spikes.

### 6.4 Scalability
- NFR-8: The architecture shall support adding data owners (banks) without requiring existing owners to re-share any historical raw data.

### 6.5 Maintainability
- NFR-9: The plaintext model, quantized model, and encrypted-inference pipeline shall be versioned independently so accuracy loss can be attributed to the correct pipeline stage.

### 6.6 Usability
- NFR-10: Data owners shall be able to integrate via a client SDK without needing to understand HE internals.

---

## 7. System Models

### 7.1 Data Model (per transaction node)

| Field | Type | Notes |
|---|---|---|
| transaction_id | string | Unique identifier |
| features[166] | float vector | Amount, time, in/out counts, fee rate, etc. |
| label | enum {licit, illicit, unknown} | Present only in training data |
| edges | list of (from_id, to_id) | Payment flow; topology visible to server in v1 |

### 7.2 Use Cases (summary)

| Use case | Primary actor | Goal |
|---|---|---|
| Encrypt and submit transaction graph | Data Owner | Get a fraud score without exposing raw data |
| Train baseline model | ML Team | Produce a plaintext GNN to later convert for HE |
| Run encrypted inference | Inference Engine | Score a transaction without ever decrypting it |
| Review alert | Compliance Officer | Decide whether to escalate a flagged transaction |
| Audit encrypted logs | Regulator | Confirm compliant processing without seeing raw data |

---

## 8. Compliance Mapping

| Regulation | Requirement | System behavior |
|---|---|---|
| GDPR (EU) | Data minimization, privacy by design | Raw data never leaves the data owner's environment |
| RBI Guidelines (India) | Data localization | Encryption happens locally; data stays in-country |
| FATF AML Rules | Suspicious transaction reporting | Alerts generated without exposing raw data to the server |
| DPDP Act 2023 (India) | Personal data protection | No plaintext personal data is ever transmitted |
| SOC 2 / ISO 27001 | Security certification | Required of the inference service operator |

---

## 9. Appendix

### 9.1 Known Limitations (carried from README)
- Encrypted inference is significantly slower than plaintext.
- Training under full homomorphic encryption is not practical today.
- Graph topology is visible to the server in the v1 architecture.
- Polynomial approximation of nonlinear activations introduces some accuracy loss.

### 9.2 Future Work
- Secure Multi-Party Computation (SMPC) or oblivious graph protocols to hide topology as well as features.
- Formal evaluation of the hybrid HE+TEE path against a defined threat model.
- Extending multi-party aggregation to support dynamic addition/removal of data owners without re-encrypting historical data.
