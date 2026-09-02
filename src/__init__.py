"""
CRYPTO ML — Privacy-Preserving GNN Inference on Crypto Transaction Graphs
=========================================================================
Package entry point.

Modules:
    preprocess    - Data loading and preprocessing (Elliptic dataset -> PyG graph)
    model         - GNN architecture definitions (GCN, GIN)
    train         - Plaintext training loop with metrics
    quantize      - Post-training quantization and polynomial activation approximation
    encrypt_infer - CKKS homomorphic encryption + encrypted GNN inference pipeline
    evaluate      - Accuracy / F1 / latency comparison across all pipeline stages
"""

__version__ = "0.1.0"
__author__  = "CRYPTO ML Project"
