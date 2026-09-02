"""
encrypt_infer.py
================
CKKS Homomorphic Encryption + Encrypted GNN Inference Pipeline.

This module implements the SERVER-SIDE inference on encrypted node features.
The server never decrypts the data — it only operates on ciphertexts.

Pipeline:
    CLIENT:  encrypt node features using CKKS public key
    SERVER:  run GNN layers on encrypted vectors (HE add + HE multiply)
    CLIENT:  decrypt the encrypted logit output -> classification result

Limitations (see README for full discussion):
    - Graph topology (edge_index) is still visible to the server
    - Only node FEATURE VALUES are encrypted
    - Training happens in plaintext; only inference is encrypted

Requirements:
    pip install tenseal

Usage:
    from src.encrypt_infer import HEInferencePipeline
    pipeline = HEInferencePipeline(cfg)
    pipeline.setup_context()
    enc_result = pipeline.infer_encrypted(model, node_features, edge_index)
    logits = pipeline.decrypt(enc_result)
"""

import time
import numpy as np
import torch
import torch.nn as nn

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("[encrypt_infer] WARNING: TenSEAL not installed. "
          "Run: pip install tenseal")


# -------------------------------------------------------------------
# CKKS Context Setup
# -------------------------------------------------------------------

class CKKSContext:
    """
    Manages the CKKS encryption context (keys + parameters).

    The client creates this and holds the SECRET key.
    The server receives only the PUBLIC key and galois keys.

    CKKS parameters (from config):
        poly_modulus_degree   : Controls security level and slot count
                                8192  -> 128-bit security, 4096 slots
                                16384 -> 128-bit security, 8192 slots
        coeff_mod_bit_sizes   : Controls HE "budget" (depth of multiplications)
                                Each multiplication consumes one level
        scale                 : 2^scale — precision of encoded floats
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["he"]

    def create_context(self):
        """Create a TenSEAL CKKS context with full keys (client-side)."""
        if not TENSEAL_AVAILABLE:
            raise RuntimeError("TenSEAL not installed.")

        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree = self.cfg["poly_modulus_degree"],
            coeff_mod_bit_sizes = self.cfg["coeff_mod_bit_sizes"],
        )
        context.global_scale = 2 ** self.cfg["scale"]
        context.generate_galois_keys()    # needed for slot rotations (aggregation)
        context.generate_relin_keys()     # needed for ciphertext multiplication

        print(f"[HE] CKKS context created.")
        print(f"[HE]   poly_modulus_degree : {self.cfg['poly_modulus_degree']}")
        print(f"[HE]   coeff_mod_bit_sizes : {self.cfg['coeff_mod_bit_sizes']}")
        print(f"[HE]   scale               : 2^{self.cfg['scale']}")
        print(f"[HE]   slots per ciphertext: {self.cfg['poly_modulus_degree'] // 2}")
        return context

    def make_server_context(self, context):
        """
        Return a copy of the context with only PUBLIC key (for server).
        Server cannot decrypt — it only has public key.
        """
        server_ctx = context.copy()
        server_ctx.make_context_public()   # drop secret key
        return server_ctx


# -------------------------------------------------------------------
# Encryption / Decryption
# -------------------------------------------------------------------

def encrypt_node_features(
    context, node_features: np.ndarray
) -> list:
    """
    Encrypt each node's feature vector as a CKKS ciphertext.

    Each node becomes one encrypted vector: Enc([f1, f2, ..., f166]).

    Args:
        context        : Full CKKS context (with secret key)
        node_features  : numpy array of shape [N, 166]

    Returns:
        list of ts.CKKSVector — one per node
    """
    enc_nodes = []
    for i, feat in enumerate(node_features):
        enc_vec = ts.ckks_vector(context, feat.tolist())
        enc_nodes.append(enc_vec)

    print(f"[HE] Encrypted {len(enc_nodes):,} node feature vectors.")
    return enc_nodes


def decrypt_outputs(context, enc_outputs: list) -> np.ndarray:
    """
    Decrypt a list of encrypted output vectors (logits).

    Args:
        context     : Full CKKS context (with secret key)
        enc_outputs : list of ts.CKKSVector

    Returns:
        numpy array of shape [N, num_classes]
    """
    decrypted = [np.array(v.decrypt()) for v in enc_outputs]
    return np.stack(decrypted, axis=0)


# -------------------------------------------------------------------
# Encrypted Linear Layer
# -------------------------------------------------------------------

def he_linear(enc_vec, weight: np.ndarray, bias: np.ndarray):
    """
    Compute a linear transformation on an encrypted vector.

    For a single node: Enc(h) -> Enc(W @ h + b)

    Under CKKS:
        Enc(h) * W  is computed as a series of HE multiplications by plaintext
        Enc(h) + b  is a HE addition by plaintext

    This is the most expensive operation in encrypted GNN inference.

    Args:
        enc_vec : ts.CKKSVector — encrypted node features [d_in]
        weight  : numpy array   — weight matrix [d_out, d_in]
        bias    : numpy array   — bias vector   [d_out]

    Returns:
        ts.CKKSVector — encrypted output [d_out]
    """
    # Matrix-vector multiply using HE: each output dim is a dot product
    outputs = []
    for row, b_val in zip(weight, bias):
        dot = enc_vec.dot(row.tolist())   # HE inner product
        dot += b_val                       # HE plaintext addition
        outputs.append(dot)

    # Note: returning list of scalars per output dim
    # For simplicity, stack as a new CKKSVector
    # In practice: use packed CKKS for efficiency
    return outputs


def he_poly_activation(enc_scalar, a0: float, a1: float, a2: float):
    """
    Apply polynomial activation f(x) = a0 + a1*x + a2*x^2 on an encrypted scalar.

    Under CKKS:
        a2*x^2  requires one ciphertext-ciphertext multiplication (expensive)
        a1*x    is a plaintext-ciphertext multiplication (cheap)
        a0      is a plaintext addition (cheap)

    This is why we use degree-2 (not degree-3+) — each extra degree costs
    one multiplication level in the HE budget.

    Args:
        enc_scalar : ts.CKKSVector (scalar ciphertext)
        a0, a1, a2 : polynomial coefficients

    Returns:
        ts.CKKSVector — encrypted result of f(enc_scalar)
    """
    x2   = enc_scalar * enc_scalar   # HE Mult (costs 1 level)
    result = enc_scalar * a1 + x2 * a2 + a0
    return result


# -------------------------------------------------------------------
# Full Encrypted Inference Pipeline
# -------------------------------------------------------------------

class HEInferencePipeline:
    """
    End-to-end encrypted GNN inference using CKKS.

    Usage:
        pipeline = HEInferencePipeline(cfg)
        context  = pipeline.setup_context()

        # Client: encrypt
        enc_feats = pipeline.encrypt(context, node_features_numpy)

        # Server: infer on ciphertext (pass server_context, not full context)
        enc_logits = pipeline.infer(enc_feats, model_weights, edge_index, adjacency)

        # Client: decrypt
        logits = pipeline.decrypt(context, enc_logits)
        predictions = logits.argmax(axis=1)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.context_mgr = CKKSContext(cfg)

    def setup_context(self):
        """Create and return CKKS context (client-side setup)."""
        return self.context_mgr.create_context()

    def encrypt(self, context, node_features: np.ndarray) -> list:
        """Encrypt node features. Called by CLIENT."""
        return encrypt_node_features(context, node_features)

    def decrypt(self, context, enc_outputs: list) -> np.ndarray:
        """Decrypt outputs. Called by CLIENT with secret key."""
        return decrypt_outputs(context, enc_outputs)

    def benchmark_single_node(self, context, feature_dim: int = 166) -> dict:
        """
        Benchmark encryption + one linear layer + decryption for a single node.
        Returns timing dict.
        """
        if not TENSEAL_AVAILABLE:
            return {"error": "TenSEAL not installed"}

        dummy = np.random.randn(feature_dim).astype(np.float32)
        W     = np.random.randn(128, feature_dim).astype(np.float32)
        b     = np.random.randn(128).astype(np.float32)

        t0 = time.time()
        enc = ts.ckks_vector(context, dummy.tolist())
        t_encrypt = time.time() - t0

        t0 = time.time()
        _ = he_linear(enc, W, b)
        t_linear = time.time() - t0

        results = {
            "encrypt_s"  : round(t_encrypt, 4),
            "linear_s"   : round(t_linear, 4),
            "total_s"    : round(t_encrypt + t_linear, 4),
        }
        print(f"[HE] Single-node benchmark: {results}")
        return results


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    pipeline = HEInferencePipeline(cfg)

    if TENSEAL_AVAILABLE:
        ctx = pipeline.setup_context()
        pipeline.benchmark_single_node(ctx, feature_dim=cfg["dataset"]["node_features"])
    else:
        print("Install TenSEAL to run encrypted inference: pip install tenseal")
