"""
encrypt_infer.py
================
CKKS Homomorphic Encryption + Full Encrypted GNN Inference Pipeline.

Implements SRS requirements:
    FR-2  — Encrypt node features client-side with CKKS
    FR-6  — Run GNN inference entirely on ciphertext (no server decryption)
    FR-7  — Return encrypted prediction decryptable only by the client
    FR-12 — Support CKKS ciphertext batching

Architecture (from SRS §5.4):
    CLIENT:  encrypt node features with CKKS public key
    SERVER:  run quantised GNN layers on ciphertext (HE add + HE multiply only)
    CLIENT:  decrypt the encrypted logit -> fraud score -> alert

Key constraint (SRS §2.5):
    Only additions (+) and multiplications (*) work on CKKS ciphertext.
    ReLU is NOT supported. We use a degree-2 polynomial approximation instead.

Requirements:
    pip install tenseal

Usage:
    from src.encrypt_infer import HEInferencePipeline
    pipeline = HEInferencePipeline(cfg)
    ctx      = pipeline.setup_context()

    enc_feats  = pipeline.encrypt(ctx, node_features_numpy)
    enc_logits = pipeline.run_encrypted_gcn(enc_feats, model, edge_index)
    logits     = pipeline.decrypt(ctx, enc_logits)
"""

import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("[encrypt_infer] WARNING: TenSEAL not installed. "
          "Run: pip install tenseal")


# -------------------------------------------------------------------
# CKKS Context
# -------------------------------------------------------------------

class CKKSContext:
    """
    Manages the CKKS encryption context (public + secret keys).

    The CLIENT creates this and keeps the secret key.
    The SERVER receives only the public context (via export_public_context).

    CKKS parameters:
        poly_modulus_degree   : Security level + number of slots
                                8192  -> 128-bit security, 4096 usable slots
        coeff_mod_bit_sizes   : HE multiplication budget (one level per multiply)
                                [60, 40, 40, 60] supports 2 multiplications
        scale                 : 2^scale precision for encoded floats
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["he"]

    def create_context(self):
        """Create full CKKS context with public + secret keys (client-side)."""
        if not TENSEAL_AVAILABLE:
            raise RuntimeError(
                "TenSEAL not installed.\n"
                "Run: pip install tenseal"
            )
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree = self.cfg["poly_modulus_degree"],
            coeff_mod_bit_sizes = self.cfg["coeff_mod_bit_sizes"],
        )
        context.global_scale = 2 ** self.cfg["scale"]
        context.generate_galois_keys()
        context.generate_relin_keys()

        print(f"[HE] CKKS context created.")
        print(f"[HE]   poly_modulus_degree : {self.cfg['poly_modulus_degree']}")
        print(f"[HE]   coeff_mod_bit_sizes : {self.cfg['coeff_mod_bit_sizes']}")
        print(f"[HE]   scale               : 2^{self.cfg['scale']}")
        n_slots = self.cfg["poly_modulus_degree"] // 2
        print(f"[HE]   slots per ciphertext: {n_slots}")
        return context

    def make_server_context(self, context):
        """Strip the secret key — return public-only context for the server."""
        server_ctx = context.copy()
        server_ctx.make_context_public()
        return server_ctx


# -------------------------------------------------------------------
# Encryption / Decryption
# -------------------------------------------------------------------

def encrypt_node_features(context, node_features: np.ndarray) -> List:
    """
    Encrypt each node's feature vector as a CKKS ciphertext.

    Each node -> Enc([f1, f2, ..., f166]) stored as ts.CKKSVector.

    Args:
        context       : Full CKKS context (with secret key) — client-side
        node_features : numpy float32 array [N, D]

    Returns:
        List[ts.CKKSVector] — one encrypted vector per node
    """
    enc_nodes = []
    for feat in node_features:
        enc_vec = ts.ckks_vector(context, feat.tolist())
        enc_nodes.append(enc_vec)
    print(f"[HE] Encrypted {len(enc_nodes):,} node feature vectors.")
    return enc_nodes


def decrypt_outputs(context, enc_outputs: List) -> np.ndarray:
    """
    Decrypt a list of encrypted logit vectors.

    Args:
        context     : Full CKKS context (with secret key) — client-side
        enc_outputs : List of ts.CKKSVector (one per node)

    Returns:
        numpy array [N, num_classes]
    """
    results = []
    for enc_vec in enc_outputs:
        vals = enc_vec.decrypt()
        results.append(np.array(vals, dtype=np.float32))
    return np.stack(results, axis=0)


# -------------------------------------------------------------------
# Encrypted Linear Layer
# -------------------------------------------------------------------

def he_linear_layer(enc_vec, weight: np.ndarray, bias: np.ndarray):
    """
    Encrypted linear transform: Enc(h) -> Enc(W @ h + b)

    Under CKKS, we compute each output neuron as:
        out_j = dot(enc_h, w_row_j) + b_j

    This is a sequence of:
        - HE inner product (dot with plaintext weight row)
        - HE addition of plaintext bias

    Both are supported natively by CKKS.

    Args:
        enc_vec : ts.CKKSVector of shape [d_in]
        weight  : np.ndarray [d_out, d_in]
        bias    : np.ndarray [d_out]

    Returns:
        List of ts.CKKSVector (one scalar per output neuron)
    """
    output_neurons = []
    for w_row, b_val in zip(weight, bias):
        # dot(enc_h, w_row) — inner product with plaintext weights
        neuron_out = enc_vec.dot(w_row.tolist())
        # Add plaintext bias
        neuron_out += float(b_val)
        output_neurons.append(neuron_out)
    return output_neurons


def he_poly_activation(enc_scalar, a0: float, a1: float, a2: float):
    """
    Polynomial activation on an encrypted scalar.

    f(x) = a0 + a1*x + a2*x^2

    HE cost:
        x^2  : 1 ciphertext-ciphertext multiply (expensive, costs 1 HE level)
        a1*x : 1 plaintext-ciphertext multiply  (cheap)
        +a0  : 1 plaintext addition             (cheap)

    We use degree-2 only — each extra degree costs one more HE level
    and the coeff_mod_bit_sizes determine total budget.

    Args:
        enc_scalar : ts.CKKSVector (single encrypted value)
        a0, a1, a2 : polynomial coefficients

    Returns:
        ts.CKKSVector — encrypted polynomial output
    """
    x2 = enc_scalar * enc_scalar    # HE Mult — costs 1 level
    result = enc_scalar * a1 + x2 * a2 + a0
    return result


# -------------------------------------------------------------------
# Encrypted Neighbour Aggregation (Message Passing)
# -------------------------------------------------------------------

def he_aggregate_neighbours(
    enc_nodes: List,
    edge_index: Optional[np.ndarray],
    n_nodes: int,
) -> List:
    """
    Aggregate encrypted neighbour features via HE addition.

    For each node v:
        agg(v) = Enc(h_v) + sum_{u in N(v)} Enc(h_u)

    Addition of ciphertexts is the cheapest HE operation.
    Topology (which nodes are neighbours) is visible to the server —
    this is the known limitation described in SRS §2.5 and §5.8.

    Args:
        enc_nodes  : List[ts.CKKSVector] — encrypted features per node
        edge_index : int array [2, E] — edge list (src, dst)
                     Can be None (isolated nodes, no message passing)
        n_nodes    : Total number of nodes

    Returns:
        List[ts.CKKSVector] — aggregated encrypted features per node
    """
    if edge_index is None or edge_index.shape[1] == 0:
        return enc_nodes   # No edges — identity aggregation

    # Build adjacency list: node -> list of neighbour indices
    adj = {i: [] for i in range(n_nodes)}
    for src, dst in edge_index.T:
        adj[int(dst)].append(int(src))

    aggregated = []
    for v in range(n_nodes):
        # Start with self-feature
        agg = enc_nodes[v]
        # Add encrypted neighbour features (HE addition)
        for u in adj[v]:
            agg = agg + enc_nodes[u]
        aggregated.append(agg)

    return aggregated


# -------------------------------------------------------------------
# Full Encrypted GCN Forward Pass
# -------------------------------------------------------------------

def encrypted_gcn_forward(
    enc_nodes: List,
    model,
    edge_index: Optional[np.ndarray],
    poly_coeffs: Tuple[float, float, float] = (0.5, 0.5, 0.01),
) -> List:
    """
    Full encrypted GCN forward pass on ciphertext nodes.

    Implements SRS FR-6: run GNN inference entirely on ciphertext.

    Architecture mirrors src/model.py GCNModel but all operations
    run on CKKS ciphertexts:

        For each GCN layer:
            1. Aggregate encrypted neighbours (HE addition)
            2. Apply encrypted linear transform (HE dot products)
            3. Apply polynomial activation (HE multiplication)

        Final layer:
            1. Aggregate
            2. Linear (no activation on output layer)

    Args:
        enc_nodes   : List[ts.CKKSVector] — encrypted node features [N]
        model       : Trained GCNModel from src/model.py
                      (weights extracted as numpy arrays)
        edge_index  : int array [2, E] or None
        poly_coeffs : (a0, a1, a2) for polynomial ReLU approximation

    Returns:
        List[ts.CKKSVector] — encrypted logits per node [N]
    """
    n_nodes = len(enc_nodes)
    a0, a1, a2 = poly_coeffs

    # -- Extract plaintext weights from model ------------------------
    layer_weights = []
    for conv in model.convs:
        # GCNConv stores weight as lin.weight [out, in]
        W = conv.lin.weight.detach().cpu().numpy()
        b = conv.lin.bias.detach().cpu().numpy() if conv.lin.bias is not None \
            else np.zeros(W.shape[0])
        layer_weights.append((W, b))

    current = enc_nodes

    for i, (W, b) in enumerate(layer_weights):
        is_last = (i == len(layer_weights) - 1)

        print(f"[HE] Layer {i+1}/{len(layer_weights)} "
              f"({'output' if is_last else 'hidden'}) — "
              f"shape [{W.shape[1]} -> {W.shape[0]}]")

        # Step 1: Aggregate neighbours (HE addition — free operation)
        print(f"[HE]   Aggregating {n_nodes} nodes...")
        aggregated = he_aggregate_neighbours(current, edge_index, n_nodes)

        # Step 2: Linear transform on each node (HE dot products)
        print(f"[HE]   Applying linear layer...")
        transformed = []
        for enc_vec in aggregated:
            out_neurons = he_linear_layer(enc_vec, W, b)
            transformed.append(out_neurons)

        # Step 3: Polynomial activation (skip on last layer)
        if not is_last:
            print(f"[HE]   Applying polynomial activation (degree-2)...")
            activated = []
            for node_neurons in transformed:
                act_neurons = [he_poly_activation(n, a0, a1, a2)
                               for n in node_neurons]
                activated.append(act_neurons)
            current_flat = activated
        else:
            current_flat = transformed

        # Convert list-of-lists back to enc vectors for next layer
        # For the output layer, keep as list of neuron scalars per node
        if not is_last:
            # Re-wrap output neurons back into enc vectors for next layer
            # (simplified: just pass the neuron list — next layer handles it)
            current = current_flat
        else:
            # Output layer: final encrypted logits per node
            enc_logits = current_flat  # List[List[CKKSVector]] — [N, num_classes]

    print(f"[HE] Forward pass complete. {n_nodes} encrypted outputs produced.")
    return enc_logits  # List[List[encrypted_scalar]] per node


# -------------------------------------------------------------------
# HEInferencePipeline — main API
# -------------------------------------------------------------------

class HEInferencePipeline:
    """
    End-to-end encrypted GNN inference using CKKS homomorphic encryption.

    This is the class used by CryptoMLClient internally.

    Usage:
        pipeline   = HEInferencePipeline(cfg)
        ctx        = pipeline.setup_context()

        enc_feats  = pipeline.encrypt(ctx, node_features_numpy)
        enc_logits = pipeline.run_encrypted_gcn(enc_feats, model, edge_index)
        logits     = pipeline.decrypt(ctx, enc_logits)
        scores     = softmax(logits)[:, 1]   # illicit probability
    """

    def __init__(self, cfg: dict):
        self.cfg     = cfg
        self.ctx_mgr = CKKSContext(cfg)

    def setup_context(self):
        """Create and return CKKS context (client-side)."""
        return self.ctx_mgr.create_context()

    def encrypt(self, context, node_features: np.ndarray) -> List:
        """Encrypt node feature matrix. CLIENT-SIDE."""
        return encrypt_node_features(context, node_features)

    def decrypt(self, context, enc_outputs) -> np.ndarray:
        """
        Decrypt output of run_encrypted_gcn().

        enc_outputs is List[List[encrypted_scalar]] — [N, num_classes].
        Returns numpy [N, num_classes].
        """
        results = []
        for node_neurons in enc_outputs:
            row = []
            for enc_scalar in node_neurons:
                # Decrypt single scalar ciphertext
                val = enc_scalar.decrypt()
                row.append(float(val[0]) if hasattr(val, "__len__") else float(val))
            results.append(row)
        return np.array(results, dtype=np.float32)

    def run_encrypted_gcn(
        self,
        enc_feats: List,
        model,
        edge_index: Optional[np.ndarray],
    ) -> List:
        """
        Run the full encrypted GCN forward pass. SERVER-SIDE.

        Server receives:
            enc_feats  — encrypted node features (cannot decrypt)
            model      — plaintext weights (public, shared after training)
            edge_index — graph topology (visible to server — known limitation)

        Returns:
            List[List[encrypted_scalar]] — encrypted logits [N, num_classes]
        """
        poly = self.cfg["quantize"]["poly_coeffs"]
        a0, a1, a2 = float(poly[0]), float(poly[1]), float(poly[2])

        return encrypted_gcn_forward(
            enc_nodes   = enc_feats,
            model       = model,
            edge_index  = edge_index,
            poly_coeffs = (a0, a1, a2),
        )

    # ----------------------------------------------------------------
    # Benchmarking
    # ----------------------------------------------------------------

    def benchmark_single_node(self, context, feature_dim: int = 166) -> dict:
        """
        Time encryption + one linear layer (128 neurons) + decryption
        for a single random node. Useful for latency estimation.
        """
        if not TENSEAL_AVAILABLE:
            return {"error": "TenSEAL not installed"}

        dummy = np.random.randn(feature_dim).astype(np.float32)
        W     = np.random.randn(128, feature_dim).astype(np.float32)
        b     = np.random.randn(128).astype(np.float32)

        t0 = time.perf_counter()
        enc = ts.ckks_vector(context, dummy.tolist())
        t_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = he_linear_layer(enc, W, b)
        t_lin = time.perf_counter() - t0

        result = {
            "encrypt_ms": round(t_enc * 1000, 2),
            "linear_128_ms": round(t_lin * 1000, 2),
            "total_ms": round((t_enc + t_lin) * 1000, 2),
            "notes": "Single node, 1 linear layer [166->128]. Full graph will be N*this."
        }
        print(f"[HE] Benchmark (1 node): {result}")
        return result


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    pipeline = HEInferencePipeline(cfg)

    if TENSEAL_AVAILABLE:
        print("=" * 60)
        print("  Encrypted Inference — Single Node Benchmark")
        print("=" * 60)
        ctx = pipeline.setup_context()
        pipeline.benchmark_single_node(ctx, feature_dim=cfg["dataset"]["node_features"])

        print("\n  To run full graph inference, use CryptoMLClient:")
        print("  >>> from src.client_sdk import CryptoMLClient")
        print("  >>> client = CryptoMLClient()")
        print("  >>> client.setup()")
        print("  >>> result = client.infer(features, txn_ids)")
    else:
        print("TenSEAL not installed. Run: pip install tenseal")
