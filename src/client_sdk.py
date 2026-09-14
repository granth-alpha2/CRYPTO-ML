"""
client_sdk.py
=============
High-level Client SDK for Privacy-Preserving GNN Inference.

Implements SRS NFR-10:
    "Data owners shall be able to integrate via a client SDK without
     needing to understand HE internals."

The SDK wraps the full encrypt -> submit -> decrypt flow in 3 simple calls:

    client = CryptoMLClient(config_path="configs/config.yaml")
    client.setup()

    enc_result = client.infer(node_features, transaction_ids)
    alerts     = client.get_alerts()

That is all a data owner (bank, exchange) needs to write.

All CKKS key management, encryption, decryption, and alert generation
happen inside this class — completely hidden from the caller.
"""

import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import yaml

from src.utils import load_config, get_logger, print_metrics_table, Timer


logger = get_logger(__name__, log_file="results/client_sdk.log")


# -------------------------------------------------------------------
# CryptoMLClient
# -------------------------------------------------------------------

class CryptoMLClient:
    """
    One-stop client SDK for privacy-preserving GNN inference.

    Usage (minimal):
        client = CryptoMLClient()
        client.setup()                         # generate HE keys
        result = client.infer(features, ids)   # encrypt -> infer -> decrypt
        alerts = client.get_alerts()           # threshold + audit log

    Args:
        config_path   (str) : Path to configs/config.yaml
        institution_id(str) : Identifier for this data owner (used in audit log)
        server_mode   (bool): If False (default), runs inference locally
                              (research mode). Set True for remote server mode.
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        institution_id: str = "INSTITUTION_A",
        server_mode: bool = False,
    ):
        self.cfg            = load_config(config_path)
        self.institution_id = institution_id
        self.server_mode    = server_mode
        self._context       = None      # CKKS context (holds secret key)
        self._model         = None      # Trained plaintext model (for local mode)
        self._last_alerts   = []        # Most recent batch of alerts
        self._ready         = False

        logger.info(f"CryptoMLClient initialised | institution={institution_id} | "
                    f"server_mode={server_mode}")

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------

    def setup(self, model_path: Optional[str] = None):
        """
        Initialise the client:
          1. Generate CKKS encryption context + keys
          2. Load the trained GNN model (local mode only)

        Call this once before any infer() calls.

        Args:
            model_path: Override path to trained model weights (.pth).
                        Defaults to config paths.model_save.
        """
        logger.info("Setting up CKKS context...")
        self._setup_he_context()

        if not self.server_mode:
            logger.info("Loading trained model (local inference mode)...")
            self._load_model(model_path)

        self._ready = True
        logger.info("CryptoMLClient ready.")

    def _setup_he_context(self):
        """Create CKKS context. Gracefully falls back if TenSEAL missing."""
        try:
            from src.encrypt_infer import CKKSContext
            ctx_mgr        = CKKSContext(self.cfg)
            self._context  = ctx_mgr.create_context()
            self._ctx_mgr  = ctx_mgr
            logger.info("CKKS context ready (TenSEAL).")
        except ImportError:
            logger.warning("TenSEAL not installed — HE encryption disabled. "
                           "Running in plaintext-only mode.")
            self._context = None

    def _load_model(self, model_path: Optional[str] = None):
        """Load the trained GNN model from disk."""
        import torch
        from src.model import build_model

        path = model_path or self.cfg["paths"]["model_save"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Trained model not found: {path}\n"
                "Run python src/train.py first."
            )
        self._model = build_model(self.cfg)
        try:
            self._model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        except TypeError:
            self._model.load_state_dict(torch.load(path, map_location="cpu"))
        self._model.eval()
        logger.info(f"Model loaded from {path}")

    # ----------------------------------------------------------------
    # Core API — 3 methods the data owner actually calls
    # ----------------------------------------------------------------

    def infer(
        self,
        node_features: np.ndarray,
        transaction_ids: List[str],
        edge_index: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Full encrypt -> infer -> decrypt pipeline for a batch of transactions.

        Steps (hidden from caller):
            1. Encrypt node_features with CKKS public key
            2. Run GNN inference (encrypted if TenSEAL available, else plaintext)
            3. Decrypt the output
            4. Apply threshold -> generate alerts
            5. Write to audit log

        Args:
            node_features   : float32 array [N, 166] — one row per transaction
            transaction_ids : list of N transaction ID strings
            edge_index      : int array [2, E] of edges (topology). If None,
                              treats each node as isolated (no message passing).

        Returns:
            result dict:
                scores        : float array [N] — fraud probability per node
                predictions   : int array   [N] — 0=licit, 1=illicit
                alerts        : list of alert dicts for flagged transactions
                latency_ms    : total wall-clock time in milliseconds
                encrypted     : bool — True if HE was used
        """
        if not self._ready:
            raise RuntimeError("Call client.setup() before client.infer()")

        assert len(node_features) == len(transaction_ids), \
            "node_features and transaction_ids must have the same length"

        t0 = time.perf_counter()

        # -- Step 1: Encrypt ------------------------------------------
        if self._context is not None:
            scores, encrypted = self._encrypted_infer(node_features, edge_index)
        else:
            scores, encrypted = self._plaintext_infer(node_features, edge_index)

        # -- Step 2: Threshold -> Alerts -------------------------------
        from src.alert import ThresholdAlerter, AuditLogger
        alerter  = ThresholdAlerter(
            threshold      = self.cfg["eval"]["threshold"],
            institution_id = self.institution_id,
        )
        alerts = alerter.check_batch(transaction_ids, scores)

        # -- Step 3: Audit log ----------------------------------------
        log_path = os.path.join(self.cfg["paths"]["results"], "audit_log.jsonl")
        AuditLogger(log_path).log_batch(alerts)

        self._last_alerts = alerts
        latency_ms = (time.perf_counter() - t0) * 1000

        predictions = (scores > self.cfg["eval"]["threshold"]).astype(int)
        flagged     = [a for a in alerts if a["flagged"]]

        logger.info(
            f"infer() | nodes={len(transaction_ids)} | "
            f"flagged={len(flagged)} | latency={latency_ms:.1f}ms | "
            f"encrypted={encrypted}"
        )

        return {
            "scores"      : scores,
            "predictions" : predictions,
            "alerts"      : alerts,
            "latency_ms"  : round(latency_ms, 2),
            "encrypted"   : encrypted,
        }

    def get_alerts(self) -> List[dict]:
        """Return alerts from the most recent infer() call."""
        return [a for a in self._last_alerts if a["flagged"]]

    def generate_compliance_report(self) -> str:
        """
        Generate compliance report from the full audit log.

        Returns:
            Path to the generated HTML report.
        """
        from src.alert import ComplianceReporter
        log_path = os.path.join(self.cfg["paths"]["results"], "audit_log.jsonl")
        reporter = ComplianceReporter(self.cfg["paths"]["results"], log_path)
        summary  = reporter.generate()
        print_metrics_table(
            {k: v for k, v in summary.items() if k != "report_generated_utc"},
            title="Compliance Report Summary"
        )
        return os.path.join(self.cfg["paths"]["results"], "compliance_report.html")

    # ----------------------------------------------------------------
    # Internal inference implementations
    # ----------------------------------------------------------------

    def _encrypted_infer(
        self,
        node_features: np.ndarray,
        edge_index: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, bool]:
        """
        Run inference on CKKS-encrypted features.
        Returns (fraud_scores [N], encrypted=True).
        """
        try:
            from src.encrypt_infer import HEInferencePipeline
            pipeline = HEInferencePipeline(self.cfg)

            logger.info(f"Encrypting {len(node_features)} nodes...")
            with Timer("Encryption"):
                enc_feats = pipeline.encrypt(self._context, node_features)

            logger.info("Running encrypted GNN inference...")
            with Timer("HE Inference"):
                enc_logits = pipeline.run_encrypted_gcn(
                    enc_feats, self._model, edge_index
                )

            with Timer("Decryption"):
                logits = pipeline.decrypt(self._context, enc_logits)

            scores = self._logits_to_scores(logits)
            return scores, True

        except Exception as e:
            logger.warning(f"HE inference failed ({e}). Falling back to plaintext.")
            return self._plaintext_infer(node_features, edge_index)

    def _plaintext_infer(
        self,
        node_features: np.ndarray,
        edge_index: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, bool]:
        """
        Run plaintext GNN inference (fallback when TenSEAL unavailable).
        Returns (fraud_scores [N], encrypted=False).
        """
        import torch
        import torch.nn.functional as F

        x = torch.tensor(node_features, dtype=torch.float32)

        if edge_index is not None:
            ei = torch.tensor(edge_index, dtype=torch.long)
        else:
            # No edges — create empty edge_index
            ei = torch.zeros((2, 0), dtype=torch.long)

        with torch.no_grad():
            logits = self._model(x, ei)
            probs  = F.softmax(logits, dim=1).numpy()

        scores = probs[:, 1]   # probability of class 1 = illicit
        return scores, False

    @staticmethod
    def _logits_to_scores(logits: np.ndarray) -> np.ndarray:
        """Convert raw logits to fraud probability (softmax class 1)."""
        exp  = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs[:, 1]

    # ----------------------------------------------------------------
    # Key management helpers
    # ----------------------------------------------------------------

    def export_public_context(self, path: str):
        """
        Export the public CKKS context (no secret key) for the server.

        In production: send this to the inference server so it can
        operate on your encrypted data without ever having your private key.
        """
        if self._context is None:
            raise RuntimeError("No HE context. Call setup() first.")
        server_ctx = self._ctx_mgr.make_server_context(self._context)
        server_ctx.save(path)
        logger.info(f"Public context saved to {path} (no secret key included)")

    def __repr__(self) -> str:
        return (f"CryptoMLClient(institution={self.institution_id}, "
                f"ready={self._ready}, server_mode={self.server_mode})")


# -------------------------------------------------------------------
# Entry point — smoke test
# -------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("  CryptoMLClient — Smoke Test")
    print("=" * 60)

    client = CryptoMLClient(institution_id="HDFC_BANK")
    client.setup()
    print(f"\n  {client}\n")

    # Simulate 10 transactions with random features
    np.random.seed(0)
    fake_features = np.random.randn(10, 166).astype(np.float32)
    fake_ids      = [f"txn_{i:04d}" for i in range(10)]

    result = client.infer(fake_features, fake_ids)

    print(f"\n  Results:")
    print(f"    Nodes processed : {len(fake_ids)}")
    print(f"    Encrypted       : {result['encrypted']}")
    print(f"    Latency         : {result['latency_ms']} ms")
    print(f"    Flagged alerts  : {len(client.get_alerts())}")

    for alert in client.get_alerts():
        print(f"    !  {alert['transaction_id']} — score={alert['fraud_score']:.4f}")

    print(f"\n  Generating compliance report...")
    html_path = client.generate_compliance_report()
    print(f"  Report: {html_path}")
