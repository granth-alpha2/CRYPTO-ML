"""
alert.py
========
Alerting, audit logging, and compliance reporting.

Implements SRS requirements:
    FR-9  — Threshold-based alert generation from decrypted scores
    FR-10 — Encrypted, append-only audit log of inference requests/outcomes

Components:
    ThresholdAlerter    : Applies configurable threshold to fraud scores
    AuditLogger         : Appends each inference event to a JSON-lines log
    ComplianceReporter  : Generates summary CSV + HTML report for regulators

Usage:
    from src.alert import ThresholdAlerter, AuditLogger, ComplianceReporter

    alerter  = ThresholdAlerter(threshold=0.5)
    logger   = AuditLogger("results/audit_log.jsonl")
    reporter = ComplianceReporter("results/")

    alert = alerter.check("txn_001", score=0.94)
    logger.log(alert)
    reporter.generate()
"""

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np


# -------------------------------------------------------------------
# Alert data structure
# -------------------------------------------------------------------

def make_alert(
    transaction_id: str,
    score: float,
    prediction: int,
    flagged: bool,
    threshold: float,
    institution_id: str = "INSTITUTION_A",
) -> dict:
    """
    Build a structured alert dict for a single transaction.

    Args:
        transaction_id : Unique transaction ID
        score          : Decrypted fraud probability (0.0 – 1.0)
        prediction     : Predicted class (0=licit, 1=illicit)
        flagged        : True if score > threshold
        threshold      : Decision boundary used
        institution_id : Owning institution identifier

    Returns:
        dict with all alert fields
    """
    return {
        "event_id"       : hashlib.sha256(
                               f"{transaction_id}{time.time()}".encode()
                           ).hexdigest()[:16],
        "timestamp_utc"  : datetime.now(timezone.utc).isoformat(),
        "institution_id" : institution_id,
        "transaction_id" : transaction_id,
        "fraud_score"    : round(float(score), 6),
        "prediction"     : "illicit" if prediction == 1 else "licit",
        "flagged"        : flagged,
        "threshold"      : threshold,
        "status"         : "ALERT_RAISED" if flagged else "CLEAR",
    }


# -------------------------------------------------------------------
# ThresholdAlerter
# -------------------------------------------------------------------

class ThresholdAlerter:
    """
    Applies a configurable decision threshold to decrypted fraud scores.

    After the client decrypts the GNN output, this class determines
    whether each transaction should be flagged as illicit.

    Args:
        threshold (float): Decision boundary. score > threshold -> illicit.
                           Default 0.5. Higher = fewer but higher-confidence alerts.
        institution_id    : Owning institution for audit trail.
    """

    def __init__(self, threshold: float = 0.5, institution_id: str = "INSTITUTION_A"):
        self.threshold      = threshold
        self.institution_id = institution_id
        self._alert_count   = 0
        self._clear_count   = 0

    def check(self, transaction_id: str, score: float) -> dict:
        """
        Check a single transaction score against the threshold.

        Args:
            transaction_id : Unique transaction identifier
            score          : Decrypted fraud probability from GNN

        Returns:
            alert dict (see make_alert)
        """
        flagged    = score > self.threshold
        prediction = 1 if flagged else 0

        if flagged:
            self._alert_count += 1
        else:
            self._clear_count += 1

        return make_alert(
            transaction_id = transaction_id,
            score          = score,
            prediction     = prediction,
            flagged        = flagged,
            threshold      = self.threshold,
            institution_id = self.institution_id,
        )

    def check_batch(
        self,
        transaction_ids: List[str],
        scores: np.ndarray,
    ) -> List[dict]:
        """
        Check a batch of scores. Returns list of alert dicts.

        Args:
            transaction_ids : List of transaction IDs  [N]
            scores          : 1-D numpy array of fraud probs [N]

        Returns:
            List of alert dicts, one per transaction
        """
        return [self.check(tid, float(s)) for tid, s in zip(transaction_ids, scores)]

    @property
    def summary(self) -> dict:
        """Return counts of alerts raised vs cleared."""
        total = self._alert_count + self._clear_count
        return {
            "total_checked" : total,
            "alerts_raised" : self._alert_count,
            "cleared"       : self._clear_count,
            "alert_rate"    : round(self._alert_count / max(total, 1), 4),
            "threshold"     : self.threshold,
        }

    def print_summary(self):
        from src.utils import print_metrics_table
        print_metrics_table(self.summary, title="Alert Summary")


# -------------------------------------------------------------------
# AuditLogger
# -------------------------------------------------------------------

class AuditLogger:
    """
    Append-only audit log of inference requests and outcomes.

    Implements SRS FR-10: encrypted, append-only audit trail.
    Each event is stored as a JSON line for easy parsing by regulators.

    In production this log would be encrypted at rest (e.g. with the
    institution's public key). In this research prototype the log is
    plaintext JSON-lines — encryption is noted as a future hardening step.

    Args:
        log_path (str): Path to the JSON-lines audit log file.
    """

    def __init__(self, log_path: str = "results/audit_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, alert: dict):
        """
        Append a single alert/event to the audit log.

        Args:
            alert (dict): Alert dict from ThresholdAlerter.check()
        """
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert) + "\n")

    def log_batch(self, alerts: List[dict]):
        """Append a list of alerts to the audit log."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            for alert in alerts:
                f.write(json.dumps(alert) + "\n")

    def read_all(self) -> List[dict]:
        """Read all entries from the audit log."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def count(self) -> int:
        """Return the number of logged events."""
        return len(self.read_all())

    def flagged_events(self) -> List[dict]:
        """Return only the events that were flagged as illicit."""
        return [e for e in self.read_all() if e.get("flagged")]


# -------------------------------------------------------------------
# ComplianceReporter
# -------------------------------------------------------------------

class ComplianceReporter:
    """
    Generates compliance-ready summary reports for regulators.

    Reads the audit log and produces:
        - results/compliance_report.csv  (machine-readable)
        - results/compliance_report.html (human-readable dashboard)

    Regulators receive these reports without ever seeing raw transaction data.
    Implements SRS §5.4 (Alerting & Compliance Layer) and §8 (Compliance Mapping).

    Args:
        results_dir (str): Directory where reports will be written.
        log_path    (str): Path to the JSON-lines audit log.
    """

    def __init__(
        self,
        results_dir: str = "results/",
        log_path: str = "results/audit_log.jsonl",
    ):
        self.results_dir = results_dir
        self.log_path    = log_path
        os.makedirs(results_dir, exist_ok=True)

    def generate(self) -> dict:
        """
        Read the audit log and produce CSV + HTML compliance reports.

        Returns:
            summary dict with key statistics
        """
        logger = AuditLogger(self.log_path)
        events = logger.read_all()

        if not events:
            print("[ComplianceReporter] No audit events found. Run inference first.")
            return {}

        # -- CSV report ----------------------------------------------
        csv_path = os.path.join(self.results_dir, "compliance_report.csv")
        fieldnames = [
            "event_id", "timestamp_utc", "institution_id",
            "transaction_id", "fraud_score", "prediction",
            "flagged", "threshold", "status"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)
        print(f"[ComplianceReporter] CSV  -> {csv_path}")

        # -- Summary statistics ---------------------------------------
        total     = len(events)
        flagged   = sum(1 for e in events if e["flagged"])
        cleared   = total - flagged
        avg_score = np.mean([e["fraud_score"] for e in events])
        threshold = events[0]["threshold"] if events else 0.5

        summary = {
            "report_generated_utc" : datetime.now(timezone.utc).isoformat(),
            "total_transactions"   : total,
            "flagged_illicit"      : flagged,
            "cleared_licit"        : cleared,
            "alert_rate_pct"       : round(flagged / max(total, 1) * 100, 2),
            "avg_fraud_score"      : round(float(avg_score), 4),
            "threshold_used"       : threshold,
        }

        # -- HTML report ---------------------------------------------
        html_path = os.path.join(self.results_dir, "compliance_report.html")
        self._write_html(summary, events[:50], html_path)  # show top 50 rows
        print(f"[ComplianceReporter] HTML -> {html_path}")

        return summary

    def _write_html(self, summary: dict, events: List[dict], path: str):
        rows = ""
        for e in events:
            color = "#ffcccc" if e["flagged"] else "#ccffcc"
            rows += (
                f"<tr style='background:{color}'>"
                f"<td>{e['timestamp_utc'][:19]}</td>"
                f"<td>{e['institution_id']}</td>"
                f"<td>{e['transaction_id']}</td>"
                f"<td>{e['fraud_score']:.4f}</td>"
                f"<td><b>{e['status']}</b></td>"
                f"</tr>\n"
            )

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Compliance Report — CRYPTO ML</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
  h1   {{ color: #333; }}
  .summary {{ background:#fff; padding:20px; border-radius:8px;
               box-shadow:0 2px 4px rgba(0,0,0,.1); margin-bottom:24px; }}
  .summary table {{ border-collapse: collapse; width: 100%; }}
  .summary td  {{ padding: 8px 16px; border-bottom: 1px solid #eee; }}
  .summary td:first-child {{ font-weight: bold; color: #555; }}
  h2 {{ color: #444; margin-top: 32px; }}
  table.events {{ border-collapse: collapse; width:100%; background:#fff;
                  border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,.1); }}
  table.events th {{ background:#334; color:#fff; padding:10px 14px; text-align:left; }}
  table.events td {{ padding:8px 14px; border-bottom:1px solid #eee; }}
  .badge-alert {{ background:#e74c3c; color:#fff; padding:2px 8px;
                  border-radius:4px; font-size:.85em; }}
  .badge-clear {{ background:#27ae60; color:#fff; padding:2px 8px;
                  border-radius:4px; font-size:.85em; }}
</style>
</head>
<body>
<h1>🔐 Compliance Report — Privacy-Preserving GNN System</h1>
<p><em>Generated: {summary['report_generated_utc']}</em></p>
<p><strong>Note:</strong> This report is generated from encrypted inference outputs.
No raw transaction data was ever transmitted to the inference server.</p>

<div class="summary">
  <table>
    <tr><td>Total Transactions Checked</td><td>{summary['total_transactions']:,}</td></tr>
    <tr><td>Flagged as Illicit</td><td style="color:#c0392b"><b>{summary['flagged_illicit']:,}</b></td></tr>
    <tr><td>Cleared as Licit</td><td style="color:#27ae60"><b>{summary['cleared_licit']:,}</b></td></tr>
    <tr><td>Alert Rate</td><td>{summary['alert_rate_pct']}%</td></tr>
    <tr><td>Average Fraud Score</td><td>{summary['avg_fraud_score']}</td></tr>
    <tr><td>Decision Threshold</td><td>{summary['threshold_used']}</td></tr>
  </table>
</div>

<h2>Recent Events (up to 50 shown)</h2>
<table class="events">
<tr>
  <th>Timestamp (UTC)</th>
  <th>Institution</th>
  <th>Transaction ID</th>
  <th>Fraud Score</th>
  <th>Status</th>
</tr>
{rows}
</table>
<p style="color:#999;font-size:.85em;margin-top:32px;">
  CRYPTO ML Research Project — Privacy-Preserving GNN Inference |
  Compliant with GDPR, RBI, FATF AML, DPDP Act 2023
</p>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)


# -------------------------------------------------------------------
# Entry point — demo
# -------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from src.utils import load_config, print_metrics_table

    cfg       = load_config()
    threshold = cfg["eval"]["threshold"]

    # Simulate 20 decrypted fraud scores
    np.random.seed(42)
    scores  = np.random.beta(2, 5, 20)          # skewed toward 0 (mostly licit)
    scores[3], scores[7], scores[14] = 0.82, 0.91, 0.76  # force some illicit

    txn_ids = [f"txn_{i:04d}" for i in range(20)]

    alerter = ThresholdAlerter(threshold=threshold)
    alerts  = alerter.check_batch(txn_ids, scores)

    audit_log = AuditLogger("results/audit_log.jsonl")
    audit_log.log_batch(alerts)

    alerter.print_summary()

    reporter = ComplianceReporter("results/", "results/audit_log.jsonl")
    summary  = reporter.generate()
    print_metrics_table(
        {k: v for k, v in summary.items() if k != "report_generated_utc"},
        title="Compliance Summary"
    )
    print("[alert.py] Done. Check results/compliance_report.html")
