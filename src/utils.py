"""
utils.py
========
Shared utility functions used across all pipeline modules.

Includes:
    - set_seed()           : reproducible RNG seeding
    - load_config()        : YAML config loader
    - get_logger()         : consistent logging to console + file
    - print_metrics_table(): pretty-print a dict of metrics
    - ensure_dirs()        : create output directories safely
    - Timer                : context manager for wall-clock timing
"""

import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Dict, Optional

import numpy as np
import yaml


# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------

def set_seed(seed: int = 42):
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.
    Call this at the start of every training or evaluation run.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load YAML config file and return as dict.

    Args:
        config_path: Path to config.yaml

    Returns:
        cfg (dict): Parsed configuration

    Raises:
        FileNotFoundError: If config file does not exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            "Make sure you are running from the project root."
        )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

def get_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    Create and return a logger that writes to console (and optionally a file).

    Args:
        name     : Logger name (usually __name__ of the calling module)
        log_file : Optional path to write log file
        level    : Logging level (default INFO)

    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # Avoid duplicate handlers on re-import
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# -------------------------------------------------------------------
# Directory helpers
# -------------------------------------------------------------------

def ensure_dirs(cfg: dict):
    """
    Create all output directories defined in config paths section.
    Safe to call multiple times (uses exist_ok=True).
    """
    for key, path in cfg.get("paths", {}).items():
        if path and not path.endswith((".pt", ".pth", ".csv", ".pkl")):
            os.makedirs(path, exist_ok=True)


# -------------------------------------------------------------------
# Metrics display
# -------------------------------------------------------------------

def print_metrics_table(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty-print a dict of metric name -> value as a table.

    Example:
        print_metrics_table({"accuracy": 0.92, "f1_macro": 0.88}, title="Test Results")

        -------------------------------
        - Test Results               -
        ------------------------------
        - accuracy     -   0.9200    -
        - f1_macro     -   0.8800    -
        ------------------------------
    """
    width_key = max(len(k) for k in metrics) + 2
    width_val = 12
    sep = "-" * (width_key + width_val + 3)

    print(f"\n  +{sep}+")
    print(f"  | {title:<{width_key + width_val}}|")
    print(f"  +{'-'*width_key}+{'-'*width_val}+")
    for k, v in metrics.items():
        if isinstance(v, float):
            val_str = f"{v:.4f}"
        elif v == "N/A" or v is None:
            val_str = "N/A"
        else:
            val_str = str(v)
        print(f"  | {k:<{width_key-1}}| {val_str:>{width_val-1}} |")
    print(f"  +{'-'*width_key}+{'-'*width_val}+\n")


# -------------------------------------------------------------------
# Timer
# -------------------------------------------------------------------

@contextmanager
def Timer(label: str = ""):
    """
    Context manager that prints the elapsed wall-clock time.

    Usage:
        with Timer("Encrypted inference"):
            result = model.infer(enc_data)
        # Prints: [Timer] Encrypted inference: 4.32s
    """
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    tag = f" {label}" if label else ""
    print(f"  [Timer]{tag}: {elapsed:.4f}s  ({elapsed*1000:.1f}ms)")


# -------------------------------------------------------------------
# Device helper
# -------------------------------------------------------------------

def get_device(prefer_gpu: bool = True) -> str:
    """Return 'cuda' if available and preferred, else 'cpu'."""
    try:
        import torch
        if prefer_gpu and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


if __name__ == "__main__":
    cfg = load_config()
    logger = get_logger("utils_test")
    logger.info("Config loaded OK.")
    logger.info(f"Device: {get_device()}")
    print_metrics_table({"accuracy": 0.92, "f1_macro": 0.88, "auc_roc": 0.95}, "Test Results")
    with Timer("Example operation"):
        time.sleep(0.1)
