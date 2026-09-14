"""
run_pipeline.py
===============
One-click runner for the full CRYPTO ML pipeline.

Runs all steps in order:
    Step 1 -> Preprocess  (CSV -> PyG graph)
    Step 2 -> Train       (plaintext GNN)
    Step 3 -> Evaluate    (accuracy + F1 + plots)
    Step 4 -> Encrypt     (HE inference benchmark)

Each step checks for errors before moving on.
If a step fails, it tells you exactly what to fix and stops cleanly.

Usage:
    python run_pipeline.py                   # run all steps
    python run_pipeline.py --skip-encrypt    # skip slow HE step
    python run_pipeline.py --step preprocess # run one step only
    python run_pipeline.py --config configs/config.yaml
"""

import argparse
import os
import sys
import time
import traceback
import yaml


# -------------------------------------------------------------------
# Colours for terminal output (works on Windows 10+ and Linux/Mac)
# -------------------------------------------------------------------

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"


def banner(text: str):
    width = 65
    print(f"\n{C.CYAN}{'='*width}{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}  {text}{C.RESET}")
    print(f"{C.CYAN}{'='*width}{C.RESET}")


def ok(msg: str):
    print(f"  {C.GREEN}OK  {msg}{C.RESET}")


def warn(msg: str):
    print(f"  {C.YELLOW}!  {msg}{C.RESET}")


def err(msg: str):
    print(f"  {C.RED}X  {msg}{C.RESET}")


def info(msg: str):
    print(f"  {C.CYAN}->  {msg}{C.RESET}")


def divider():
    print(f"{C.CYAN}{'-'*65}{C.RESET}")


# -------------------------------------------------------------------
# Pre-flight checks
# -------------------------------------------------------------------

def check_python_version():
    """Make sure Python 3.10+ is being used."""
    banner("Pre-flight Check")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        err(f"Python 3.10+ required. You have {major}.{minor}.")
        err("Download from https://python.org")
        sys.exit(1)
    ok(f"Python {major}.{minor} OK")


def check_dependencies():
    """Check all required libraries are installed."""
    required = {
        "torch":              "pip install torch",
        "torch_geometric":    "pip install torch-geometric",
        "pandas":             "pip install pandas",
        "numpy":              "pip install numpy",
        "sklearn":            "pip install scikit-learn",
        "yaml":               "pip install pyyaml",
        "tqdm":               "pip install tqdm",
    }
    optional = {
        "matplotlib": "pip install matplotlib (optional, for offline PNG charts)",
        "tenseal":    "pip install tenseal   (needed for Step 4 - encrypted inference)",
    }

    all_good = True
    for lib, install_cmd in required.items():
        try:
            __import__(lib)
            ok(f"{lib}")
        except Exception as e:
            err(f"{lib} NOT found or blocked ({e})  ->  Run: {install_cmd}")
            all_good = False

    for lib, note in optional.items():
        try:
            __import__(lib)
            ok(f"{lib} (optional)")
        except Exception:
            warn(f"{lib} not available/blocked  ->  {note}")

    if not all_good:
        print()
        err("Some required libraries are missing.")
        info("Install everything at once:  pip install -r requirements.txt")
        sys.exit(1)

    ok("All required core libraries present.")


def check_dataset(cfg: dict) -> bool:
    """Check whether raw Elliptic CSV files or bundled sample dataset exist."""
    raw_dir = cfg["paths"]["raw_data"]
    files   = [
        "elliptic_txs_features.csv",
        "elliptic_txs_edgelist.csv",
        "elliptic_txs_classes.csv",
    ]
    missing = [f for f in files if not os.path.exists(os.path.join(raw_dir, f))]

    if missing:
        sample_dir = os.path.join("data", "sample")
        sample_missing = [f for f in files if not os.path.exists(os.path.join(sample_dir, f))]
        if not sample_missing:
            ok("Bundled sample dataset found in data/sample/ (8,800 nodes, 9,493 edges - ready to run)")
            return True

        err("Elliptic dataset CSV files not found in data/raw/ or data/sample/")
        print()
        print(f"  {C.YELLOW}Missing files:{C.RESET}")
        for f in missing:
            print(f"    - {f}")
        print()
        info("Download steps:")
        info("  1. Run: python download_dataset.py")
        info("  2. Or download manually from: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
        return False

    ok("All 3 Elliptic CSV files found in data/raw/")
    return True


def check_processed_graph(cfg: dict) -> bool:
    """Check whether processed PyG graph exists."""
    proc_dir   = cfg["paths"]["processed_data"]
    graph_path = os.path.join(proc_dir, "elliptic_graph.pt")
    sample_path = os.path.join("data", "sample", "sample_graph.pt")

    if os.path.exists(graph_path):
        ok("Processed graph found in data/processed/")
        return True
    elif os.path.exists(sample_path):
        ok("Bundled processed graph found in data/sample/sample_graph.pt")
        return True
    else:
        warn("Processed graph not found - will run preprocessing first.")
        return False


def check_trained_model(cfg: dict) -> bool:
    """Check whether a trained model checkpoint exists."""
    model_path = cfg["paths"]["model_save"]
    if not os.path.exists(model_path):
        warn("No trained model found — will run training first.")
        return False
    ok(f"Trained model found:  {model_path}")
    return True


# -------------------------------------------------------------------
# Pipeline Steps
# -------------------------------------------------------------------

def run_step(step_name: str, fn, *args, **kwargs):
    """
    Run a pipeline step, time it, and catch any errors cleanly.
    Returns True if success, False if failed.
    """
    print()
    banner(step_name)
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        elapsed = time.time() - t0
        ok(f"{step_name} completed in {elapsed:.1f}s")
        return True
    except FileNotFoundError as e:
        err(f"File not found: {e}")
        info("Check that all previous steps completed successfully.")
        return False
    except ImportError as e:
        err(f"Missing library: {e}")
        info("Run:  pip install -r requirements.txt")
        return False
    except RuntimeError as e:
        err(f"Runtime error: {e}")
        traceback.print_exc()
        return False
    except KeyboardInterrupt:
        warn("Step interrupted by user (Ctrl+C).")
        return False
    except Exception as e:
        err(f"Unexpected error in {step_name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# -------------------------------------------------------------------
# Individual step wrappers
# -------------------------------------------------------------------

def step_preprocess(cfg: dict):
    """Step 1 — Preprocess raw CSVs into PyG graph."""
    from src.preprocess import load_raw_data, build_pyg_graph, save_processed

    raw_dir  = cfg["paths"]["raw_data"]
    proc_dir = cfg["paths"]["processed_data"]

    df_feats, df_edges, df_cls = load_raw_data(raw_dir)
    data, scaler = build_pyg_graph(df_feats, df_edges, df_cls)
    save_processed(data, scaler, proc_dir)


def step_train(cfg: dict):
    """Step 2 — Train the GNN in plaintext."""
    from src.train import train
    train(cfg)


def step_evaluate(cfg: dict):
    """Step 3 — Evaluate all pipeline stages and save report + plots."""
    from src.evaluate import evaluate_all_stages
    evaluate_all_stages(cfg)


def step_encrypt(cfg: dict):
    """Step 4 — Run CKKS encrypted inference benchmark."""
    try:
        import tenseal as ts
    except ImportError:
        raise ImportError(
            "tenseal is not installed.\n"
            "     Run:  pip install tenseal\n"
            "     Or skip this step:  python run_pipeline.py --skip-encrypt"
        )

    from src.encrypt_infer import HEInferencePipeline
    from src.preprocess import load_processed
    import numpy as np

    info("Setting up CKKS context (this may take a moment)...")
    pipeline = HEInferencePipeline(cfg)
    ctx      = pipeline.setup_context()

    info("Running single-node encryption benchmark...")
    results  = pipeline.benchmark_single_node(ctx, feature_dim=cfg["dataset"]["node_features"])

    info("Loading test graph for sample inference...")
    data, _  = load_processed(cfg["paths"]["processed_data"])

    # Encrypt a small batch of 5 test nodes as a demo
    test_idx   = data.test_mask.nonzero(as_tuple=True)[0][:5]
    sample_feats = data.x[test_idx].numpy()

    info(f"Encrypting {len(test_idx)} sample test nodes...")
    t0 = time.time()
    enc_feats = pipeline.encrypt(ctx, sample_feats)
    t_enc = time.time() - t0

    ok(f"Encrypted {len(enc_feats)} nodes in {t_enc:.3f}s "
       f"({t_enc/len(enc_feats)*1000:.1f}ms per node)")

    info("(Full encrypted GNN inference over all nodes coming in next phase)")
    info("See src/encrypt_infer.py for the complete HE pipeline.")


# -------------------------------------------------------------------
# Final summary
# -------------------------------------------------------------------

def print_summary(results: dict):
    banner("Pipeline Summary")
    all_passed = all(results.values())
    for step, passed in results.items():
        if passed:
            ok(step)
        else:
            err(f"{step}  <- FAILED")

    print()
    if all_passed:
        print(f"  {C.GREEN}{C.BOLD}All steps completed successfully!{C.RESET}")
        print()
        info("Next steps:")
        info("  View metrics:  results/evaluation_report.csv")
        info("  View plots:    results/plots/stage_comparison.png")
        info("  View plots:    results/plots/latency_comparison.png")
    else:
        print(f"  {C.RED}{C.BOLD}Some steps failed. See errors above.{C.RESET}")
        print()
        info("Fix the errors and re-run:  python run_pipeline.py")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    # Enable ANSI colours on Windows
    os.system("")

    parser = argparse.ArgumentParser(
        description="CRYPTO ML — Full Pipeline Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--step",
        choices=["preprocess", "train", "evaluate", "encrypt"],
        help="Run only one specific step instead of all"
    )
    parser.add_argument(
        "--skip-encrypt", action="store_true",
        help="Skip the slow encrypted inference step (Step 4)"
    )
    args = parser.parse_args()

    # -- Load config -----------------------------------------------
    if not os.path.exists(args.config):
        err(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # -- Pre-flight ------------------------------------------------
    check_python_version()
    check_dependencies()

    # -- Single-step mode -----------------------------------------
    if args.step:
        step_map = {
            "preprocess": step_preprocess,
            "train":      step_train,
            "evaluate":   step_evaluate,
            "encrypt":    step_encrypt,
        }
        fn = step_map[args.step]
        success = run_step(f"Step: {args.step}", fn, cfg)
        sys.exit(0 if success else 1)

    # -- Full pipeline ---------------------------------------------
    results = {}

    # Step 1 — Preprocess
    if not check_dataset(cfg):
        err("Cannot continue without the dataset. Download it first (see above).")
        sys.exit(1)

    success = run_step("Step 1 / 4 — Preprocessing", step_preprocess, cfg)
    results["Step 1: Preprocess"] = success
    if not success:
        err("Preprocessing failed. Cannot continue.")
        print_summary(results)
        sys.exit(1)

    # Step 2 — Train
    success = run_step("Step 2 / 4 — Training (plaintext GNN)", step_train, cfg)
    results["Step 2: Train"] = success
    if not success:
        err("Training failed. Cannot continue.")
        print_summary(results)
        sys.exit(1)

    # Step 3 — Evaluate
    success = run_step("Step 3 / 4 — Evaluation", step_evaluate, cfg)
    results["Step 3: Evaluate"] = success

    # Step 4 — Encrypted Inference (optional)
    if args.skip_encrypt:
        warn("Step 4 (encrypted inference) skipped via --skip-encrypt flag.")
        results["Step 4: Encrypt (skipped)"] = True
    else:
        success = run_step("Step 4 / 4 — Encrypted Inference (CKKS)", step_encrypt, cfg)
        results["Step 4: Encrypt"] = success
        if not success:
            warn("Encrypted inference failed — other steps still completed.")

    # -- Summary ---------------------------------------------------
    print_summary(results)


if __name__ == "__main__":
    main()
