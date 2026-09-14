"""
download_dataset.py
===================
Automatically downloads the Elliptic Bitcoin dataset from Kaggle.

Two modes:
  Mode 1 (Kaggle API token) - fastest, fully automatic
  Mode 2 (opendatasets)     - prompts for Kaggle username + API key

Usage:
    python download_dataset.py                    # auto-detect mode
    python download_dataset.py --mode kaggle-api  # use Kaggle API token
    python download_dataset.py --mode opendatasets
"""

import argparse
import os
import sys
import zipfile
import shutil


DATASET_SLUG  = "ellipticco/elliptic-data-set"
DATASET_URL   = "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
RAW_DIR       = "data/raw"
EXPECTED_FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
]

# Colours (works on Windows 10+)
os.system("")
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def already_downloaded() -> bool:
    """Return True if all 3 dataset files exist in data/raw/."""
    return all(
        os.path.exists(os.path.join(RAW_DIR, f))
        for f in EXPECTED_FILES
    )


def ensure_raw_dir():
    os.makedirs(RAW_DIR, exist_ok=True)


def move_csvs_to_raw(src_dir: str):
    """Move any Elliptic CSV files from src_dir into data/raw/."""
    moved = 0
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f in EXPECTED_FILES:
                src  = os.path.join(root, f)
                dst  = os.path.join(RAW_DIR, f)
                shutil.move(src, dst)
                print(f"  {GREEN}Moved:{RESET} {f}")
                moved += 1
    return moved


def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file and move CSVs to data/raw/."""
    print(f"  Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    moved = move_csvs_to_raw(extract_to)
    os.remove(zip_path)
    return moved


# -------------------------------------------------------------------
# Mode 1 — Kaggle API (uses ~/.kaggle/kaggle.json)
# -------------------------------------------------------------------

def download_via_kaggle_api():
    """
    Download using the official Kaggle API library.
    Requires ~/.kaggle/kaggle.json (API token from Kaggle account settings).

    How to get your token:
      1. Go to https://www.kaggle.com/settings
      2. Scroll to API section -> click "Create New Token"
      3. A kaggle.json file downloads automatically
      4. Place it at: C:\\Users\\<you>\\.kaggle\\kaggle.json
    """
    try:
        import kaggle
    except ImportError:
        print(f"  {YELLOW}Installing kaggle library...{RESET}")
        os.system(f"{sys.executable} -m pip install kaggle -q")
        import kaggle

    token_path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(token_path):
        print(f"\n  {RED}kaggle.json not found at: {token_path}{RESET}")
        print(f"\n  {YELLOW}To get your Kaggle API token:{RESET}")
        print("    1. Go to https://www.kaggle.com/settings")
        print("    2. Scroll to API section -> click 'Create New Token'")
        print("    3. Save kaggle.json to:")
        print(f"       {token_path}")
        print("    4. Re-run this script\n")
        return False

    print(f"  Kaggle token found: {token_path}")
    print(f"  Downloading dataset: {DATASET_SLUG}")
    print(f"  This may take a minute (dataset ~60 MB)...\n")

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        DATASET_SLUG,
        path=RAW_DIR,
        unzip=True,
        quiet=False,
    )
    return True


# -------------------------------------------------------------------
# Mode 2 — opendatasets (prompts for credentials)
# -------------------------------------------------------------------

def download_via_opendatasets():
    """
    Download using opendatasets library.
    Will prompt for Kaggle username and API key (one-time).
    """
    try:
        import opendatasets as od
    except ImportError:
        print(f"  {YELLOW}Installing opendatasets...{RESET}")
        os.system(f"{sys.executable} -m pip install opendatasets -q")
        import opendatasets as od

    print(f"\n  {YELLOW}You will be prompted for your Kaggle username and API key.{RESET}")
    print("  Get your key from: https://www.kaggle.com/settings -> API -> Create New Token\n")

    od.download(DATASET_URL, data_dir=RAW_DIR)

    # opendatasets creates a subfolder — move files up to data/raw/
    inner = os.path.join(RAW_DIR, "elliptic-data-set")
    if os.path.isdir(inner):
        move_csvs_to_raw(inner)
        shutil.rmtree(inner, ignore_errors=True)
    return True


# -------------------------------------------------------------------
# Mode 3 — Manual fallback instructions
# -------------------------------------------------------------------

def print_manual_instructions():
    print(f"""
  {YELLOW}Automatic download requires a Kaggle account.{RESET}

  Manual steps (2 minutes):
  ------------------------------------------
  1. Go to: {CYAN}{DATASET_URL}{RESET}
  2. Sign in to Kaggle (free account)
  3. Click the Download button
  4. Extract the zip file
  5. Place these 3 files in {BOLD}data/raw/{RESET}:
       - elliptic_txs_features.csv
       - elliptic_txs_edgelist.csv
       - elliptic_txs_classes.csv
  6. Re-run setup.bat
  ------------------------------------------
""")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download Elliptic dataset")
    parser.add_argument(
        "--mode",
        choices=["auto", "kaggle-api", "opendatasets", "manual"],
        default="auto",
        help="Download mode (default: auto)"
    )
    args = parser.parse_args()

    print(f"\n{CYAN}{'='*55}{RESET}")
    print(f"{BOLD}  Elliptic Dataset Downloader{RESET}")
    print(f"{CYAN}{'='*55}{RESET}")

    ensure_raw_dir()

    if already_downloaded():
        print(f"\n  {GREEN}Dataset already present in data/raw/ - skipping download.{RESET}")
        for f in EXPECTED_FILES:
            size_mb = os.path.getsize(os.path.join(RAW_DIR, f)) / 1e6
            print(f"    {GREEN}Found:{RESET} {f}  ({size_mb:.1f} MB)")
        print()
        return True

    mode = args.mode
    success = False

    if mode in ("auto", "kaggle-api"):
        print(f"\n  Trying Kaggle API download...")
        try:
            success = download_via_kaggle_api()
        except Exception as e:
            print(f"  {YELLOW}Kaggle API failed: {e}{RESET}")
            if mode == "kaggle-api":
                print_manual_instructions()
                return False

    if not success and mode in ("auto", "opendatasets"):
        print(f"\n  Trying opendatasets download...")
        try:
            success = download_via_opendatasets()
        except Exception as e:
            print(f"  {YELLOW}opendatasets failed: {e}{RESET}")

    if not success:
        print_manual_instructions()
        return False

    # Verify files landed correctly
    missing = [f for f in EXPECTED_FILES
               if not os.path.exists(os.path.join(RAW_DIR, f))]
    if missing:
        print(f"\n  {YELLOW}Download complete but files not found in data/raw/:{RESET}")
        for f in missing:
            print(f"    Missing: {f}")
        print(f"\n  Check for the files and move them to data/raw/ manually.")
        return False

    print(f"\n  {GREEN}Dataset downloaded successfully!{RESET}")
    for f in EXPECTED_FILES:
        size_mb = os.path.getsize(os.path.join(RAW_DIR, f)) / 1e6
        print(f"    {GREEN}OK:{RESET} {f}  ({size_mb:.1f} MB)")
    print()
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
