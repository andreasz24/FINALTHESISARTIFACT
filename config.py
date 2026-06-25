"""Project-wide configuration: paths, constants, device, and TabPFN token.

All file locations are resolved relative to the repository root by default, but
can be overridden with environment variables so the same code runs locally and
on Google Colab without editing source.

Environment variables
----------------------
THESIS_DATA_DIR     directory containing the raw CSV         (default: <repo>/data)
THESIS_RESULTS_DIR  directory for saved splits/results/figs  (default: <repo>/results)
TABPFN_TOKEN        Prior Labs license token for TabPFN-3
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Directories ─────────────────────────────────────────────────────────────
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

DATA_DIR = Path(os.environ.get("THESIS_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("THESIS_RESULTS_DIR", REPO_ROOT / "results"))
FIGURES_DIR = RESULTS_DIR / "figures"

# Created lazily so importing the package never fails on a read-only filesystem.
for _d in (RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Files ───────────────────────────────────────────────────────────────────
RAW_CSV = DATA_DIR / "corporateCreditRatingWithFinancialRatios.csv"
SPLITS_PATH = RESULTS_DIR / "splits.pkl"

# ── Data constants ──────────────────────────────────────────────────────────
# Encoding: D=0, C=1, CC=2, CCC=3, B=4, BB=5, BBB=6, A=7, AA=8, AAA=9
RATING_LABELS = ["D", "C", "CC", "CCC", "B", "BB", "BBB", "A", "AA", "AAA"]
N_CLASSES = len(RATING_LABELS)
ALL_CLASSES = list(range(N_CLASSES))

FEATURE_COLS = [
    "Current Ratio", "Long-term Debt / Capital", "Debt/Equity Ratio",
    "Gross Margin", "Operating Margin", "EBIT Margin", "EBITDA Margin",
    "Pre-Tax Profit Margin", "Net Profit Margin", "Asset Turnover",
    "ROE - Return On Equity", "Return On Tangible Equity",
    "ROA - Return On Assets", "ROI - Return On Investment",
    "Operating Cash Flow Per Share", "Free Cash Flow Per Share",
]

# Company-level group column used for the group-aware split (no company appears
# in both train and test). The original notebook saved its splits using this.
GROUP_COL = "Corporation"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Per-model plot colours, kept consistent across every figure.
MODEL_COLORS = {"XGBoost": "steelblue", "TabPFN-3": "darkorange", "AutoGluon": "seagreen"}


def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_tabpfn_token() -> bool:
    """Make sure TABPFN_TOKEN is set in the environment.

    TabPFN-3 needs a one-time license token to download weights for local
    inference. We look for it in this order:
      1. an already-set TABPFN_TOKEN environment variable;
      2. a Google Colab secret named TABPFN_TOKEN.

    Returns True if a token is present after the call.
    """
    if os.environ.get("TABPFN_TOKEN"):
        return True
    try:  # Colab secret fallback
        from google.colab import userdata  # type: ignore
        token = userdata.get("TABPFN_TOKEN")
        if token:
            os.environ["TABPFN_TOKEN"] = token
            return True
    except Exception:
        pass
    return bool(os.environ.get("TABPFN_TOKEN"))
