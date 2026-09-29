"""Repository paths for the paper's analysis and figure scripts.

Everything is relative to this file, so the scripts run from any checkout; set LOOPDYN_DIR to point at another copy of the
experiment outputs. Importing this module also puts the experiment code (scripts/) on the path, for figstyle and mqa_common.
"""
import os
import sys
from pathlib import Path

LOOPDYN = Path(os.environ.get("LOOPDYN_DIR", Path(__file__).resolve().parents[1]))
OUTDIR = LOOPDYN / "analysis" / "out"
RES = str(LOOPDYN / "results")          # raw outputs of the experiment scripts
DATA = str(LOOPDYN / "data")            # exported datasets (MuSiQue development selection)
NOTES = str(OUTDIR / "notes")           # routing-window table
OUT = str(OUTDIR / "figs")              # figures
TABLES = str(OUTDIR / "tables")         # generated tables and macros
CHECKS = str(OUTDIR / "checks")         # machine-readable summaries of the analyses
for _d in (OUT, TABLES, CHECKS, NOTES):
    os.makedirs(_d, exist_ok=True)
sys.path.insert(0, str(LOOPDYN / "scripts"))
