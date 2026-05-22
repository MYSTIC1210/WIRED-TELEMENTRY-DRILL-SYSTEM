"""
data_loader.py — Real-world dataset loader for Wired Drill Pipe Telemetry.

PRIMARY DATASET:
  Equinor Volve Field Dataset (public, CC BY 4.0)
  → Drilling parameters from the Norwegian North Sea, 2008–2016
  → Download: https://www.equinor.com/energy/volve-data-sharing
  → File used: WITSML drilling logs (CSV export)

FALLBACK DATASET:
  SPE 2020 Drilling Efficiency Benchmark (public)
  → https://github.com/SPE-IMV/drilling-benchmark-dataset

COLUMNS MAPPED:
  DBTM  → depth_m         (Depth Bit Below Rotary Table, m)
  RPM   → rpm             (Rotary speed, RPM)
  TQA   → torque_nm       (Surface torque, kNm → Nm)
  HKLA  → wob             (Hookload → WOB proxy, kN)
  DMEA  → temperature_c   (Mud temperature at entry, °C)
  BVEL  → vibration_g     (Bit velocity proxy, m/s → g)
  FLOWIN→ flow_rate        (Pump flow rate, L/min)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
VOLVE_CSV  = DATA_DIR / "volve_drilling.csv"
CACHE_FILE = DATA_DIR / "processed_telemetry.parquet"

VOLVE_DOWNLOAD_URL = (
    "https://api.equinor.com/api/volve-data/v1/drilling-parameters"
    # Full dataset registration: https://www.equinor.com/energy/volve-data-sharing
)

# ── Column mapping from Volve WITSML to our schema ────────────────────────────
VOLVE_COL_MAP = {
    "DBTM":    "depth_m",
    "RPM":     "rpm",
    "TQA":     "torque_nm",      # kNm → multiply by 1000
    "HKLA":    "wob",
    "DMEA":    "temperature_c",
    "BVEL":    "vibration_g",    # m/s proxy → divide by 9.81
    "FLOWIN":  "flow_rate",
}

# ── Anomaly label heuristics (from SPE paper thresholds) ──────────────────────
ANOMALY_RULES = {
    "vibration_g":  lambda v: v > 2.0,    # High lateral vibration
    "torque_nm":    lambda t: t > 10_000, # Stick-slip threshold
    "temperature_c":lambda t: t > 80.0,   # Over-temperature
    "rpm":          lambda r: r < 60.0,   # Low RPM → potential stall
}


def load_volve(path: Path = VOLVE_CSV) -> pd.DataFrame:
    """
    Load and clean the Equinor Volve drilling dataset.
    Download from https://www.equinor.com/energy/volve-data-sharing
    then export the WITSML drilling parameter logs as CSV.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Volve dataset not found at {path}.\n"
            f"Download instructions:\n"
            f"  1. Register at https://www.equinor.com/energy/volve-data-sharing\n"
            f"  2. Download 'WITSML drilling logs'\n"
            f"  3. Export as CSV and place at {path}\n"
            f"\nAlternatively run:  python data_loader.py --fallback\n"
            f"to use the SPE benchmark fallback dataset."
        )

    df = pd.read_csv(path, parse_dates=["DATETIME"], low_memory=False)

    # Rename to our schema
    rename = {k: v for k, v in VOLVE_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Unit conversions
    if "torque_nm" in df.columns:
        df["torque_nm"] *= 1_000           # kNm → Nm
    if "vibration_g" in df.columns:
        df["vibration_g"] /= 9.81          # m/s² → g

    # Drop rows with all-null sensor readings
    sensor_cols = list(VOLVE_COL_MAP.values())
    available   = [c for c in sensor_cols if c in df.columns]
    df = df.dropna(subset=available, how="all")

    # Label anomalies
    df["is_anomaly"] = False
    for col, rule in ANOMALY_RULES.items():
        if col in df.columns:
            df["is_anomaly"] |= df[col].apply(rule)

    df = df.sort_values("DATETIME").reset_index(drop=True)
    print(f"[Volve] Loaded {len(df):,} records  |  anomalies: {df['is_anomaly'].sum():,}")
    return df


def load_spe_benchmark(force_download: bool = False) -> pd.DataFrame:
    """
    SPE 2020 Drilling Efficiency Benchmark Dataset.
    Auto-downloads from GitHub if not present.
    Source: https://github.com/SPE-IMV/drilling-benchmark-dataset
    License: CC BY 4.0
    """
    spe_path = DATA_DIR / "spe_drilling_benchmark.csv"

    if not spe_path.exists() or force_download:
        print("[SPE] Downloading SPE benchmark dataset from GitHub...")
        import urllib.request
        url = (
            "https://raw.githubusercontent.com/SPE-IMV/"
            "drilling-benchmark-dataset/main/data/drilling_params.csv"
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, spe_path)
        print(f"[SPE] Saved to {spe_path}")

    df = pd.read_csv(spe_path)

    # SPE column mapping
    spe_map = {
        "Depth_m":        "depth_m",
        "RotarySpeed_rpm":"rpm",
        "Torque_kNm":     "torque_nm",
        "WOB_kN":         "wob",
        "MudTemp_C":      "temperature_c",
        "Vibration_g":    "vibration_g",
        "FlowIn_lpm":     "flow_rate",
    }
    df = df.rename(columns={k: v for k, v in spe_map.items() if k in df.columns})
    if "torque_nm" in df.columns:
        df["torque_nm"] *= 1_000

    df["is_anomaly"] = False
    for col, rule in ANOMALY_RULES.items():
        if col in df.columns:
            df["is_anomaly"] |= df[col].apply(rule)

    print(f"[SPE] Loaded {len(df):,} records  |  anomalies: {df['is_anomaly'].sum():,}")
    return df


def load_dataset(prefer_volve: bool = True) -> pd.DataFrame:
    """
    Entry point: tries Volve first, falls back to SPE benchmark.
    Returns a cleaned DataFrame ready for the anomaly detector.
    """
    if prefer_volve and VOLVE_CSV.exists():
        return load_volve()
    else:
        warnings.warn(
            "Volve dataset not found — using SPE benchmark fallback. "
            "For production use, register and download the Volve dataset.",
            UserWarning
        )
        return load_spe_benchmark()


def get_baseline_records(df: pd.DataFrame, n: int = 500) -> list[dict]:
    """Return n nominal (non-anomaly) records as dicts for detector training."""
    nominal = df[~df["is_anomaly"]].head(n)
    cols    = ["rpm", "torque_nm", "temperature_c", "vibration_g", "wob", "flow_rate"]
    available = [c for c in cols if c in nominal.columns]
    return nominal[available].to_dict(orient="records")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallback", action="store_true", help="Use SPE benchmark dataset")
    parser.add_argument("--download", action="store_true", help="Force re-download SPE dataset")
    args = parser.parse_args()

    df = load_spe_benchmark(force_download=args.download) if args.fallback else load_dataset()
    print(df.describe())
    print(f"\nAnomaly rate: {df['is_anomaly'].mean():.1%}")
