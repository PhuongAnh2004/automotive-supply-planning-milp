# src/utils.py
from __future__ import annotations

import numpy as np
import pandas as pd


def trim(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_float_money(x) -> float:
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return 0.0
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def to_rate(x) -> float:
    """
    Convert rate to decimal:
      "20%" -> 0.2
      20    -> 0.2
      0.2   -> 0.2
      "0.2%"-> 0.002
    """
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v / 100.0 if v > 1.0 else v

    s = str(x).strip()
    if s == "":
        return 0.0

    if s.endswith("%"):
        s2 = s[:-1].strip().replace(",", "")
        try:
            return float(s2) / 100.0
        except Exception:
            return 0.0

    s = s.replace(",", "")
    try:
        v = float(s)
        return v / 100.0 if v > 1.0 else v
    except Exception:
        return 0.0


def is_intish(x: float, tol: float = 1e-9) -> bool:
    return abs(float(x) - round(float(x))) <= tol


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Return first existing column name from candidates (for robust schema handling).
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing columns. Tried {candidates}. Existing: {list(df.columns)}")
