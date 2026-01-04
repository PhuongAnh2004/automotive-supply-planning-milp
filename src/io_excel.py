# src/io_excel.py
from __future__ import annotations

import pandas as pd

DEFAULT_SHEETS = [
    "BOM_FG_L1",
    "BOM_L1_L2",
    "DEMAND_FG",
    "INV_START",
    "CAP_FG",
    "ITEM_MASTER",
    "COST_BUY_MONTHLY",
]


def read_excel_input(path: str, sheets: list[str] | None = None) -> dict[str, pd.DataFrame]:
    sheets = sheets or DEFAULT_SHEETS
    xl = pd.ExcelFile(path)
    return {name: xl.parse(name).copy() for name in sheets}


def write_excel_output(path: str, results: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for tab, df in results.items():
            df.to_excel(w, sheet_name=tab[:31], index=False)  # Excel tab name <= 31 chars
