# src/run.py
from __future__ import annotations

import argparse
import pathlib
import sys

# Make "src" importable when running: python src/run.py
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_excel import read_excel_input, write_excel_output
from src.model import solve_mps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Excel input path (e.g., data/input.xlsx)")
    ap.add_argument("--output", required=True, help="Excel output path (e.g., outputs/output.xlsx)")

    ap.add_argument("--inv0_fg", type=float, default=0.0)
    ap.add_argument("--inv0_l1", type=float, default=0.0)
    ap.add_argument("--hold_rate_annual", type=float, default=0.20)
    ap.add_argument("--time_limit_sec", type=int, default=180)
    ap.add_argument("--gap_rel", type=float, default=0.01)
    ap.add_argument("--solver_msg", action="store_true", help="Show solver logs")

    args = ap.parse_args()

    data = read_excel_input(args.input)
    results = solve_mps(
        data,
        inv0_fg=args.inv0_fg,
        inv0_l1=args.inv0_l1,
        hold_rate_annual=args.hold_rate_annual,
        time_limit_sec=args.time_limit_sec,
        gap_rel=args.gap_rel,
        solver_msg=args.solver_msg,
    )

    write_excel_output(args.output, results)
    print(f"Saved output -> {args.output}")


if __name__ == "__main__":
    main()
