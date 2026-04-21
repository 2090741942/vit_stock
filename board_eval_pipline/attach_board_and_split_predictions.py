from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from board_utils import BOARD_ORDER, normalize_symbol


def main():
    parser = argparse.ArgumentParser(
        description="Attach board labels to prediction csv and split into per-board prediction files."
    )
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--symbol_to_board_csv", type=str, required=True)
    parser.add_argument("--out_with_board_csv", type=str, required=True)
    parser.add_argument("--split_out_dir", type=str, required=True)
    parser.add_argument("--drop_unknown", action="store_true")
    args = parser.parse_args()

    pred = pd.read_csv(args.pred_csv)
    required = {"symbol", "end_date", "prob_up"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"pred csv missing columns: {missing}")

    pred["symbol"] = pred["symbol"].map(normalize_symbol)

    m = pd.read_csv(args.symbol_to_board_csv)
    if not {"symbol", "board"}.issubset(m.columns):
        raise ValueError("symbol_to_board csv must contain columns: symbol, board")
    m["symbol"] = m["symbol"].map(normalize_symbol)
    m = m[["symbol", "board"]].drop_duplicates()

    out = pred.merge(m, on="symbol", how="left")
    out["board"] = out["board"].fillna("unknown")

    if args.drop_unknown:
        out = out.loc[out["board"] != "unknown"].reset_index(drop=True)

    out_path = Path(args.out_with_board_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved tagged prediction csv to: {out_path}")
    print(out["board"].value_counts(dropna=False).sort_index())

    split_dir = Path(args.split_out_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    for board in BOARD_ORDER + ["unknown"]:
        sub = out.loc[out["board"] == board].copy()
        if len(sub) == 0:
            continue
        fp = split_dir / f"{Path(args.pred_csv).stem}_{board}.csv"
        sub.to_csv(fp, index=False, encoding="utf-8-sig")
        print(f"saved {board}: {fp} | rows={len(sub)}")


if __name__ == "__main__":
    main()
