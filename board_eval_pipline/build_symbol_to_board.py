from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from board_utils import normalize_symbol, infer_board_from_symbol


def load_symbols_from_meta(meta_path: str) -> pd.DataFrame:
    meta = np.load(meta_path, allow_pickle=True)
    if "symbol" not in meta:
        raise ValueError(f"meta file missing key 'symbol': {meta_path}")

    df = pd.DataFrame({"symbol": meta["symbol"]})
    return df


def load_symbols_from_pred(pred_csv: str) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    if "symbol" not in df.columns:
        raise ValueError(f"pred csv missing column 'symbol': {pred_csv}")
    return df[["symbol"]].copy()


def main():
    parser = argparse.ArgumentParser(description="Build symbol -> board mapping by static code rules.")
    parser.add_argument("--meta_path", type=str, default=None, help="从 meta_N*_clean.npz 提取 symbol")
    parser.add_argument("--pred_csv", type=str, default=None, help="从预测 csv 提取 symbol")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--drop_unknown", action="store_true", help="是否丢弃 unknown 板块")
    args = parser.parse_args()

    if not args.meta_path and not args.pred_csv:
        raise ValueError("至少提供 --meta_path 或 --pred_csv 之一")

    parts = []
    if args.meta_path:
        parts.append(load_symbols_from_meta(args.meta_path))
    if args.pred_csv:
        parts.append(load_symbols_from_pred(args.pred_csv))

    df = pd.concat(parts, axis=0, ignore_index=True)
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df = df.drop_duplicates().sort_values("symbol").reset_index(drop=True)

    df["board"] = df["symbol"].map(infer_board_from_symbol)

    if args.drop_unknown:
        df = df.loc[df["board"] != "unknown"].reset_index(drop=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"saved symbol->board map to: {out_path}")
    print(df["board"].value_counts(dropna=False).sort_index())
    print(df.head())


if __name__ == "__main__":
    main()
