from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def normalize_symbol(x: object) -> str:
    s = str(x).strip()
    if "." in s:
        s = s.split(".")[0]
    return s.zfill(6)


def main():
    parser = argparse.ArgumentParser(description="Normalize symbol column in prediction csv.")
    parser.add_argument("--in_csv", type=str, required=True, help="Input prediction csv")
    parser.add_argument("--out_csv", type=str, required=True, help="Output normalized csv")
    args = parser.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)

    df = pd.read_csv(in_path)

    if "symbol" not in df.columns:
        raise ValueError("Input csv must contain a 'symbol' column.")

    df = df.copy()
    df["symbol_raw"] = df["symbol"]
    df["symbol"] = df["symbol"].map(normalize_symbol)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"saved normalized prediction csv to: {out_path}")
    print("preview:")
    print(df[["symbol_raw", "symbol"]].head(10))


if __name__ == "__main__":
    main()

"""
python normalize_prediction_symbols.py \
  --in_csv /workspace/predict/preds_I60_R5.csv \
  --out_csv /workspace/predict/preds_I60_R5_normalized.csv
"""