from __future__ import annotations

import json
import tarfile
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def build_member_name(window: int, symbol: str, end_date: str) -> str:
    """
    根据 tar 内部成员命名规则构造成员路径。

    当前按你的 tar 结构默认使用：
        {symbol}/{symbol}_end{end_date}_N{window}.png

    例如：
        000001/000001_end20180206_N60.png
    """
    return f"{symbol}/{symbol}_end{end_date}_N{window}.png"


def load_symbol_to_tar(json_path: str) -> Dict[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {str(k): str(v) for k, v in mapping.items()}


def group_indices_by_tar(
    symbols: np.ndarray,
    symbol_to_tar: Dict[str, str],
) -> Tuple[Dict[str, List[int]], List[int]]:
    """
    按 tar_path 对样本下标分组。
    返回:
        grouped: tar_path -> [idx1, idx2, ...]
        missing_tar_indices: 找不到 tar 映射的样本下标
    """
    grouped: Dict[str, List[int]] = defaultdict(list)
    missing_tar_indices: List[int] = []

    for i, sym in enumerate(symbols):
        tar_path = symbol_to_tar.get(str(sym))
        if tar_path is None:
            missing_tar_indices.append(i)
        else:
            grouped[tar_path].append(i)

    return grouped, missing_tar_indices


def validate_members_and_build_mask(
    symbols: np.ndarray,
    end_dates: np.ndarray,
    window: int,
    symbol_to_tar: Dict[str, str],
    verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    根据 tar 中真实存在的成员，构造保留掩码 keep_mask。

    规则：
    - symbol 找不到 tar -> 丢弃
    - tar 无法读取 -> 丢弃该 tar 下全部样本
    - member_name 不在 tar 中 -> 丢弃

    返回:
        keep_mask: shape [M], bool
        stats: 统计信息
    """
    n = len(symbols)
    keep_mask = np.zeros(n, dtype=bool)

    grouped, missing_tar_indices = group_indices_by_tar(symbols, symbol_to_tar)

    bad_tar_count = 0
    missing_member_count = 0
    kept_count = 0

    if verbose:
        print(f"[clean] total rows in meta: {n}")
        print(f"[clean] tar groups to validate: {len(grouped)}")
        if missing_tar_indices:
            print(f"[clean] rows missing symbol->tar mapping: {len(missing_tar_indices)}")

    for group_idx, (tar_path, indices) in enumerate(grouped.items(), start=1):
        try:
            with tarfile.open(tar_path, mode="r") as tf:
                member_names = {m.name for m in tf if m.isfile()}
        except Exception as e:
            bad_tar_count += len(indices)
            if verbose:
                print(f"[WARN] failed to open tar: {tar_path}")
                print(f"       dropped rows in this tar: {len(indices)}")
                print(f"       error: {e}")
            continue

        for i in indices:
            member_name = build_member_name(window, str(symbols[i]), str(end_dates[i]))
            if member_name in member_names:
                keep_mask[i] = True
                kept_count += 1
            else:
                missing_member_count += 1

        if verbose and (group_idx % 20 == 0 or group_idx == len(grouped)):
            print(f"[clean] validated tar groups: {group_idx}/{len(grouped)}")

    stats = {
        "total_rows": int(n),
        "missing_tar_mapping": int(len(missing_tar_indices)),
        "bad_tar_dropped": int(bad_tar_count),
        "missing_member_dropped": int(missing_member_count),
        "kept_rows": int(kept_count),
        "dropped_rows_total": int(n - kept_count),
    }

    if verbose:
        print("[clean] summary:")
        for k, v in stats.items():
            print(f"  - {k}: {v}")

    return keep_mask, stats


def save_clean_npz(
    src_meta_path: str,
    out_meta_path: str,
    keep_mask: np.ndarray,
) -> None:
    """
    按 keep_mask 过滤原始 meta，并保存成新的 clean npz。
    除 window / horizons / seed / test_start_date 外，其余按第一维过滤。
    """
    meta = np.load(src_meta_path, allow_pickle=True)

    out = {}

    for key in meta.files:
        arr = meta[key]

        # 这些字段是全局元信息，不按样本维过滤
        if key in ("horizons", "window", "seed", "test_start_date"):
            out[key] = arr
            continue

        # 一维样本字段: symbol / end_date / split
        if arr.ndim >= 1 and len(arr) == len(keep_mask):
            out[key] = arr[keep_mask]
        else:
            # 理论上这里一般不会遇到其它情况，保守直接原样保留
            out[key] = arr

    out_path = Path(out_meta_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    print(f"[clean] saved clean meta to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate tar members and create clean meta npz.")
    parser.add_argument("--meta_path", type=str, required=True, help="原始 meta_N*.npz 路径")
    parser.add_argument("--symbol_to_tar_json", type=str, required=True, help="symbol->tar 映射 json")
    parser.add_argument("--out_meta_path", type=str, required=True, help="输出 clean npz 路径")
    parser.add_argument("--quiet", action="store_true", help="少打印日志")

    args = parser.parse_args()
    verbose = not args.quiet

    meta = np.load(args.meta_path, allow_pickle=True)

    symbols = meta["symbol"].astype(str)
    end_dates = meta["end_date"].astype(str)
    window = int(meta["window"][0])

    symbol_to_tar = load_symbol_to_tar(args.symbol_to_tar_json)

    keep_mask, stats = validate_members_and_build_mask(
        symbols=symbols,
        end_dates=end_dates,
        window=window,
        symbol_to_tar=symbol_to_tar,
        verbose=verbose,
    )

    save_clean_npz(
        src_meta_path=args.meta_path,
        out_meta_path=args.out_meta_path,
        keep_mask=keep_mask,
    )


if __name__ == "__main__":
    main()

"""
python clean_meta_by_tar.py \
  --meta_path /workspace/label_npz/meta_N60.npz \
  --symbol_to_tar_json /workspace/symbol_to_tar/symbol_to_tar_N60.json \
  --out_meta_path /workspace/label_npz/meta_N60_clean.npz
"""