from __future__ import annotations

import os
import re
import json
import tarfile
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List


SYMBOL_RE = re.compile(r"^\d{6}$")


def extract_symbol_from_member(member_name: str) -> str | None:
    """
    从 tar 成员路径中提取股票代码 symbol。

    兼容两种常见情况：
    1) 000001/000001_end20180206_N5.png
    2) ohlc_image_N5/000001/000001_end20180206_N5.png

    返回:
        - 6位 symbol 字符串
        - 若无法识别则返回 None
    """
    parts = [p for p in member_name.split("/") if p not in ("", ".")]

    if not parts:
        return None

    # 情况1：第一层就是 symbol 文件夹
    if SYMBOL_RE.fullmatch(parts[0]):
        return parts[0]

    # 情况2：第一层是 ohlc_image_N5 之类，第二层是 symbol
    if len(parts) >= 2 and parts[0].startswith("ohlc_image_N") and SYMBOL_RE.fullmatch(parts[1]):
        return parts[1]

    return None


def scan_one_tar(tar_path: str) -> Set[str]:
    """
    扫描单个 tar，返回其中包含的所有 symbol。
    只读取成员名，不解码图片内容。
    """
    symbols: Set[str] = set()

    try:
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                # 只看路径名，不读取内容
                name = member.name
                symbol = extract_symbol_from_member(name)
                if symbol is not None:
                    symbols.add(symbol)

    except tarfile.TarError as e:
        raise RuntimeError(f"无法读取 tar 文件: {tar_path}\n原始错误: {e}") from e

    return symbols


def build_symbol_to_tar(tar_dir: str, suffixes: tuple[str, ...] = (".tar",)) -> Dict[str, str]:
    """
    扫描 tar_dir 下所有 tar 文件，建立 symbol -> tar_path 映射。

    要求：
    - 同一个 symbol 只能属于一个 tar
    """
    tar_dir = os.path.abspath(tar_dir)
    p = Path(tar_dir)

    if not p.exists():
        raise FileNotFoundError(f"目录不存在: {tar_dir}")
    if not p.is_dir():
        raise NotADirectoryError(f"不是目录: {tar_dir}")

    tar_files = sorted(
        str(x) for x in p.iterdir()
        if x.is_file() and x.suffix.lower() in suffixes
    )

    if not tar_files:
        raise FileNotFoundError(f"在目录 {tar_dir} 下没有找到 tar 文件")

    symbol_to_tar: Dict[str, str] = {}
    duplicate_symbols: defaultdict[str, List[str]] = defaultdict(list)
    empty_tars: List[str] = []

    print(f"发现 {len(tar_files)} 个 tar 文件，开始扫描...")

    for idx, tar_path in enumerate(tar_files, 1):
        symbols = scan_one_tar(tar_path)

        if not symbols:
            empty_tars.append(tar_path)
            print(f"[{idx}/{len(tar_files)}] 空 tar: {tar_path}")
            continue

        for sym in sorted(symbols):
            if sym in symbol_to_tar and symbol_to_tar[sym] != tar_path:
                duplicate_symbols[sym].append(tar_path)
            else:
                symbol_to_tar[sym] = tar_path

        if idx % 20 == 0 or idx == len(tar_files):
            print(f"[{idx}/{len(tar_files)}] 已扫描完成")

    if duplicate_symbols:
        print("\n发现 symbol 出现在多个 tar 中：")
        for sym, extra_tars in sorted(duplicate_symbols.items())[:20]:
            all_tars = [symbol_to_tar[sym]] + extra_tars
            print(f"  {sym}:")
            for t in all_tars:
                print(f"    - {t}")
        raise RuntimeError(
            f"共有 {len(duplicate_symbols)} 个 symbol 出现在多个 tar 中，请先检查打包是否重复。"
        )

    if empty_tars:
        print(f"\n警告：发现 {len(empty_tars)} 个空 tar。示例：")
        for t in empty_tars[:10]:
            print(f"  - {t}")

    print(f"\n成功建立映射，共 {len(symbol_to_tar)} 个 symbol")
    return symbol_to_tar


def main():
    parser = argparse.ArgumentParser(description="从 tar 包建立 symbol -> tar_path 映射")
    parser.add_argument(
        "--tar_dir",
        type=str,
        required=True,
        help="存放 tar 文件的目录，例如 /workspace/data/N5_tars",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="输出 json 路径，例如 /workspace/data/symbol_to_tar_N5.json",
    )
    args = parser.parse_args()

    mapping = build_symbol_to_tar(args.tar_dir)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"\n已保存到: {out_path}")


if __name__ == "__main__":
    main()


"""
python build_symbol_to_tar.py \
  --tar_dir /workspace/ohlc_image_N60/packs \
  --out_json /workspace/symbol_to_tar/symbol_to_tar_N60.json
"""