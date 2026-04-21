from __future__ import annotations

from typing import Optional

# 统一的板块标签
BOARD_SH_MAIN = "sh_main"
BOARD_SZ_MAIN = "sz_main"
BOARD_CHINEXT = "chinext"
BOARD_STAR = "star"
BOARD_BSE = "bse"
BOARD_UNKNOWN = "unknown"

BOARD_ORDER = [
    BOARD_SH_MAIN,
    BOARD_SZ_MAIN,
    BOARD_CHINEXT,
    BOARD_STAR,
    BOARD_BSE,
]


def normalize_symbol(x: object) -> str:
    s = str(x).strip()
    if "." in s:
        s = s.split(".")[0]
    return s.zfill(6)


def infer_board_from_symbol(symbol: str) -> str:
    """
    按“当前证券代码前缀”静态划分板块，不追踪历史转板。

    约定：
    - 上海主板: 600 / 601 / 603 / 605
    - 科创板:   688
    - 深圳主板: 000 / 001 / 002 / 003
      注：这里把 002 也并入当前“深圳主板”口径，便于和创业板分开评估
    - 创业板:   300 / 301
    - 北交所:   4* / 8* / 92*
      注：这是代码静态近似口径；不额外处理精选层/转板历史
    """
    s = normalize_symbol(symbol)

    if s.startswith(("688",)):
        return BOARD_STAR

    if s.startswith(("600", "601", "603", "605")):
        return BOARD_SH_MAIN

    if s.startswith(("300", "301")):
        return BOARD_CHINEXT

    if s.startswith(("000", "001", "002", "003")):
        return BOARD_SZ_MAIN

    if s.startswith(("4", "8", "92")):
        return BOARD_BSE

    return BOARD_UNKNOWN


def board_to_suffix(board: str) -> str:
    return board
