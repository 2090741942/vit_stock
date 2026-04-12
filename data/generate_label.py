from __future__ import annotations

import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================
# 配置区
# =========================

DATA_ROOT = r"/workspace/stock_data_csv"
OUT_DIR = r"/workspace/label_npz"

WINDOWS = (5, 20, 60)
HORIZONS = (5, 20, 60)

SYMBOL_COL = "Symbol"
DATE_COL = "TradingDate"
PRICE_COLS = ("OpenPrice", "HighPrice", "LowPrice", "ClosePrice")
VOLUME_COL = "Volume"

SEED = 20260315
TEST_START_DATE = "2024-01-31"

SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2

# True 更省磁盘，False 更快
USE_COMPRESSED_SAVE = True

# Windows 下建议别开太大，4~8 通常够了
# MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 8))
MAX_WORKERS = 32

# 每个任务块包含多少只股票；太小会调度开销大，太大内存峰值高
SYMBOLS_PER_CHUNK = 512


# =========================
# 基础工具
# =========================

def load_all_csvs(data_root: str) -> pd.DataFrame:
    """
    一次性读取所有 CSV，只读必要列。
    """
    pattern = os.path.join(data_root, "**", "TRD_BwardQuotation*.csv")
    csv_files = sorted(p for p in glob.glob(pattern, recursive=True) if "[DES]" not in p)
    if not csv_files:
        raise FileNotFoundError(f"No csv files matched: {pattern}")

    usecols = [SYMBOL_COL, DATE_COL, *PRICE_COLS, VOLUME_COL]
    dtype = {
        SYMBOL_COL: str,
        PRICE_COLS[0]: "float64",
        PRICE_COLS[1]: "float64",
        PRICE_COLS[2]: "float64",
        PRICE_COLS[3]: "float64",
        VOLUME_COL: "float64",
    }

    frames = []
    for p in tqdm(csv_files, desc="Reading CSVs"):
        df_i = pd.read_csv(
            p,
            usecols=usecols,
            dtype=dtype,
            low_memory=False,
        )
        frames.append(df_i)

    df = pd.concat(frames, ignore_index=True)
    df[SYMBOL_COL] = df[SYMBOL_COL].astype(str).str.zfill(6)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, SYMBOL_COL]).sort_values([SYMBOL_COL, DATE_COL], kind="mergesort")
    return df


def make_split_array(end_dates: np.ndarray, test_start_date: str, seed: int) -> np.ndarray:
    """
    split:
      0 -> train
      1 -> val
      2 -> test
    """
    rng = np.random.default_rng(seed)
    split = np.full(len(end_dates), SPLIT_TEST, dtype=np.int8)

    test_start_int = int(pd.Timestamp(test_start_date).strftime("%Y%m%d"))
    end_date_int = end_dates.astype(np.int32)

    train_val_mask = end_date_int < test_start_int
    train_val_idx = np.flatnonzero(train_val_mask)

    split[train_val_idx] = SPLIT_TRAIN

    n_train_val = len(train_val_idx)
    n_val = int(round(n_train_val * 0.30))
    if n_val > 0:
        choose_local = rng.choice(n_train_val, size=n_val, replace=False)
        val_idx = train_val_idx[choose_local]
        split[val_idx] = SPLIT_VAL

    return split


def split_chunks(seq: List, chunk_size: int) -> List[List]:
    return [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]


# =========================
# 核心数值逻辑（全 numpy）
# =========================

def align_symbol_to_calendar(
    pos_idx: np.ndarray,
    o_raw: np.ndarray,
    h_raw: np.ndarray,
    l_raw: np.ndarray,
    c_raw: np.ndarray,
    v_raw: np.ndarray,
    calendar_len: int,
) -> Dict[str, np.ndarray]:
    """
    把单只股票对齐到全市场 trading calendar。
    返回：
      - 原始 close（用于 label）
      - 按图片逻辑处理后的 O/H/L/C/V（用于判断图片是否存在）
    """
    # 原始 close：只 reindex，不做 no_trade -> NaN
    close_raw_aligned = np.full(calendar_len, np.nan, dtype=np.float64)
    close_raw_aligned[pos_idx] = c_raw

    # 图片判断用对齐数组
    O = np.full(calendar_len, np.nan, dtype=np.float64)
    H = np.full(calendar_len, np.nan, dtype=np.float64)
    L = np.full(calendar_len, np.nan, dtype=np.float64)
    C = np.full(calendar_len, np.nan, dtype=np.float64)
    V = np.full(calendar_len, np.nan, dtype=np.float64)

    O[pos_idx] = o_raw
    H[pos_idx] = h_raw
    L[pos_idx] = l_raw
    C[pos_idx] = c_raw
    V[pos_idx] = v_raw

    # no_trade: volume<=0 且 O=H=L=C
    v0 = np.isfinite(V) & (V <= 0)
    eq_ohlc = np.isfinite(O) & np.isfinite(H) & np.isfinite(L) & np.isfinite(C) & (O == H) & (H == L) & (L == C)
    no_trade = v0 & eq_ohlc

    O[no_trade] = np.nan
    H[no_trade] = np.nan
    L[no_trade] = np.nan
    C[no_trade] = np.nan
    V[no_trade] = np.nan

    return {
        "raw_close": close_raw_aligned,
        "O": O,
        "H": H,
        "L": L,
        "C": C,
        "V": V,
    }


def compute_labels_forward_search(raw_close: np.ndarray, horizons: Iterable[int]) -> np.ndarray:
    """
    对位置 i、horizon=h：
    从 i+h 开始向后找第一个有效 raw close，若存在则算方向标签，否则 -1。
    """
    horizons = list(horizons)
    T = len(raw_close)
    labels = np.full((T, len(horizons)), -1, dtype=np.int8)

    valid = np.isfinite(raw_close) & (raw_close > 0)

    next_valid_from = np.full(T, -1, dtype=np.int64)
    nxt = -1
    for i in range(T - 1, -1, -1):
        if valid[i]:
            nxt = i
        next_valid_from[i] = nxt

    base_idx = np.arange(T, dtype=np.int64)

    for j, h in enumerate(horizons):
        target = base_idx + h
        ok = target < T

        fut_pos = np.full(T, -1, dtype=np.int64)
        fut_pos[ok] = next_valid_from[target[ok]]

        good = valid & (fut_pos != -1)
        if not np.any(good):
            continue

        ret = raw_close[fut_pos[good]] / raw_close[good] - 1.0
        labels[good, j] = (ret > 0).astype(np.int8)

    return labels


def window_image_exists_numpy(
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    C: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    返回长度 T 的 bool 数组：
      valid_end[e] = True 表示以 e 为结束位置、长度 n 的窗口会生成图片。
    逻辑与 build_window_image 一致，但全用 numpy。
    """
    T = len(C)
    valid_end = np.zeros(T, dtype=bool)
    if T < n:
        return valid_end

    for e in range(n - 1, T):
        s = e - n + 1

        c0 = C[s]
        c_last = C[e]
        if not (np.isfinite(c0) and c0 > 0 and np.isfinite(c_last) and c_last > 0):
            continue

        O_w = O[s:e + 1]
        H_w = H[s:e + 1]
        L_w = L[s:e + 1]
        C_w = C[s:e + 1]

        # p[0] = 1.0
        p = np.full(n, np.nan, dtype=np.float64)
        p[0] = 1.0
        last_close = c0
        last_p = 1.0

        for i in range(1, n):
            c_now = C_w[i]
            if np.isfinite(c_now) and c_now > 0 and np.isfinite(last_close) and last_close > 0:
                ret = c_now / last_close - 1.0
                last_p = (1.0 + ret) * last_p
                p[i] = last_p
                last_close = c_now

        ratio_OC = np.divide(O_w, C_w, out=np.full(n, np.nan), where=np.isfinite(O_w) & np.isfinite(C_w) & (C_w != 0))
        ratio_HC = np.divide(H_w, C_w, out=np.full(n, np.nan), where=np.isfinite(H_w) & np.isfinite(C_w) & (C_w != 0))
        ratio_LC = np.divide(L_w, C_w, out=np.full(n, np.nan), where=np.isfinite(L_w) & np.isfinite(C_w) & (C_w != 0))

        o = p * ratio_OC
        h = p * ratio_HC
        l = p * ratio_LC
        c = p

        # rolling mean(min_periods=1)
        ma = np.full(n, np.nan, dtype=np.float64)
        valid_c = np.isfinite(c)
        run_sum = 0.0
        run_cnt = 0
        for i in range(n):
            if valid_c[i]:
                run_sum += c[i]
                run_cnt += 1
            ma[i] = run_sum / run_cnt if run_cnt > 0 else np.nan
            if not valid_c[i]:
                ma[i] = np.nan

        arr = np.concatenate([o, h, l, c, ma])
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue

        vmax = finite.max()
        vmin = finite.min()
        if np.isfinite(vmax) and np.isfinite(vmin) and (vmax > vmin):
            valid_end[e] = True

    return valid_end


# =========================
# 单个 chunk 的 worker
# =========================

def process_symbol_chunk(
    chunk_items: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    calendar_len: int,
    calendar_dates_str: np.ndarray,
    windows: Tuple[int, ...],
    horizons: Tuple[int, ...],
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    输入一批股票的原始数组。
    返回按 window 聚合的结果：
      result[n] = (labels_block, symbol_block, end_date_block)
    """
    out: Dict[int, List] = {n: [[], [], []] for n in windows}

    for sym, pos_idx, o_raw, h_raw, l_raw, c_raw, v_raw in chunk_items:
        aligned = align_symbol_to_calendar(
            pos_idx=pos_idx,
            o_raw=o_raw,
            h_raw=h_raw,
            l_raw=l_raw,
            c_raw=c_raw,
            v_raw=v_raw,
            calendar_len=calendar_len,
        )

        raw_close = aligned["raw_close"]
        O = aligned["O"]
        H = aligned["H"]
        L = aligned["L"]
        C = aligned["C"]

        labels_all = compute_labels_forward_search(raw_close, horizons=horizons)

        for n in windows:
            valid_end = window_image_exists_numpy(O, H, L, C, n=n)
            end_idx = np.flatnonzero(valid_end)
            if end_idx.size == 0:
                continue

            out[n][0].append(labels_all[end_idx].astype(np.int8, copy=False))
            out[n][1].append(np.full(end_idx.size, sym, dtype="<U6"))
            out[n][2].append(calendar_dates_str[end_idx])

    ret: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for n in windows:
        if len(out[n][0]) == 0:
            ret[n] = (
                np.empty((0, len(horizons)), dtype=np.int8),
                np.empty((0,), dtype="<U6"),
                np.empty((0,), dtype="<U8"),
            )
        else:
            ret[n] = (
                np.concatenate(out[n][0], axis=0),
                np.concatenate(out[n][1], axis=0),
                np.concatenate(out[n][2], axis=0),
            )
    return ret


# =========================
# 主流程
# =========================

def prepare_symbol_items(df: pd.DataFrame, trading_calendar: pd.DatetimeIndex) -> List[
    Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    把 DataFrame 预拆成按股票的 numpy 任务，避免 worker 里再做 groupby/pandas。
    """
    date_to_pos = {d: i for i, d in enumerate(trading_calendar)}

    items = []
    for sym, g in tqdm(df.groupby(SYMBOL_COL, sort=False), desc="Preparing symbol arrays"):
        dates = g[DATE_COL].to_numpy()
        pos_idx = np.fromiter((date_to_pos[d] for d in dates), dtype=np.int64, count=len(dates))

        items.append((
            sym,
            pos_idx,
            g[PRICE_COLS[0]].to_numpy(dtype=np.float64, copy=False),
            g[PRICE_COLS[1]].to_numpy(dtype=np.float64, copy=False),
            g[PRICE_COLS[2]].to_numpy(dtype=np.float64, copy=False),
            g[PRICE_COLS[3]].to_numpy(dtype=np.float64, copy=False),
            g[VOLUME_COL].to_numpy(dtype=np.float64, copy=False),
        ))
    return items


def save_npz(
    save_path: str,
    labels: np.ndarray,
    symbols: np.ndarray,
    end_dates: np.ndarray,
    split: np.ndarray,
    horizons: Tuple[int, ...],
    window: int,
    seed: int,
    test_start_date: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    saver = np.savez_compressed if USE_COMPRESSED_SAVE else np.savez

    saver(
        save_path,
        labels=labels,
        symbol=symbols,
        end_date=end_dates,
        split=split,
        horizons=np.array(horizons, dtype=np.int16),
        window=np.array([window], dtype=np.int16),
        seed=np.array([seed], dtype=np.int64),
        test_start_date=np.array([str(test_start_date)], dtype="<U10"),
    )


def main():
    df = load_all_csvs(DATA_ROOT)

    trading_calendar = pd.DatetimeIndex(sorted(df[DATE_COL].dropna().unique()))
    calendar_dates_str = trading_calendar.strftime("%Y%m%d").to_numpy(dtype="<U8")

    items = prepare_symbol_items(df, trading_calendar)
    chunks = split_chunks(items, SYMBOLS_PER_CHUNK)

    collected: Dict[int, List] = {n: [[], [], []] for n in WINDOWS}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(
                process_symbol_chunk,
                chunk,
                len(trading_calendar),
                calendar_dates_str,
                WINDOWS,
                HORIZONS,
            )
            for chunk in chunks
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            res = fut.result()
            for n in WINDOWS:
                lab, symb, ed = res[n]
                if lab.shape[0] == 0:
                    continue
                collected[n][0].append(lab)
                collected[n][1].append(symb)
                collected[n][2].append(ed)

    for n in WINDOWS:
        if len(collected[n][0]) == 0:
            raise RuntimeError(f"No valid rows produced for N={n}")

        labels = np.concatenate(collected[n][0], axis=0).astype(np.int8, copy=False)
        symbols = np.concatenate(collected[n][1], axis=0)
        end_dates = np.concatenate(collected[n][2], axis=0)

        split = make_split_array(
            end_dates=end_dates,
            test_start_date=TEST_START_DATE,
            seed=SEED,
        )

        save_path = os.path.join(OUT_DIR, f"meta_N{n}.npz")
        save_npz(
            save_path=save_path,
            labels=labels,
            symbols=symbols,
            end_dates=end_dates,
            split=split,
            horizons=HORIZONS,
            window=n,
            seed=SEED,
            test_start_date=TEST_START_DATE,
        )

        n_train = int(np.sum(split == SPLIT_TRAIN))
        n_val = int(np.sum(split == SPLIT_VAL))
        n_test = int(np.sum(split == SPLIT_TEST))

        print(f"[OK] saved to {save_path}")
        print(f"labels.shape = {labels.shape}")
        print(f"split counts: train={n_train}, val={n_val}, test={n_test}")
        for col, h in enumerate(HORIZONS):
            valid_cnt = int(np.sum(labels[:, col] != -1))
            print(f"h={h} valid labels: {valid_cnt}")


if __name__ == "__main__":
    main()