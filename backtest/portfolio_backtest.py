from __future__ import annotations

import os
import math
import argparse
from pathlib import Path
from typing import List, Optional
import bisect
import numpy as np
import pandas as pd

# 对模型预测结果进行回测分析，构建基于 decile 的投资组合，计算收益和夏普比率等指标。



def normalize_symbol(x: object) -> str:
    """
    统一证券代码为 6 位字符串。
    """
    s = str(x).strip()
    if "." in s:
        s = s.split(".")[0]
    return s.zfill(6)


def load_prediction_csv(pred_csv: str) -> pd.DataFrame:
    """
    读取模型预测结果。
    期望至少包含:
        - symbol
        - end_date
        - prob_up
    """
    df = pd.read_csv(pred_csv)

    required = {"symbol", "end_date", "prob_up"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pred csv missing columns: {missing}")

    df = df.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["prob_up"] = df["prob_up"].astype(float)

    return df[["symbol", "end_date", "prob_up"]].drop_duplicates()


def discover_csv_files(root_dir: str) -> List[str]:
    """
    递归找到 root_dir 下所有 csv。
    """
    root = Path(root_dir)
    files = sorted(str(p) for p in root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No csv files found under: {root_dir}")
    return files


def load_market_data(
    csv_root: str,
    weighting: str,
    cap_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    读取并拼接所有市场数据 csv。

    默认使用这些列：
        TradingDate, Symbol, OpenPrice, ClosePrice
    如果 weighting == value，则还需要 cap_col
    """
    files = discover_csv_files(csv_root)

    usecols = ["TradingDate", "Symbol", "OpenPrice", "ClosePrice"]
    if weighting == "value":
        if not cap_col:
            raise ValueError("weighting=value 时必须提供 --cap_col")
        usecols.append(cap_col)

    dfs = []
    for f in files:
        print(f"loading: {f}")
        df = pd.read_csv(f, usecols=usecols)
        dfs.append(df)

    mkt = pd.concat(dfs, axis=0, ignore_index=True)

    mkt["symbol"] = mkt["Symbol"].map(normalize_symbol)
    mkt["date"] = pd.to_datetime(mkt["TradingDate"])

    mkt["OpenPrice"] = pd.to_numeric(mkt["OpenPrice"], errors="coerce")
    mkt["ClosePrice"] = pd.to_numeric(mkt["ClosePrice"], errors="coerce")

    if weighting == "value":
        mkt[cap_col] = pd.to_numeric(mkt[cap_col], errors="coerce")

    keep_cols = ["symbol", "date", "OpenPrice", "ClosePrice"]
    if weighting == "value":
        keep_cols.append(cap_col)

    mkt = mkt[keep_cols].dropna(subset=["symbol", "date", "OpenPrice", "ClosePrice"])

    # 去重后按股票和日期排序
    mkt = (
        mkt.sort_values(["symbol", "date"])
           .drop_duplicates(subset=["symbol", "date"], keep="last")
           .reset_index(drop=True)
    )

    return mkt

def build_global_calendar(mkt: pd.DataFrame) -> pd.DatetimeIndex:
    """
    用所有股票出现过的 date 去重排序，构造全市场 trading calendar。
    """
    cal = pd.DatetimeIndex(sorted(mkt["date"].dropna().unique()))
    return cal

def build_forward_returns_one_symbol(
    g: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    horizon_days: int,
    weighting: str,
    cap_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    对单只股票构造未来收益，严格按：
      - 当前样本日 = t
      - 开仓参考点 = 全市场 calendar 上 t 之后的第 1 个位置
      - 平仓参考点 = 全市场 calendar 上 t 之后的第 horizon_days 个位置
      - 从这些参考点开始，对该股票 forward-search 第一个有效交易日

    收益定义：
      fwd_ret = exit_close / entry_open - 1

    这里与原先的 shift 不同，完全基于全市场 calendar。
    """
    g = g.sort_values("date").copy()
    g = g.drop_duplicates(subset=["date"], keep="last")

    # 先 reindex 到全市场 calendar
    g = g.set_index("date").reindex(calendar)
    g.index.name = "date"
    g = g.reset_index()

    # 当前股票哪些行是有效交易日（至少 open/close 有值）
    valid_mask = g["OpenPrice"].notna() & g["ClosePrice"].notna()
    valid_pos = np.flatnonzero(valid_mask.values)

    # 为 forward-search 准备：从某个位置 start_pos 开始，找到第一个有效位置
    def next_valid_pos(start_pos: int) -> Optional[int]:
        k = bisect.bisect_left(valid_pos, start_pos)
        if k >= len(valid_pos):
            return None
        return int(valid_pos[k])

    entry_dates = []
    entry_opens = []
    exit_dates = []
    exit_closes = []
    fwd_rets = []
    weight_caps = []

    n = len(g)

    for i in range(n):
        # 当前行如果本身都不是这只股票的有效交易日，就不作为 formation date
        if not valid_mask.iloc[i]:
            entry_dates.append(pd.NaT)
            entry_opens.append(np.nan)
            exit_dates.append(pd.NaT)
            exit_closes.append(np.nan)
            fwd_rets.append(np.nan)
            if weighting == "value":
                weight_caps.append(np.nan)
            continue

        entry_ref = i + 1
        exit_ref = i + horizon_days

        if entry_ref >= n or exit_ref >= n:
            entry_dates.append(pd.NaT)
            entry_opens.append(np.nan)
            exit_dates.append(pd.NaT)
            exit_closes.append(np.nan)
            fwd_rets.append(np.nan)
            if weighting == "value":
                weight_caps.append(np.nan)
            continue

        entry_pos = next_valid_pos(entry_ref)
        exit_pos = next_valid_pos(exit_ref)

        if entry_pos is None or exit_pos is None:
            entry_dates.append(pd.NaT)
            entry_opens.append(np.nan)
            exit_dates.append(pd.NaT)
            exit_closes.append(np.nan)
            fwd_rets.append(np.nan)
            if weighting == "value":
                weight_caps.append(np.nan)
            continue

        entry_date = g.loc[entry_pos, "date"]
        entry_open = g.loc[entry_pos, "OpenPrice"]
        exit_date = g.loc[exit_pos, "date"]
        exit_close = g.loc[exit_pos, "ClosePrice"]

        if pd.isna(entry_open) or pd.isna(exit_close):
            entry_dates.append(pd.NaT)
            entry_opens.append(np.nan)
            exit_dates.append(pd.NaT)
            exit_closes.append(np.nan)
            fwd_rets.append(np.nan)
            if weighting == "value":
                weight_caps.append(np.nan)
            continue

        fwd_ret = exit_close / entry_open - 1.0

        entry_dates.append(entry_date)
        entry_opens.append(float(entry_open))
        exit_dates.append(exit_date)
        exit_closes.append(float(exit_close))
        fwd_rets.append(float(fwd_ret))

        if weighting == "value":
            # 用实际开仓日的市值做权重
            cap_val = g.loc[entry_pos, cap_col] if cap_col in g.columns else np.nan
            weight_caps.append(float(cap_val) if pd.notna(cap_val) else np.nan)

    out = pd.DataFrame(
        {
            "symbol": g["symbol"].values,
            "date": g["date"].values,
            "entry_date": entry_dates,
            "entry_open": entry_opens,
            "exit_date": exit_dates,
            "exit_close": exit_closes,
            "fwd_ret": fwd_rets,
        }
    )

    if weighting == "value":
        out["weight_cap"] = weight_caps

    # 只保留原本这只股票真实存在的 date 行，避免把纯 reindex 出来的空日期也保留下来
    valid_original = valid_mask.values
    out = out.loc[valid_original].reset_index(drop=True)

    return out

def build_forward_returns(
    mkt: pd.DataFrame,
    horizon_days: int,
    weighting: str,
    cap_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    基于全市场 trading calendar 构造 forward returns。
    """
    calendar = build_global_calendar(mkt)
    print(f"global calendar length = {len(calendar)}")

    parts = []
    grouped = mkt.groupby("symbol", sort=False)

    for idx, (sym, g) in enumerate(grouped, start=1):
        part = build_forward_returns_one_symbol(
            g=g,
            calendar=calendar,
            horizon_days=horizon_days,
            weighting=weighting,
            cap_col=cap_col,
        )
        parts.append(part)

        if idx % 500 == 0:
            print(f"processed symbols: {idx}")

    out = pd.concat(parts, axis=0, ignore_index=True)

    keep_cols = ["symbol", "date", "entry_date", "entry_open", "exit_date", "exit_close", "fwd_ret"]
    if weighting == "value":
        keep_cols.append("weight_cap")

    return out[keep_cols]

# def build_forward_returns(
#     mkt: pd.DataFrame,
#     horizon_days: int,
#     weighting: str,
#     cap_col: Optional[str] = None,
# ) -> pd.DataFrame:
#     """
#     对每只股票构建：
#       - entry_date: 下一有效交易日
#       - entry_open: 下一有效交易日开盘价
#       - exit_date: 从当前样本往后第 horizon_days 个有效交易日
#       - exit_close: 第 horizon_days 个有效交易日收盘价
#       - fwd_ret: 从 next open 到 horizon close 的收益

#     对于当前行 t：
#       entry = t+1 的开盘价
#       exit  = t+h 的收盘价
#       fwd_ret = Close[t+h] / Open[t+1] - 1
#     """
#     def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
#         g = g.sort_values("date").copy()

#         g["entry_date"] = g["date"].shift(-1)
#         g["entry_open"] = g["OpenPrice"].shift(-1)

#         g["exit_date"] = g["date"].shift(-horizon_days)
#         g["exit_close"] = g["ClosePrice"].shift(-horizon_days)

#         g["fwd_ret"] = g["exit_close"] / g["entry_open"] - 1.0

#         if weighting == "value":
#             # 用下一交易日的流通市值作为建仓权重，更符合“下一交易日开仓”的时点逻辑
#             g["weight_cap"] = g[cap_col].shift(-1)

#         return g

#     out = (
#         mkt.groupby("symbol", group_keys=False)
#            .apply(_per_symbol)
#            .reset_index(drop=True)
#     )

#     keep_cols = ["symbol", "date", "entry_date", "entry_open", "exit_date", "exit_close", "fwd_ret"]
#     if weighting == "value":
#         keep_cols.append("weight_cap")

#     return out[keep_cols]


def assign_deciles(x: pd.Series) -> pd.Series:
    """
    对单日 prob_up 截面分成 10 组，1=Low, 10=High。
    用 rank(method='first') 避免 qcut 因重复值报错。
    """
    n = len(x)
    if n < 10:
        return pd.Series([np.nan] * n, index=x.index)

    ranks = x.rank(method="first")
    deciles = pd.qcut(ranks, 10, labels=False) + 1
    return deciles.astype(float)


def compute_portfolio_returns(
    merged: pd.DataFrame,
    weighting: str,
) -> pd.DataFrame:
    """
    对每个 end_date 的每个 decile 计算组合收益。
    输出列：
        end_date, decile, n_stocks, port_ret
    """
    rows = []

    for dt, g in merged.groupby("end_date"):
        g = g.dropna(subset=["prob_up", "fwd_ret", "decile"]).copy()
        if len(g) == 0:
            continue

        for decile, gg in g.groupby("decile"):
            if len(gg) == 0:
                continue

            if weighting == "equal":
                port_ret = gg["fwd_ret"].mean()
            elif weighting == "value":
                gg = gg.dropna(subset=["weight_cap"])
                if len(gg) == 0:
                    continue
                w = gg["weight_cap"].values.astype(float)
                if np.any(w < 0) or w.sum() <= 0:
                    continue
                w = w / w.sum()
                port_ret = np.sum(w * gg["fwd_ret"].values)
            else:
                raise ValueError(f"unknown weighting: {weighting}")

            rows.append(
                {
                    "end_date": dt,
                    "decile": int(decile),
                    "n_stocks": int(len(gg)),
                    "port_ret": float(port_ret),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No portfolio returns generated. Check merge, dates, or decile assignment.")
    return out.sort_values(["end_date", "decile"]).reset_index(drop=True)


def summarize_deciles(
    port_rets: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """
    生成 Table I 风格的汇总：
      - 每个 decile 的平均收益 Ret
      - annualized Sharpe SR
      - H-L

    这里 Ret 是每个 formation date 的 holding-period return 的时间序列平均。
    SR 用 sqrt(252 / horizon_days) 年化。
    """
    ann_factor = math.sqrt(252.0 / horizon_days)

    rows = []

    for decile in range(1, 11):
        s = port_rets.loc[port_rets["decile"] == decile, "port_ret"].dropna()
        if len(s) == 0:
            continue

        mean_ret = s.mean()
        std_ret = s.std(ddof=1)

        sr = np.nan
        if std_ret is not None and std_ret > 0:
            sr = (mean_ret / std_ret) * ann_factor

        rows.append(
            {
                "group": "Low" if decile == 1 else ("High" if decile == 10 else str(decile)),
                "Ret": mean_ret,
                "SR": sr,
                "n_periods": int(len(s)),
            }
        )

    # H-L
    wide = port_rets.pivot(index="end_date", columns="decile", values="port_ret")
    if 1 in wide.columns and 10 in wide.columns:
        hl = (wide[10] - wide[1]).dropna()
        if len(hl) > 0:
            mean_ret = hl.mean()
            std_ret = hl.std(ddof=1)
            sr = np.nan
            if std_ret is not None and std_ret > 0:
                sr = (mean_ret / std_ret) * ann_factor

            rows.append(
                {
                    "group": "H-L",
                    "Ret": mean_ret,
                    "SR": sr,
                    "n_periods": int(len(hl)),
                }
            )

    summary = pd.DataFrame(rows)

    # 为了更接近论文表格，额外给一个百分数版本
    summary["Ret_pct"] = summary["Ret"] * 100.0

    return summary


def main():
    parser = argparse.ArgumentParser(description="Build decile portfolios from model probabilities.")

    parser.add_argument("--pred_csv", type=str, required=True, help="模型预测结果 csv，至少含 symbol/end_date/prob_up")
    parser.add_argument("--csv_root", type=str, required=True, help="原始后复权日行情 csv 根目录")
    parser.add_argument("--horizon_days", type=int, required=True, choices=[5, 20, 60], help="持有期天数")
    parser.add_argument("--weighting", type=str, default="equal", choices=["equal", "value"], help="等权或流通市值加权")
    parser.add_argument("--cap_col", type=str, default="CirculatedMarketValue", help="value-weight 时使用的市值列名")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== step 1: load predictions =====")
    pred = load_prediction_csv(args.pred_csv)
    print(f"pred rows = {len(pred)}")

    print("\n===== step 2: load market data =====")
    mkt = load_market_data(
        csv_root=args.csv_root,
        weighting=args.weighting,
        cap_col=args.cap_col,
    )
    print(f"market rows = {len(mkt)}")

    print("\n===== step 3: build forward returns =====")
    fwd = build_forward_returns(
        mkt=mkt,
        horizon_days=args.horizon_days,
        weighting=args.weighting,
        cap_col=args.cap_col,
    )
    print(f"forward-return rows = {len(fwd)}")

    print("\n===== step 4: merge predictions with market data =====")
    merged = pred.merge(
        fwd,
        how="left",
        left_on=["symbol", "end_date"],
        right_on=["symbol", "date"],
    )

    # 保留能真正形成交易的样本
    merged = merged.dropna(subset=["entry_date", "entry_open", "exit_date", "exit_close", "fwd_ret"]).copy()
    print(f"merged usable rows = {len(merged)}")

    print("\n===== step 5: assign daily deciles =====")
    merged["decile"] = (
        merged.groupby("end_date")["prob_up"]
              .transform(assign_deciles)
    )
    merged = merged.dropna(subset=["decile"]).copy()
    merged["decile"] = merged["decile"].astype(int)
    print(f"rows after decile assignment = {len(merged)}")

    print("\n===== step 6: compute portfolio returns =====")
    port_rets = compute_portfolio_returns(
        merged=merged,
        weighting=args.weighting,
    )
    print(f"portfolio-return rows = {len(port_rets)}")

    print("\n===== step 7: summarize =====")
    summary = summarize_deciles(
        port_rets=port_rets,
        horizon_days=args.horizon_days,
    )

    merged_out = out_dir / "merged_predictions_with_returns.csv"
    port_out = out_dir / f"portfolio_returns_{args.weighting}.csv"
    summary_out = out_dir / f"summary_{args.weighting}.csv"

    merged.to_csv(merged_out, index=False, encoding="utf-8-sig")
    port_rets.to_csv(port_out, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print("\n===== done =====")
    print(f"saved merged table   -> {merged_out}")
    print(f"saved portfolio ts   -> {port_out}")
    print(f"saved summary table  -> {summary_out}")

    print("\n===== preview =====")
    print(summary)


if __name__ == "__main__":
    main()

"""
python portfolio_backtest.py \
  --pred_csv /workspace/predict/preds_I60_R5.csv \
  --csv_root /workspace/stock_data_csv \
  --horizon_days 5 \
  --weighting equal \
  --out_dir /workspace/backtest_I60_R5_equal
"""

"""
python portfolio_backtest.py \
  --pred_csv /workspace/predict/preds_I60_R5.csv \
  --csv_root /workspace/stock_data_csv \
  --horizon_days 5 \
  --weighting value \
  --out_dir /workspace/backtest_I60_R5_value
"""