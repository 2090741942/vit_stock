from __future__ import annotations

import math
import argparse
from pathlib import Path
from typing import List, Optional
import bisect

import numpy as np
import pandas as pd


def normalize_symbol(x: object) -> str:
    s = str(x).strip()
    if "." in s:
        s = s.split(".")[0]
    return s.zfill(6)


def load_prediction_csv(pred_csv: str) -> pd.DataFrame:
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
    files = discover_csv_files(csv_root)

    usecols = ["TradingDate", "Symbol", "OpenPrice", "ClosePrice"]
    if weighting == "value" or cap_col is not None:
        if not cap_col:
            raise ValueError("需要提供 --cap_col")
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

    if cap_col and cap_col in mkt.columns:
        mkt[cap_col] = pd.to_numeric(mkt[cap_col], errors="coerce")

    keep_cols = ["symbol", "date", "OpenPrice", "ClosePrice"]
    if cap_col and cap_col in mkt.columns:
        keep_cols.append(cap_col)

    mkt = mkt[keep_cols].dropna(subset=["symbol", "date", "OpenPrice", "ClosePrice"])
    mkt = (
        mkt.sort_values(["symbol", "date"])
           .drop_duplicates(subset=["symbol", "date"], keep="last")
           .reset_index(drop=True)
    )
    return mkt


def build_global_calendar(mkt: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(mkt["date"].dropna().unique()))


def build_forward_returns_one_symbol(
    g: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    horizon_days: int,
    weighting: str,
    cap_col: Optional[str] = None,
    buy_cost_rate: float = 0.0000641,
    sell_cost_rate: float = 0.0005641,
) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    g = g.drop_duplicates(subset=["date"], keep="last")

    g = g.set_index("date").reindex(calendar)
    g.index.name = "date"
    g = g.reset_index()

    valid_mask = g["OpenPrice"].notna() & g["ClosePrice"].notna()
    valid_pos = np.flatnonzero(valid_mask.values)

    def next_valid_pos(start_pos: int) -> Optional[int]:
        k = bisect.bisect_left(valid_pos, start_pos)
        if k >= len(valid_pos):
            return None
        return int(valid_pos[k])

    entry_dates = []
    entry_opens = []
    exit_dates = []
    exit_closes = []
    gross_fwd_rets = []
    net_fwd_rets = []
    total_cost_rates = []
    weight_caps = []
    formation_caps = []

    n = len(g)
    for i in range(n):
        if not valid_mask.iloc[i]:
            entry_dates.append(pd.NaT)
            entry_opens.append(np.nan)
            exit_dates.append(pd.NaT)
            exit_closes.append(np.nan)
            gross_fwd_rets.append(np.nan)
            net_fwd_rets.append(np.nan)
            total_cost_rates.append(np.nan)
            if cap_col:
                formation_caps.append(np.nan)
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
            gross_fwd_rets.append(np.nan)
            net_fwd_rets.append(np.nan)
            total_cost_rates.append(np.nan)
            if cap_col:
                formation_caps.append(np.nan)
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
            gross_fwd_rets.append(np.nan)
            net_fwd_rets.append(np.nan)
            total_cost_rates.append(np.nan)
            if cap_col:
                formation_caps.append(np.nan)
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
            gross_fwd_rets.append(np.nan)
            net_fwd_rets.append(np.nan)
            total_cost_rates.append(np.nan)
            if cap_col:
                formation_caps.append(np.nan)
            if weighting == "value":
                weight_caps.append(np.nan)
            continue

        gross_fwd_ret = exit_close / entry_open - 1.0
        net_fwd_ret = (exit_close * (1.0 - sell_cost_rate)) / (entry_open * (1.0 + buy_cost_rate)) - 1.0

        entry_dates.append(entry_date)
        entry_opens.append(float(entry_open))
        exit_dates.append(exit_date)
        exit_closes.append(float(exit_close))
        gross_fwd_rets.append(float(gross_fwd_ret))
        net_fwd_rets.append(float(net_fwd_ret))
        total_cost_rates.append(float(buy_cost_rate + sell_cost_rate))

        if cap_col:
            form_cap_val = g.loc[i, cap_col] if cap_col in g.columns else np.nan
            formation_caps.append(float(form_cap_val) if pd.notna(form_cap_val) else np.nan)

        if weighting == "value":
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
            "gross_fwd_ret": gross_fwd_rets,
            "net_fwd_ret": net_fwd_rets,
            "total_cost_rate": total_cost_rates,
        }
    )

    if cap_col:
        out["formation_cap"] = formation_caps
    if weighting == "value":
        out["weight_cap"] = weight_caps

    out = out.loc[valid_mask.values].reset_index(drop=True)
    return out


def build_forward_returns(
    mkt: pd.DataFrame,
    horizon_days: int,
    weighting: str,
    cap_col: Optional[str] = None,
    buy_cost_rate: float = 0.0000641,
    sell_cost_rate: float = 0.0005641,
) -> pd.DataFrame:
    calendar = build_global_calendar(mkt)
    print(f"global calendar length = {len(calendar)}")

    parts = []
    grouped = mkt.groupby("symbol", sort=False)
    for idx, (_, g) in enumerate(grouped, start=1):
        part = build_forward_returns_one_symbol(
            g=g,
            calendar=calendar,
            horizon_days=horizon_days,
            weighting=weighting,
            cap_col=cap_col,
            buy_cost_rate=buy_cost_rate,
            sell_cost_rate=sell_cost_rate,
        )
        parts.append(part)
        if idx % 500 == 0:
            print(f"processed symbols: {idx}")

    out = pd.concat(parts, axis=0, ignore_index=True)
    keep_cols = [
        "symbol", "date", "entry_date", "entry_open",
        "exit_date", "exit_close", "gross_fwd_ret",
        "net_fwd_ret", "total_cost_rate",
    ]
    if cap_col:
        keep_cols.append("formation_cap")
    if weighting == "value":
        keep_cols.append("weight_cap")
    return out[keep_cols]


def assign_deciles(x: pd.Series) -> pd.Series:
    n = len(x)
    if n < 10:
        return pd.Series([np.nan] * n, index=x.index)
    ranks = x.rank(method="first")
    deciles = pd.qcut(ranks, 10, labels=False) + 1
    return deciles.astype(float)


def filter_top_n_by_cap(
    merged: pd.DataFrame,
    top_n_by_cap: Optional[int],
    cap_col_name: str = "formation_cap",
) -> pd.DataFrame:
    if top_n_by_cap is None or top_n_by_cap <= 0:
        return merged

    if cap_col_name not in merged.columns:
        raise ValueError(f"top_n_by_cap requires column: {cap_col_name}")

    kept = []
    for dt, g in merged.groupby("end_date"):
        gg = g.dropna(subset=[cap_col_name]).copy()
        if len(gg) == 0:
            continue
        gg = gg.sort_values([cap_col_name, "symbol"], ascending=[False, True]).head(top_n_by_cap).copy()
        kept.append(gg)

    if not kept:
        raise RuntimeError("No rows left after top_n_by_cap filtering.")

    out = pd.concat(kept, axis=0, ignore_index=True)
    return out.sort_values(["end_date", "symbol"]).reset_index(drop=True)


def compute_portfolio_returns(
    merged: pd.DataFrame,
    weighting: str,
    ret_col: str,
) -> pd.DataFrame:
    rows = []

    for dt, g in merged.groupby("end_date"):
        g = g.dropna(subset=["prob_up", ret_col, "decile"]).copy()
        if len(g) == 0:
            continue

        for decile, gg in g.groupby("decile"):
            if len(gg) == 0:
                continue

            if weighting == "equal":
                port_ret = gg[ret_col].mean()
            elif weighting == "value":
                gg = gg.dropna(subset=["weight_cap"])
                if len(gg) == 0:
                    continue
                w = gg["weight_cap"].values.astype(float)
                if np.any(w < 0) or w.sum() <= 0:
                    continue
                w = w / w.sum()
                port_ret = float(np.sum(w * gg[ret_col].values))
            else:
                raise ValueError(f"unknown weighting: {weighting}")

            rows.append(
                {
                    "end_date": dt,
                    "decile": int(decile),
                    "port_ret": float(port_ret),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No portfolio returns generated.")
    return out.sort_values(["end_date", "decile"]).reset_index(drop=True)


def compute_holding_weights(
    merged: pd.DataFrame,
    weighting: str,
    drift_ret_col: str = "net_fwd_ret",
) -> pd.DataFrame:
    rows = []

    for dt, g in merged.groupby("end_date"):
        for decile, gg in g.groupby("decile"):
            if len(gg) == 0:
                continue

            if weighting == "equal":
                w = np.full(len(gg), 1.0 / len(gg), dtype=float)
            elif weighting == "value":
                gg = gg.dropna(subset=["weight_cap"]).copy()
                if len(gg) == 0:
                    continue
                raw = gg["weight_cap"].values.astype(float)
                if np.any(raw < 0) or raw.sum() <= 0:
                    continue
                w = raw / raw.sum()
            else:
                raise ValueError(f"unknown weighting: {weighting}")

            for (_, row), wi in zip(gg.iterrows(), w):
                rows.append(
                    {
                        "end_date": dt,
                        "decile": int(decile),
                        "symbol": row["symbol"],
                        "weight": float(wi),
                        "fwd_ret_for_drift": float(row[drift_ret_col]),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No holding weights generated.")
    return out.sort_values(["decile", "end_date", "symbol"]).reset_index(drop=True)


def compute_turnover(
    holdings: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    rows = []

    for decile, hd in holdings.groupby("decile"):
        hd = hd.sort_values(["end_date", "symbol"]).copy()
        dates = sorted(hd["end_date"].unique())

        for t_idx in range(len(dates) - 1):
            dt = dates[t_idx]
            dt_next = dates[t_idx + 1]

            cur = hd.loc[hd["end_date"] == dt, ["symbol", "weight", "fwd_ret_for_drift"]].copy()
            nxt = hd.loc[hd["end_date"] == dt_next, ["symbol", "weight"]].copy()

            if len(cur) == 0 or len(nxt) == 0:
                continue

            port_ret = float(np.sum(cur["weight"].values * cur["fwd_ret_for_drift"].values))
            denom = 1.0 + port_ret
            if denom <= 0:
                continue

            cur["drift_weight"] = cur["weight"] * (1.0 + cur["fwd_ret_for_drift"]) / denom

            merged_w = pd.merge(
                cur[["symbol", "drift_weight"]],
                nxt.rename(columns={"weight": "next_weight"}),
                on="symbol",
                how="outer",
            ).fillna(0.0)

            turnover_t = float(np.abs(merged_w["next_weight"] - merged_w["drift_weight"]).sum())
            rows.append(
                {
                    "decile": int(decile),
                    "monthly_turnover": turnover_t * (21.0 / horizon_days),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No turnover time series generated.")
    return out


def period_mean_to_annual_return(mean_period_ret: float, horizon_days: int) -> float:
    periods_per_year = 252.0 / horizon_days
    base = 1.0 + mean_period_ret
    if base <= 0:
        return np.nan
    return base ** periods_per_year - 1.0


def summarize_deciles_summary_only(
    port_rets: pd.DataFrame,
    horizon_days: int,
    turnover_ts: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    ann_factor = math.sqrt(252.0 / horizon_days)
    rows = []

    turnover_map = {}
    if turnover_ts is not None and len(turnover_ts) > 0:
        tmp = turnover_ts.groupby("decile")["monthly_turnover"].mean()
        turnover_map = {int(k): float(v) for k, v in tmp.items()}

    for decile in range(1, 11):
        s = port_rets.loc[port_rets["decile"] == decile, "port_ret"].dropna()
        if len(s) == 0:
            continue

        mean_ret = float(s.mean())
        std_ret = float(s.std(ddof=1))
        sr = np.nan
        if np.isfinite(std_ret) and std_ret > 0:
            sr = (mean_ret / std_ret) * ann_factor

        annual_ret = period_mean_to_annual_return(mean_ret, horizon_days)
        rows.append(
            {
                "group": "Low" if decile == 1 else ("High" if decile == 10 else str(decile)),
                "Ret": mean_ret,
                "AnnualRet": annual_ret,
                "SR": sr,
                "Turnover": turnover_map.get(decile, np.nan),
                "n_periods": int(len(s)),
            }
        )

    wide = port_rets.pivot(index="end_date", columns="decile", values="port_ret")
    if 1 in wide.columns and 10 in wide.columns:
        hl = (wide[10] - wide[1]).dropna()
        if len(hl) > 0:
            mean_ret = float(hl.mean())
            std_ret = float(hl.std(ddof=1))
            sr = np.nan
            if np.isfinite(std_ret) and std_ret > 0:
                sr = (mean_ret / std_ret) * ann_factor
            annual_ret = period_mean_to_annual_return(mean_ret, horizon_days)

            hl_turnover = np.nan
            if 1 in turnover_map and 10 in turnover_map:
                hl_turnover = turnover_map[1] + turnover_map[10]

            rows.append(
                {
                    "group": "H-L",
                    "Ret": mean_ret,
                    "AnnualRet": annual_ret,
                    "SR": sr,
                    "Turnover": hl_turnover,
                    "n_periods": int(len(hl)),
                }
            )

    summary = pd.DataFrame(rows)
    summary["Ret_pct"] = summary["Ret"] * 100.0
    summary["AnnualRet_pct"] = summary["AnnualRet"] * 100.0
    summary["Turnover_pct"] = summary["Turnover"] * 100.0
    return summary


def main():
    parser = argparse.ArgumentParser(description="Backtest that saves summary only (no daily files).")
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--csv_root", type=str, required=True)
    parser.add_argument("--horizon_days", type=int, required=True, choices=[5, 20, 60])
    parser.add_argument("--weighting", type=str, default="equal", choices=["equal", "value"])
    parser.add_argument("--cap_col", type=str, default="CirculatedMarketValue")
    parser.add_argument("--top_n_by_cap", type=int, default=0, help="<=0 means no top-N filter")
    parser.add_argument("--buy_cost_rate", type=float, default=0.0000641)
    parser.add_argument("--sell_cost_rate", type=float, default=0.0005641)
    parser.add_argument("--out_dir", type=str, required=True)
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
        buy_cost_rate=args.buy_cost_rate,
        sell_cost_rate=args.sell_cost_rate,
    )
    print(f"forward-return rows = {len(fwd)}")

    print("\n===== step 4: merge predictions with market data =====")
    merged = pred.merge(
        fwd,
        how="left",
        left_on=["symbol", "end_date"],
        right_on=["symbol", "date"],
    )
    merged = merged.dropna(
        subset=["entry_date", "entry_open", "exit_date", "exit_close", "gross_fwd_ret", "net_fwd_ret"]
    ).copy()
    print(f"merged usable rows = {len(merged)}")

    print("\n===== step 5: filter top-N by float cap on each end_date =====")
    merged_before_topn = len(merged)
    merged = filter_top_n_by_cap(
        merged=merged,
        top_n_by_cap=args.top_n_by_cap,
        cap_col_name="formation_cap",
    )
    print(f"rows before top-N filter = {merged_before_topn}")
    print(f"rows after  top-N filter = {len(merged)}")

    print("\n===== step 6: assign daily deciles =====")
    merged["decile"] = merged.groupby("end_date")["prob_up"].transform(assign_deciles)
    merged = merged.dropna(subset=["decile"]).copy()
    merged["decile"] = merged["decile"].astype(int)
    print(f"rows after decile assignment = {len(merged)}")

    print("\n===== step 7: compute gross/net portfolio returns =====")
    port_rets_gross = compute_portfolio_returns(merged=merged, weighting=args.weighting, ret_col="gross_fwd_ret")
    port_rets_net = compute_portfolio_returns(merged=merged, weighting=args.weighting, ret_col="net_fwd_ret")
    print(f"gross portfolio-return rows = {len(port_rets_gross)}")
    print(f"net portfolio-return rows   = {len(port_rets_net)}")

    print("\n===== step 8: compute turnover (summary use only) =====")
    holdings = compute_holding_weights(merged=merged, weighting=args.weighting, drift_ret_col="net_fwd_ret")
    turnover_ts = compute_turnover(holdings=holdings, horizon_days=args.horizon_days)
    print(f"turnover ts rows = {len(turnover_ts)}")

    print("\n===== step 9: summarize summary-only =====")
    summary_gross = summarize_deciles_summary_only(
        port_rets=port_rets_gross,
        horizon_days=args.horizon_days,
        turnover_ts=turnover_ts,
    )
    summary_net = summarize_deciles_summary_only(
        port_rets=port_rets_net,
        horizon_days=args.horizon_days,
        turnover_ts=turnover_ts,
    )

    summary_gross_out = out_dir / f"summary_gross_{args.weighting}.csv"
    summary_net_out = out_dir / f"summary_net_{args.weighting}.csv"
    summary_gross.to_csv(summary_gross_out, index=False, encoding="utf-8-sig")
    summary_net.to_csv(summary_net_out, index=False, encoding="utf-8-sig")

    print("\n===== done =====")
    print(f"saved gross summary table -> {summary_gross_out}")
    print(f"saved net summary table   -> {summary_net_out}")
    print("\n===== preview: net summary =====")
    print(summary_net)


if __name__ == "__main__":
    main()
