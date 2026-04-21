#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON_BIN="${PYTHON_BIN:-python}"

PROJECT_ROOT="/workspace/vit_stock"
DATA_ROOT="/workspace/vit_stock_data"
CSV_ROOT="${CSV_ROOT:-/workspace/stock_csv_8year}"

OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/eval_summary_only_main_vit_plus_dinov2}"

BACKTEST_PY="${PROJECT_ROOT}/backtest/portfolio_backtest_summary_only_with_turnover_all_rows.py"
if [[ ! -f "${BACKTEST_PY}" ]]; then
  echo "[ERROR] backtest script not found: ${BACKTEST_PY}"
  exit 1
fi

CAP_COL="${CAP_COL:-CirculatedMarketValue}"
BUY_COST_RATE="${BUY_COST_RATE:-0.0000641}"
SELL_COST_RATE="${SELL_COST_RATE:-0.0005641}"

BOARDS=("sh_main" "sz_main" "chinext" "star" "bse")

infer_horizon_days() {
  local name="$1"
  if [[ "${name}" =~ _R5([^0-9]|$) ]]; then
    echo "5"; return
  fi
  if [[ "${name}" =~ _R20([^0-9]|$) ]]; then
    echo "20"; return
  fi
  if [[ "${name}" =~ _R60([^0-9]|$) ]]; then
    echo "60"; return
  fi
  echo ""
}

run_backtest_once() {
  local pred_csv="$1"
  local out_dir="$2"
  local horizon_days="$3"
  local weighting="$4"

  mkdir -p "${out_dir}"

  "${PYTHON_BIN}" "${BACKTEST_PY}" \
    --pred_csv "${pred_csv}" \
    --csv_root "${CSV_ROOT}" \
    --out_dir "${out_dir}" \
    --horizon_days "${horizon_days}" \
    --weighting "${weighting}" \
    --top_n_by_cap 0 \
    --buy_cost_rate "${BUY_COST_RATE}" \
    --sell_cost_rate "${SELL_COST_RATE}" \
    --cap_col "${CAP_COL}"
}

process_one_vit_item() {
  local item_root="$1"
  local base_name
  base_name="$(basename "${item_root}")"

  local horizon_days
  horizon_days="$(infer_horizon_days "${base_name}")"
  if [[ -z "${horizon_days}" ]]; then
    echo "[SKIP] cannot infer horizon days from: ${base_name}"
    return
  fi

  local split_dir="${item_root}/board_prep/pred_splits"
  if [[ ! -d "${split_dir}" ]]; then
    echo "[SKIP] split dir not found: ${split_dir}"
    return
  fi

  echo "============================================================"
  echo "[RUN] board-only backtests for ${base_name}"
  echo "  split_dir:     ${split_dir}"
  echo "  horizon_days:  ${horizon_days}"
  echo "============================================================"

  for board in "${BOARDS[@]}"; do
    local board_pred_csv="${split_dir}/${base_name}_normalized_${board}.csv"
    if [[ ! -f "${board_pred_csv}" ]]; then
      echo "[SKIP] board pred not found: ${board_pred_csv}"
      continue
    fi

    run_backtest_once \
      "${board_pred_csv}" \
      "${item_root}/3_board_equal/${board}" \
      "${horizon_days}" \
      "equal"

    run_backtest_once \
      "${board_pred_csv}" \
      "${item_root}/4_board_value/${board}" \
      "${horizon_days}" \
      "value"
  done

  echo "[DONE] ${base_name}"
  echo
}

VIT_ROOT="${OUT_ROOT}/vit"
if [[ ! -d "${VIT_ROOT}" ]]; then
  echo "[ERROR] vit root not found: ${VIT_ROOT}"
  exit 1
fi

found_any=0
while IFS= read -r -d '' item_root; do
  found_any=1
  process_one_vit_item "${item_root}"
done < <(find "${VIT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if [[ "${found_any}" == "0" ]]; then
  echo "[WARN] no vit experiment directories found under: ${VIT_ROOT}"
fi

echo "============================================================"
echo "[DONE] Board-only backtests finished."
echo "Output root: ${VIT_ROOT}"
echo "Generated:"
echo "  3_board_equal/<board>/summary_gross_equal.csv + summary_net_equal.csv"
echo "  4_board_value/<board>/summary_gross_value.csv + summary_net_value.csv"
echo "============================================================"
