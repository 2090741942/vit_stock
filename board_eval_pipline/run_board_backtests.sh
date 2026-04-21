#!/usr/bin/env bash
set -euo pipefail

# 用法示例：
# bash run_board_backtests.sh \
#   --pred_split_dir /workspace/predict_board_splits \
#   --pred_stem preds_I60_R5 \
#   --csv_root /workspace/stock_data_csv \
#   --horizon_days 5 \
#   --weighting equal \
#   --backtest_py /workspace/vit_stock/portfolio_backtest_with_costs_and_turnover.py \
#   --out_root /workspace/backtests_by_board/I60_R5_equal

PRED_SPLIT_DIR=""
PRED_STEM=""
CSV_ROOT=""
HORIZON_DAYS=""
WEIGHTING="equal"
CAP_COL="CirculatedMarketValue"
BUY_COST_RATE="0.0000641"
SELL_COST_RATE="0.0005641"
BACKTEST_PY=""
OUT_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pred_split_dir) PRED_SPLIT_DIR="$2"; shift 2 ;;
    --pred_stem) PRED_STEM="$2"; shift 2 ;;
    --csv_root) CSV_ROOT="$2"; shift 2 ;;
    --horizon_days) HORIZON_DAYS="$2"; shift 2 ;;
    --weighting) WEIGHTING="$2"; shift 2 ;;
    --cap_col) CAP_COL="$2"; shift 2 ;;
    --buy_cost_rate) BUY_COST_RATE="$2"; shift 2 ;;
    --sell_cost_rate) SELL_COST_RATE="$2"; shift 2 ;;
    --backtest_py) BACKTEST_PY="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${PRED_SPLIT_DIR}" || -z "${PRED_STEM}" || -z "${CSV_ROOT}" || -z "${HORIZON_DAYS}" || -z "${BACKTEST_PY}" || -z "${OUT_ROOT}" ]]; then
  echo "Missing required args."
  exit 1
fi

mkdir -p "${OUT_ROOT}"

BOARDS=("sh_main" "sz_main" "chinext" "star" "bse")

for BOARD in "${BOARDS[@]}"; do
  PRED_CSV="${PRED_SPLIT_DIR}/${PRED_STEM}_${BOARD}.csv"
  OUT_DIR="${OUT_ROOT}/${BOARD}"

  if [[ ! -f "${PRED_CSV}" ]]; then
    echo "[SKIP] ${BOARD}: prediction csv not found -> ${PRED_CSV}"
    continue
  fi

  mkdir -p "${OUT_DIR}"
  echo "=================================================="
  echo "[RUN] board=${BOARD}"
  echo "pred_csv=${PRED_CSV}"
  echo "out_dir=${OUT_DIR}"
  echo "=================================================="

  if [[ "${WEIGHTING}" == "value" ]]; then
    python "${BACKTEST_PY}" \
      --pred_csv "${PRED_CSV}" \
      --csv_root "${CSV_ROOT}" \
      --horizon_days "${HORIZON_DAYS}" \
      --weighting value \
      --cap_col "${CAP_COL}" \
      --buy_cost_rate "${BUY_COST_RATE}" \
      --sell_cost_rate "${SELL_COST_RATE}" \
      --out_dir "${OUT_DIR}"
  else
    python "${BACKTEST_PY}" \
      --pred_csv "${PRED_CSV}" \
      --csv_root "${CSV_ROOT}" \
      --horizon_days "${HORIZON_DAYS}" \
      --weighting equal \
      --buy_cost_rate "${BUY_COST_RATE}" \
      --sell_cost_rate "${SELL_COST_RATE}" \
      --out_dir "${OUT_DIR}"
  fi
done
