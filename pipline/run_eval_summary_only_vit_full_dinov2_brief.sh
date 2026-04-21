#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON_BIN="${PYTHON_BIN:-python}"

PROJECT_ROOT="/workspace/vit_stock"
DATA_ROOT="/workspace/vit_stock_data"
CSV_ROOT="${CSV_ROOT:-/workspace/stock_csv_8year}"

PIPELINE_DIR="${PROJECT_ROOT}/board_eval_pipline"

# 主模型 / 对照模型预测目录
VIT_PRED_DIR="${VIT_PRED_DIR:-${DATA_ROOT}/predict}"
DINOV2_PRED_DIR="${DINOV2_PRED_DIR:-${DATA_ROOT}/predict_dinov2_square}"

# 输出根目录
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/eval_summary_only_main_vit_plus_dinov2}"

# 市值列与成本参数
CAP_COL="${CAP_COL:-CirculatedMarketValue}"
BUY_COST_RATE="${BUY_COST_RATE:-0.0000641}"
SELL_COST_RATE="${SELL_COST_RATE:-0.0005641}"

# 分市场是否丢弃 unknown
DROP_UNKNOWN="${DROP_UNKNOWN:-0}"

mkdir -p "${OUT_ROOT}"

# 固定使用新的 summary-only 回测脚本
BACKTEST_PY="${PROJECT_ROOT}/backtest/portfolio_backtest_summary_only_with_turnover_all_rows.py"
if [[ ! -f "${BACKTEST_PY}" ]]; then
  echo "[ERROR] backtest script not found: ${BACKTEST_PY}"
  echo "Please place portfolio_backtest_summary_only_with_turnover_all_rows.py under ${PROJECT_ROOT}/backtest/"
  exit 1
fi

NORM_PY="${PIPELINE_DIR}/norm_predict_symbol.py"
BUILD_MAP_PY="${PIPELINE_DIR}/build_symbol_to_board.py"
ATTACH_SPLIT_PY="${PIPELINE_DIR}/attach_board_and_split_predictions.py"

for fp in "${BACKTEST_PY}" "${NORM_PY}" "${BUILD_MAP_PY}" "${ATTACH_SPLIT_PY}"; do
  if [[ ! -f "${fp}" ]]; then
    echo "[ERROR] required file not found: ${fp}"
    exit 1
  fi
done

declare -A META_PATHS
META_PATHS[5]="${DATA_ROOT}/label_npz/meta_N5_clean.npz"
META_PATHS[20]="${DATA_ROOT}/label_npz/meta_N20_clean.npz"
META_PATHS[60]="${DATA_ROOT}/label_npz/meta_N60_clean.npz"

BOARDS=("sh_main" "sz_main" "chinext" "star" "bse")

infer_img_window() {
  local name="$1"
  if [[ "${name}" =~ preds_I5_R[0-9]+ ]]; then
    echo "5"; return
  fi
  if [[ "${name}" =~ preds_I20_R[0-9]+ ]]; then
    echo "20"; return
  fi
  if [[ "${name}" =~ preds_I60_R[0-9]+ ]]; then
    echo "60"; return
  fi
  echo ""
}

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
  local top_n_by_cap="$5"

  mkdir -p "${out_dir}"

  "${PYTHON_BIN}" "${BACKTEST_PY}" \
    --pred_csv "${pred_csv}" \
    --csv_root "${CSV_ROOT}" \
    --out_dir "${out_dir}" \
    --horizon_days "${horizon_days}" \
    --weighting "${weighting}" \
    --top_n_by_cap "${top_n_by_cap}" \
    --buy_cost_rate "${BUY_COST_RATE}" \
    --sell_cost_rate "${SELL_COST_RATE}" \
    --cap_col "${CAP_COL}"
}

prepare_board_inputs() {
  local pred_csv="$1"
  local prep_root="$2"
  local img_window="$3"

  local base_name
  base_name="$(basename "${pred_csv}" .csv)"

  local meta_path="${META_PATHS[${img_window}]}"
  if [[ ! -f "${meta_path}" ]]; then
    echo "[ERROR] meta file not found for window ${img_window}: ${meta_path}"
    exit 1
  fi

  local norm_dir="${prep_root}/normalized"
  local map_dir="${prep_root}/mapping"
  local tagged_dir="${prep_root}/tagged"
  local split_dir="${prep_root}/pred_splits"

  mkdir -p "${norm_dir}" "${map_dir}" "${tagged_dir}" "${split_dir}"

  local norm_csv="${norm_dir}/${base_name}_normalized.csv"
  local board_map_csv="${map_dir}/symbol_to_board.csv"
  local with_board_csv="${tagged_dir}/${base_name}_with_board.csv"

  echo "[PREP] board inputs for ${pred_csv}"

  "${PYTHON_BIN}" "${NORM_PY}" \
    --in_csv "${pred_csv}" \
    --out_csv "${norm_csv}"

  "${PYTHON_BIN}" "${BUILD_MAP_PY}" \
    --meta_path "${meta_path}" \
    --out_csv "${board_map_csv}"

  if [[ "${DROP_UNKNOWN}" == "1" ]]; then
    "${PYTHON_BIN}" "${ATTACH_SPLIT_PY}" \
      --pred_csv "${norm_csv}" \
      --symbol_to_board_csv "${board_map_csv}" \
      --out_with_board_csv "${with_board_csv}" \
      --split_out_dir "${split_dir}" \
      --drop_unknown
  else
    "${PYTHON_BIN}" "${ATTACH_SPLIT_PY}" \
      --pred_csv "${norm_csv}" \
      --symbol_to_board_csv "${board_map_csv}" \
      --out_with_board_csv "${with_board_csv}" \
      --split_out_dir "${split_dir}"
  fi
}

process_vit_pred_csv() {
  local pred_csv="$1"

  local base_name
  base_name="$(basename "${pred_csv}" .csv)"

  if [[ "${base_name}" == *_cnn* ]]; then
    echo "[SKIP] CNN prediction excluded: ${pred_csv}"
    return
  fi

  local img_window
  img_window="$(infer_img_window "${base_name}")"
  if [[ -z "${img_window}" ]]; then
    echo "[SKIP] cannot infer image window from filename: ${pred_csv}"
    return
  fi

  local horizon_days
  horizon_days="$(infer_horizon_days "${base_name}")"
  if [[ -z "${horizon_days}" ]]; then
    echo "[SKIP] cannot infer horizon days from filename: ${pred_csv}"
    return
  fi

  local item_root="${OUT_ROOT}/vit/${base_name}"
  mkdir -p "${item_root}"

  echo "============================================================"
  echo "[RUN][ViT full] ${pred_csv}"
  echo "  image window: ${img_window}"
  echo "  horizon_days: ${horizon_days}"
  echo "  out_root:     ${item_root}"
  echo "============================================================"

  # 1. 全市场，等权（gross=无成本, net=有成本）
  run_backtest_once "${pred_csv}" "${item_root}/1_market_equal" "${horizon_days}" "equal" "0"

  # 2. 全市场，市值加权（gross=无成本, net=有成本）
  run_backtest_once "${pred_csv}" "${item_root}/2_market_value" "${horizon_days}" "value" "0"

  # 3/4. 分市场准备
  local board_prep_root="${item_root}/board_prep"
  prepare_board_inputs "${pred_csv}" "${board_prep_root}" "${img_window}"

  # 3. 分市场，等权
  for board in "${BOARDS[@]}"; do
    local board_pred_csv="${board_prep_root}/pred_splits/${base_name}_${board}.csv"
    if [[ ! -f "${board_pred_csv}" ]]; then
      echo "[SKIP] board pred not found: ${board_pred_csv}"
      continue
    fi
    run_backtest_once "${board_pred_csv}" "${item_root}/3_board_equal/${board}" "${horizon_days}" "equal" "0"
  done

  # 4. 分市场，市值加权
  for board in "${BOARDS[@]}"; do
    local board_pred_csv="${board_prep_root}/pred_splits/${base_name}_${board}.csv"
    if [[ ! -f "${board_pred_csv}" ]]; then
      continue
    fi
    run_backtest_once "${board_pred_csv}" "${item_root}/4_board_value/${board}" "${horizon_days}" "value" "0"
  done

  # 5. 全市场前500，等权
  run_backtest_once "${pred_csv}" "${item_root}/5_top500_equal" "${horizon_days}" "equal" "500"

  # 6. 全市场前500，市值加权
  run_backtest_once "${pred_csv}" "${item_root}/6_top500_value" "${horizon_days}" "value" "500"

  echo "[DONE][ViT full] ${pred_csv}"
  echo
}

process_dinov2_pred_csv() {
  local pred_csv="$1"

  local base_name
  base_name="$(basename "${pred_csv}" .csv)"

  if [[ "${base_name}" == *_cnn* ]]; then
    echo "[SKIP] CNN prediction excluded: ${pred_csv}"
    return
  fi

  local horizon_days
  horizon_days="$(infer_horizon_days "${base_name}")"
  if [[ -z "${horizon_days}" ]]; then
    echo "[SKIP] cannot infer horizon days from filename: ${pred_csv}"
    return
  fi

  local item_root="${OUT_ROOT}/dinov2/${base_name}"
  mkdir -p "${item_root}"

  echo "============================================================"
  echo "[RUN][DINOv2 brief] ${pred_csv}"
  echo "  horizon_days: ${horizon_days}"
  echo "  out_root:     ${item_root}"
  echo "============================================================"

  # A. 全市场，等权（gross=无成本, net=有成本）
  run_backtest_once "${pred_csv}" "${item_root}/1_market_equal" "${horizon_days}" "equal" "0"

  # B. 全市场，市值加权（gross=无成本, net=有成本）
  run_backtest_once "${pred_csv}" "${item_root}/2_market_value" "${horizon_days}" "value" "0"

  echo "[DONE][DINOv2 brief] ${pred_csv}"
  echo
}

process_dir_with_handler() {
  local pred_dir="$1"
  local mode="$2"

  if [[ ! -d "${pred_dir}" ]]; then
    echo "[WARN] prediction directory not found, skip: ${pred_dir}"
    return
  fi

  local found_any=0
  while IFS= read -r -d '' pred_csv; do
    found_any=1
    if [[ "${mode}" == "vit" ]]; then
      process_vit_pred_csv "${pred_csv}"
    else
      process_dinov2_pred_csv "${pred_csv}"
    fi
  done < <(find "${pred_dir}" -maxdepth 1 -type f -name 'preds_*.csv' -print0 | sort -z)

  if [[ "${found_any}" == "0" ]]; then
    echo "[WARN] no prediction csv found under: ${pred_dir}"
  fi
}

process_dir_with_handler "${VIT_PRED_DIR}" "vit"
process_dir_with_handler "${DINOV2_PRED_DIR}" "dinov2"

echo "============================================================"
echo "[DONE] Summary-only evaluation finished."
echo "Output root: ${OUT_ROOT}"
echo "Each evaluation writes only summary_gross_*.csv and summary_net_*.csv"
echo "============================================================"
