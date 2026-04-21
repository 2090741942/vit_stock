#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON_BIN="${PYTHON_BIN:-python}"

PROJECT_ROOT="/workspace/vit_stock"
DATA_ROOT="/workspace/vit_stock_data"
PIPELINE_DIR="${PROJECT_ROOT}/board_eval_pipline"

# 默认处理你当前已有的两个预测目录
PRED_DIRS="${PRED_DIRS:-${DATA_ROOT}/predict ${DATA_ROOT}/predict_dinov2_square}"

# 这里只做“回测前准备”，不实际执行回测
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/board_eval_ready}"

# 1 表示丢弃 unknown 板块；0 表示保留
DROP_UNKNOWN="${DROP_UNKNOWN:-0}"

mkdir -p "${OUT_ROOT}"

NORM_PY="${PIPELINE_DIR}/norm_predict_symbol.py"
BUILD_MAP_PY="${PIPELINE_DIR}/build_symbol_to_board.py"
ATTACH_SPLIT_PY="${PIPELINE_DIR}/attach_board_and_split_predictions.py"

for fp in "${NORM_PY}" "${BUILD_MAP_PY}" "${ATTACH_SPLIT_PY}"; do
  if [[ ! -f "${fp}" ]]; then
    echo "[ERROR] required file not found: ${fp}"
    exit 1
  fi
done

declare -A META_PATHS
META_PATHS[5]="${DATA_ROOT}/label_npz/meta_N5_clean.npz"
META_PATHS[20]="${DATA_ROOT}/label_npz/meta_N20_clean.npz"
META_PATHS[60]="${DATA_ROOT}/label_npz/meta_N60_clean.npz"

infer_img_window() {
  local name="$1"
  if [[ "${name}" =~ preds_I5_R[0-9]+ ]]; then
    echo "5"
    return
  fi
  if [[ "${name}" =~ preds_I20_R[0-9]+ ]]; then
    echo "20"
    return
  fi
  if [[ "${name}" =~ preds_I60_R[0-9]+ ]]; then
    echo "60"
    return
  fi
  echo ""
}

process_one_pred_csv() {
  local pred_csv="$1"

  local base_name
  base_name="$(basename "${pred_csv}" .csv)"

  local img_window
  img_window="$(infer_img_window "${base_name}")"
  if [[ -z "${img_window}" ]]; then
    echo "[SKIP] cannot infer image window from filename: ${pred_csv}"
    return
  fi

  local meta_path="${META_PATHS[${img_window}]}"
  if [[ ! -f "${meta_path}" ]]; then
    echo "[ERROR] meta file not found for window ${img_window}: ${meta_path}"
    exit 1
  fi

  local pred_parent
  pred_parent="$(basename "$(dirname "${pred_csv}")")"

  local item_root="${OUT_ROOT}/${pred_parent}/${base_name}"
  local norm_dir="${item_root}/normalized"
  local map_dir="${item_root}/mapping"
  local tagged_dir="${item_root}/tagged"
  local split_dir="${item_root}/pred_splits"

  mkdir -p "${norm_dir}" "${map_dir}" "${tagged_dir}" "${split_dir}"

  local norm_csv="${norm_dir}/${base_name}_normalized.csv"
  local board_map_csv="${map_dir}/symbol_to_board.csv"
  local with_board_csv="${tagged_dir}/${base_name}_with_board.csv"

  echo "============================================================"
  echo "[PREP] ${pred_csv}"
  echo "  image window inferred: N=${img_window}"
  echo "  normalized csv:        ${norm_csv}"
  echo "  symbol->board csv:     ${board_map_csv}"
  echo "  tagged csv:            ${with_board_csv}"
  echo "  split dir:             ${split_dir}"
  echo "============================================================"

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

  echo "[READY] Backtest preparation finished for: ${pred_csv}"
  echo
}

for pred_dir in ${PRED_DIRS}; do
  if [[ ! -d "${pred_dir}" ]]; then
    echo "[WARN] prediction directory not found, skip: ${pred_dir}"
    continue
  fi

  found_any=0
  while IFS= read -r -d '' pred_csv; do
    found_any=1
    process_one_pred_csv "${pred_csv}"
  done < <(find "${pred_dir}" -maxdepth 1 -type f -name 'preds_*.csv' -print0 | sort -z)

  if [[ "${found_any}" == "0" ]]; then
    echo "[WARN] no prediction csv found under: ${pred_dir}"
  fi
done

echo "============================================================"
echo "[DONE] All prediction files have been prepared for board backtests."
echo "Output root: ${OUT_ROOT}"
echo "You can now run run_board_backtests.sh on any prepared pred_splits dir."
echo "============================================================"
