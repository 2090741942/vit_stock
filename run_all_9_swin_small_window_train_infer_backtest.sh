#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Small-window Swin 3×3 全流程脚本
# 目标：
#   I5  -> R5, R20, R60
#   I20 -> R5, R20, R60
#   I60 -> R5, R20, R60
#
# 共 9 个模型，每个模型执行：
#   1) train_swin_small_window.py
#   2) infer_test_swin_small_window.py
#   3) portfolio_backtest.py
#
# 说明：
# - 本脚本按顺序串行跑 9 个任务，最稳妥。
# - 使用 small-window Swin：
#     * pad_multiple = 32
#     * swin_window_size = 4
# ============================================================

WORKDIR="/workspace/vit_stock"

TRAIN_PY="${WORKDIR}/train_swin_small_window.py"
INFER_PY="${WORKDIR}/infer_test_swin_small_window.py"
BACKTEST_PY="${WORKDIR}/portfolio_backtest.py"

# ===== 基础数据路径 =====
LABEL_DIR="/workspace/vit_stock_data/label_npz"
SYMBOL_TO_TAR_DIR="${WORKDIR}/symbol_to_tar"
CSV_ROOT="/workspace/stock_csv_8year"

# ===== 模型参数 =====
HF_MODEL_NAME="microsoft/swin-tiny-patch4-window7-224"
SWIN_WINDOW_SIZE=4
PAD_MULTIPLE=32

# ===== 训练超参 =====
EPOCHS=10
BATCH_SIZE=64
LR=1e-4
WEIGHT_DECAY=1e-4
NUM_WORKERS=8
SEED=20260315
WARMUP_RATIO=0.05
STEP_LOG_INTERVAL=100
STEP_LOG_FILE="metrics_step.csv"

# ===== 输出目录 =====
TRAIN_ROOT="/workspace/swin_small_window_outs"
PRED_ROOT="/workspace/vit_stock_data/predict_swin_small_window"
BACKTEST_ROOT="/workspace/vit_stock_data/backtest_swin_small_window"
LOG_ROOT="/workspace/vit_stock_data/logs_swin_small_window"

mkdir -p "${TRAIN_ROOT}" "${PRED_ROOT}" "${BACKTEST_ROOT}" "${LOG_ROOT}"

echo "============================================================"
echo "[INFO] TRAIN_PY         = ${TRAIN_PY}"
echo "[INFO] INFER_PY         = ${INFER_PY}"
echo "[INFO] BACKTEST_PY      = ${BACKTEST_PY}"
echo "[INFO] LABEL_DIR        = ${LABEL_DIR}"
echo "[INFO] SYMBOL_TO_TAR    = ${SYMBOL_TO_TAR_DIR}"
echo "[INFO] CSV_ROOT         = ${CSV_ROOT}"
echo "[INFO] HF_MODEL_NAME    = ${HF_MODEL_NAME}"
echo "[INFO] SWIN_WINDOW_SIZE = ${SWIN_WINDOW_SIZE}"
echo "[INFO] PAD_MULTIPLE     = ${PAD_MULTIPLE}"
echo "[INFO] TRAIN_ROOT       = ${TRAIN_ROOT}"
echo "[INFO] PRED_ROOT        = ${PRED_ROOT}"
echo "[INFO] BACKTEST_ROOT    = ${BACKTEST_ROOT}"
echo "[INFO] LOG_ROOT         = ${LOG_ROOT}"
echo "============================================================"

for f in "${TRAIN_PY}" "${INFER_PY}" "${BACKTEST_PY}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] script not found: ${f}"
    exit 1
  fi
done

run_one_task() {
  local WINDOW="$1"
  local HORIZON_IDX="$2"

  local HORIZON_DAYS
  if [[ "${HORIZON_IDX}" == "0" ]]; then
    HORIZON_DAYS=5
  elif [[ "${HORIZON_IDX}" == "1" ]]; then
    HORIZON_DAYS=20
  elif [[ "${HORIZON_IDX}" == "2" ]]; then
    HORIZON_DAYS=60
  else
    echo "[ERROR] invalid HORIZON_IDX=${HORIZON_IDX}"
    exit 1
  fi

  local META_PATH="${LABEL_DIR}/meta_N${WINDOW}_clean.npz"
  local SYMBOL_TO_TAR_JSON="${SYMBOL_TO_TAR_DIR}/symbol_to_tar_N${WINDOW}.json"

  local EXP_NAME="I${WINDOW}_R${HORIZON_DAYS}_swin_w${SWIN_WINDOW_SIZE}"
  local OUT_DIR="${TRAIN_ROOT}/${EXP_NAME}"
  local CKPT_PATH="${OUT_DIR}/best.pt"
  local OUT_CSV="${PRED_ROOT}/preds_${EXP_NAME}.csv"
  local BACKTEST_OUT_DIR="${BACKTEST_ROOT}/backtest_${EXP_NAME}_equal"
  local LOG_PATH="${LOG_ROOT}/${EXP_NAME}.log"

  echo
  echo "############################################################"
  echo "[TASK] ${EXP_NAME}"
  echo "[INFO] META_PATH         = ${META_PATH}"
  echo "[INFO] SYMBOL_TO_TAR     = ${SYMBOL_TO_TAR_JSON}"
  echo "[INFO] OUT_DIR           = ${OUT_DIR}"
  echo "[INFO] OUT_CSV           = ${OUT_CSV}"
  echo "[INFO] BACKTEST_OUT_DIR  = ${BACKTEST_OUT_DIR}"
  echo "[INFO] LOG_PATH          = ${LOG_PATH}"
  echo "############################################################"

  if [[ ! -f "${META_PATH}" ]]; then
    echo "[ERROR] meta file not found: ${META_PATH}"
    exit 1
  fi

  if [[ ! -f "${SYMBOL_TO_TAR_JSON}" ]]; then
    echo "[ERROR] symbol_to_tar json not found: ${SYMBOL_TO_TAR_JSON}"
    exit 1
  fi

  mkdir -p "${OUT_DIR}" "${BACKTEST_OUT_DIR}"

  {
    echo
    echo "==================== TRAIN ${EXP_NAME} ===================="
    python "${TRAIN_PY}" \
      --meta_path "${META_PATH}" \
      --symbol_to_tar_json "${SYMBOL_TO_TAR_JSON}" \
      --horizon_idx "${HORIZON_IDX}" \
      --hf_model_name "${HF_MODEL_NAME}" \
      --swin_window_size "${SWIN_WINDOW_SIZE}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --num_workers "${NUM_WORKERS}" \
      --seed "${SEED}" \
      --pad_multiple "${PAD_MULTIPLE}" \
      --repeat_to_3ch \
      --warmup_ratio "${WARMUP_RATIO}" \
      --step_log_interval "${STEP_LOG_INTERVAL}" \
      --step_log_file "${STEP_LOG_FILE}" \
      --out_dir "${OUT_DIR}"

    if [[ ! -f "${CKPT_PATH}" ]]; then
      echo "[ERROR] best checkpoint not found after training: ${CKPT_PATH}"
      exit 1
    fi

    echo
    echo "==================== INFER ${EXP_NAME} ===================="
    python "${INFER_PY}" \
      --meta_path "${META_PATH}" \
      --symbol_to_tar_json "${SYMBOL_TO_TAR_JSON}" \
      --checkpoint "${CKPT_PATH}" \
      --hf_model_name "${HF_MODEL_NAME}" \
      --horizon_idx "${HORIZON_IDX}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --pad_multiple "${PAD_MULTIPLE}" \
      --repeat_to_3ch \
      --out_csv "${OUT_CSV}"

    if [[ ! -f "${OUT_CSV}" ]]; then
      echo "[ERROR] prediction csv not found after inference: ${OUT_CSV}"
      exit 1
    fi

    echo
    echo "==================== BACKTEST ${EXP_NAME} ===================="
    python "${BACKTEST_PY}" \
      --pred_csv "${OUT_CSV}" \
      --csv_root "${CSV_ROOT}" \
      --horizon_days "${HORIZON_DAYS}" \
      --weighting equal \
      --out_dir "${BACKTEST_OUT_DIR}"

    echo "[OK] Finished ${EXP_NAME}"
  } 2>&1 | tee "${LOG_PATH}"
}

# ============================================================
# 3 × 3 全部任务
# ============================================================

# I5 -> R5 / R20 / R60
run_one_task 5 0
run_one_task 5 1
run_one_task 5 2

# I20 -> R5 / R20 / R60
run_one_task 20 0
run_one_task 20 1
run_one_task 20 2

# I60 -> R5 / R20 / R60
run_one_task 60 0
run_one_task 60 1
run_one_task 60 2

echo
echo "============================================================"
echo "[ALL DONE] 9 small-window Swin models have been trained, inferred, and backtested."
echo "[INFO] Training outputs are under: ${TRAIN_ROOT}"
echo "[INFO] Prediction csvs are under: ${PRED_ROOT}"
echo "[INFO] Backtest outputs are under: ${BACKTEST_ROOT}"
echo "[INFO] Per-task logs are under: ${LOG_ROOT}"
echo "============================================================"
