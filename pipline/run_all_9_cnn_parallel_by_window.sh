#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 3×3 CNN 全流程脚本（同一窗口的 3 个 horizon 并行）
# 目标：
#   I5  -> R5, R20, R60   （并行）
#   I20 -> R5, R20, R60   （并行）
#   I60 -> R5, R20, R60   （并行）
#
# 共 9 个模型，每个模型执行：
#   1) train_cnn.py
#   2) infer_test_cnn.py
#   3) portfolio_backtest.py
#
# 并行策略：
# - 同一个 WINDOW 下的 3 个 horizon 并行执行
# - 不同 WINDOW 之间串行，避免同时开太多任务
#
# 说明：
# - 该脚本假设你已经把下面三个文件放在 /workspace/vit_stock：
#     /workspace/vit_stock/train_cnn.py
#     /workspace/vit_stock/infer_test_cnn.py
#     /workspace/vit_stock/portfolio_backtest.py
# ============================================================

WORKDIR="/workspace/vit_stock"

TRAIN_PY="${WORKDIR}/train_cnn.py"
INFER_PY="${WORKDIR}/infer_test_cnn.py"
BACKTEST_PY="${WORKDIR}/portfolio_backtest.py"

# ===== 基础数据路径 =====
LABEL_DIR="/workspace/vit_stock_data/label_npz"
SYMBOL_TO_TAR_DIR="${WORKDIR}/symbol_to_tar"
CSV_ROOT="/workspace/stock_csv_8year"

# ===== 训练超参（统一给 9 个模型）=====
EPOCHS=30
BATCH_SIZE=128
LR=1e-5
NUM_WORKERS=8
PATCH_SIZE=16
SEED=20260321
WARMUP_RATIO=0.05
STEP_LOG_INTERVAL=500

# ===== CNN 结构 / 预处理参数 =====
DROPOUT=0.5
FC_HIDDEN_DIM=0
NEGATIVE_SLOPE=0.01
STATS_MODE="approx"      # full / approx / manual
STATS_BATCHES=50

# ===== 输出根目录 =====
TRAIN_ROOT="/workspace/cnn_outs"
PRED_ROOT="/workspace/vit_stock_data/predict"
BACKTEST_ROOT="/workspace/vit_stock_data/backtest"
LOG_ROOT="/workspace/logs_cnn"

mkdir -p "${TRAIN_ROOT}" "${PRED_ROOT}" "${BACKTEST_ROOT}" "${LOG_ROOT}"

echo "============================================================"
echo "[INFO] TRAIN_PY      = ${TRAIN_PY}"
echo "[INFO] INFER_PY      = ${INFER_PY}"
echo "[INFO] BACKTEST_PY   = ${BACKTEST_PY}"
echo "[INFO] LABEL_DIR     = ${LABEL_DIR}"
echo "[INFO] SYMBOL_TO_TAR = ${SYMBOL_TO_TAR_DIR}"
echo "[INFO] CSV_ROOT      = ${CSV_ROOT}"
echo "[INFO] TRAIN_ROOT    = ${TRAIN_ROOT}"
echo "[INFO] PRED_ROOT     = ${PRED_ROOT}"
echo "[INFO] BACKTEST_ROOT = ${BACKTEST_ROOT}"
echo "[INFO] LOG_ROOT      = ${LOG_ROOT}"
echo "============================================================"

for f in "${TRAIN_PY}" "${INFER_PY}" "${BACKTEST_PY}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] script not found: ${f}"
    exit 1
  fi
done

# ============================================================
# 单个任务函数
#   参数1: WINDOW       5 / 20 / 60
#   参数2: HORIZON_IDX  0 / 1 / 2
# ============================================================
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
    return 1
  fi

  local META_PATH="${LABEL_DIR}/meta_N${WINDOW}_clean.npz"
  local SYMBOL_TO_TAR_JSON="${SYMBOL_TO_TAR_DIR}/symbol_to_tar_N${WINDOW}.json"

  local EXP_NAME="I${WINDOW}_R${HORIZON_DAYS}_cnn"
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
    return 1
  fi

  if [[ ! -f "${SYMBOL_TO_TAR_JSON}" ]]; then
    echo "[ERROR] symbol_to_tar json not found: ${SYMBOL_TO_TAR_JSON}"
    return 1
  fi

  mkdir -p "${OUT_DIR}" "${BACKTEST_OUT_DIR}"

  {
    echo
    echo "==================== TRAIN ${EXP_NAME} ===================="
    python "${TRAIN_PY}" \
      --meta_path "${META_PATH}" \
      --symbol_to_tar_json "${SYMBOL_TO_TAR_JSON}" \
      --horizon_idx "${HORIZON_IDX}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --num_workers "${NUM_WORKERS}" \
      --patch_size "${PATCH_SIZE}" \
      --seed "${SEED}" \
      --warmup_ratio "${WARMUP_RATIO}" \
      --step_log_interval "${STEP_LOG_INTERVAL}" \
      --dropout "${DROPOUT}" \
      --fc_hidden_dim "${FC_HIDDEN_DIM}" \
      --negative_slope "${NEGATIVE_SLOPE}" \
      --stats_mode "${STATS_MODE}" \
      --stats_batches "${STATS_BATCHES}" \
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
      --horizon_idx "${HORIZON_IDX}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --patch_size "${PATCH_SIZE}" \
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
# 并行执行某个 WINDOW 下的 3 个 horizon
# ============================================================
run_window_parallel() {
  local WINDOW="$1"

  echo
  echo "============================================================"
  echo "[WINDOW] Start parallel group for I${WINDOW} -> R5/R20/R60"
  echo "============================================================"

  local pids=()
  run_one_task "${WINDOW}" 0 &
  pids+=($!)
  run_one_task "${WINDOW}" 1 &
  pids+=($!)
  run_one_task "${WINDOW}" 2 &
  pids+=($!)

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    echo "[ERROR] At least one task failed in window I${WINDOW}."
    exit 1
  fi

  echo
  echo "============================================================"
  echo "[WINDOW] Finished parallel group for I${WINDOW}"
  echo "============================================================"
}

# ============================================================
# 主流程：窗口间串行，窗口内并行
# ============================================================
# run_window_parallel 5
# run_window_parallel 20
# run_window_parallel 60

run_one_task 20 0

run_one_task 20 1

run_one_task 20 2

echo
echo "============================================================"
echo "[ALL DONE] 9 CNN models have been trained, inferred, and backtested."
echo "[INFO] Training outputs are under: ${TRAIN_ROOT}"
echo "[INFO] Prediction csvs are under: ${PRED_ROOT}"
echo "[INFO] Backtest outputs are under: ${BACKTEST_ROOT}"
echo "[INFO] Per-task logs are under: ${LOG_ROOT}"
echo "============================================================"
