#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 3×3 CNN 全流程脚本
# 目标：
#   I5  -> R5, R20, R60
#   I20 -> R5, R20, R60
#   I60 -> R5, R20, R60
#
# 共 9 个模型，每个模型执行：
#   1) train_cnn.py
#   2) infer_test_cnn.py
#
# 说明：
# - 该脚本假设你已经把下面两个文件放在 /workspace：
#     /workspace/train_cnn.py
#     /workspace/infer_test_cnn.py
# - 回测不在这个脚本里做；这个脚本只负责“训练 + 推理”。
# - 输出预测 CSV 后，你可以再用 portfolio_backtest.py 批量回测。
# ============================================================

WORKDIR="/workspace/vit_stock"

TRAIN_PY="${WORKDIR}/train_cnn.py"
INFER_PY="${WORKDIR}/infer_test_cnn.py"

# ===== 基础数据路径 =====
LABEL_DIR="/workspace/vit_stock_data/label_npz"
SYMBOL_TO_TAR_DIR="${WORKDIR}/symbol_to_tar"

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

mkdir -p "${TRAIN_ROOT}" "${PRED_ROOT}"

echo "============================================================"
echo "[INFO] TRAIN_PY      = ${TRAIN_PY}"
echo "[INFO] INFER_PY      = ${INFER_PY}"
echo "[INFO] LABEL_DIR     = ${LABEL_DIR}"
echo "[INFO] SYMBOL_TO_TAR = ${SYMBOL_TO_TAR_DIR}"
echo "[INFO] TRAIN_ROOT    = ${TRAIN_ROOT}"
echo "[INFO] PRED_ROOT     = ${PRED_ROOT}"
echo "============================================================"

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[ERROR] train script not found: ${TRAIN_PY}"
  exit 1
fi

if [[ ! -f "${INFER_PY}" ]]; then
  echo "[ERROR] infer script not found: ${INFER_PY}"
  exit 1
fi

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
    exit 1
  fi

  local META_PATH="${LABEL_DIR}/meta_N${WINDOW}_clean.npz"
  local SYMBOL_TO_TAR_JSON="${SYMBOL_TO_TAR_DIR}/symbol_to_tar_N${WINDOW}.json"

  local EXP_NAME="I${WINDOW}_R${HORIZON_DAYS}_cnn"
  local OUT_DIR="${TRAIN_ROOT}/${EXP_NAME}"
  local CKPT_PATH="${OUT_DIR}/best.pt"
  local OUT_CSV="${PRED_ROOT}/preds_${EXP_NAME}.csv"

  echo
  echo "############################################################"
  echo "[TASK] ${EXP_NAME}"
  echo "[INFO] META_PATH         = ${META_PATH}"
  echo "[INFO] SYMBOL_TO_TAR     = ${SYMBOL_TO_TAR_JSON}"
  echo "[INFO] OUT_DIR           = ${OUT_DIR}"
  echo "[INFO] OUT_CSV           = ${OUT_CSV}"
  echo "############################################################"

  if [[ ! -f "${META_PATH}" ]]; then
    echo "[ERROR] meta file not found: ${META_PATH}"
    exit 1
  fi

  if [[ ! -f "${SYMBOL_TO_TAR_JSON}" ]]; then
    echo "[ERROR] symbol_to_tar json not found: ${SYMBOL_TO_TAR_JSON}"
    exit 1
  fi

  mkdir -p "${OUT_DIR}"

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

  echo "[OK] Finished ${EXP_NAME}"
}

# ===== 回测脚本与数据路径 =====
BACKTEST_PY="${WORKDIR}/portfolio_backtest.py"
CSV_ROOT="$/workspace/stock_csv_8year"
BACKTEST_ROOT="/workspace/vit_stock_data/backtest"

mkdir -p "${BACKTEST_ROOT}"

run_one_backtest() {
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

  local EXP_NAME="I${WINDOW}_R${HORIZON_DAYS}_cnn"
  local PRED_CSV="${PRED_ROOT}/preds_${EXP_NAME}.csv"
  local OUT_DIR="${BACKTEST_ROOT}/backtest_${EXP_NAME}_equal"

  echo
  echo "==================== BACKTEST ${EXP_NAME} ===================="

  if [[ ! -f "${PRED_CSV}" ]]; then
    echo "[ERROR] prediction csv not found: ${PRED_CSV}"
    exit 1
  fi

  python "${BACKTEST_PY}" \
    --pred_csv "${PRED_CSV}" \
    --csv_root "${CSV_ROOT}" \
    --horizon_days "${HORIZON_DAYS}" \
    --weighting equal \
    --out_dir "${OUT_DIR}"

  echo "[OK] Finished backtest for ${EXP_NAME}"
}

# ============================================================
# 3 × 3 全部任务
# ============================================================

# I5 -> R5/R20/R60
run_one_task 5 0
run_one_task 5 1
run_one_task 5 2

# I20 -> R5/R20/R60
run_one_task 20 0
run_one_task 20 1
run_one_task 20 2

# I60 -> R5/R20/R60
run_one_task 60 0
run_one_task 60 1
run_one_task 60 2

echo
echo "============================================================"
echo "[ALL DONE] 9 CNN models have been trained and inferred."
echo "[INFO] Training outputs are under: ${TRAIN_ROOT}"
echo "[INFO] Prediction csvs are under: ${PRED_ROOT}"
echo "============================================================"

# ============================================================
# 9 个模型批量回测
# ============================================================

run_one_backtest 5 0
run_one_backtest 5 1
run_one_backtest 5 2

run_one_backtest 20 0
run_one_backtest 20 1
run_one_backtest 20 2

run_one_backtest 60 0
run_one_backtest 60 1
run_one_backtest 60 2

echo
echo "============================================================"
echo "[ALL DONE] 9 CNN models have been trained, inferred, and backtested."
echo "[INFO] Training outputs are under: ${TRAIN_ROOT}"
echo "[INFO] Prediction csvs are under: ${PRED_ROOT}"
echo "[INFO] Backtest outputs are under: ${BACKTEST_ROOT}"
echo "============================================================"