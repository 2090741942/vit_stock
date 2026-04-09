#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/workspace/vit_stock"

TRAIN_PY="${WORKDIR}/train_dinov2_square.py"
INFER_PY="${WORKDIR}/infer_test_dinov2_square.py"
BACKTEST_PY="${WORKDIR}/portfolio_backtest.py"

LABEL_DIR="/workspace/vit_stock_data/label_npz"
SYMBOL_TO_TAR_DIR="${WORKDIR}/symbol_to_tar"
CSV_ROOT="/workspace/stock_csv_8year"

HF_MODEL_NAME="facebook/dinov2-base-imagenet1k-1-layer"
PAD_MULTIPLE=14

EPOCHS=10
BATCH_SIZE=64
LR=1e-4
WEIGHT_DECAY=1e-4
NUM_WORKERS=8
SEED=20260315
WARMUP_RATIO=0.05
STEP_LOG_INTERVAL=100
STEP_LOG_FILE="metrics_step.csv"

TRAIN_ROOT="/workspace/dinov2_square_outs"
PRED_ROOT="/workspace/vit_stock_data/predict_dinov2_square"
BACKTEST_ROOT="/workspace/vit_stock_data/backtest_dinov2_square"
LOG_ROOT="/workspace/vit_stock_data/logs_dinov2_square"

mkdir -p "${TRAIN_ROOT}" "${PRED_ROOT}" "${BACKTEST_ROOT}" "${LOG_ROOT}"

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

  local EXP_NAME="I${WINDOW}_R${HORIZON_DAYS}_dinov2sq"
  local OUT_DIR="${TRAIN_ROOT}/${EXP_NAME}"
  local CKPT_PATH="${OUT_DIR}/best.pt"
  local OUT_CSV="${PRED_ROOT}/preds_${EXP_NAME}.csv"
  local BACKTEST_OUT_DIR="${BACKTEST_ROOT}/backtest_${EXP_NAME}_equal"
  local LOG_PATH="${LOG_ROOT}/${EXP_NAME}.log"

  mkdir -p "${OUT_DIR}" "${BACKTEST_OUT_DIR}"

  {
    python "${TRAIN_PY}"       --meta_path "${META_PATH}"       --symbol_to_tar_json "${SYMBOL_TO_TAR_JSON}"       --horizon_idx "${HORIZON_IDX}"       --hf_model_name "${HF_MODEL_NAME}"       --epochs "${EPOCHS}"       --batch_size "${BATCH_SIZE}"       --lr "${LR}"       --weight_decay "${WEIGHT_DECAY}"       --num_workers "${NUM_WORKERS}"       --seed "${SEED}"       --pad_multiple "${PAD_MULTIPLE}"       --repeat_to_3ch       --warmup_ratio "${WARMUP_RATIO}"       --step_log_interval "${STEP_LOG_INTERVAL}"       --step_log_file "${STEP_LOG_FILE}"       --out_dir "${OUT_DIR}"

    python "${INFER_PY}"       --meta_path "${META_PATH}"       --symbol_to_tar_json "${SYMBOL_TO_TAR_JSON}"       --checkpoint "${CKPT_PATH}"       --hf_model_name "${HF_MODEL_NAME}"       --horizon_idx "${HORIZON_IDX}"       --batch_size "${BATCH_SIZE}"       --num_workers "${NUM_WORKERS}"       --pad_multiple "${PAD_MULTIPLE}"       --repeat_to_3ch       --out_csv "${OUT_CSV}"

    python "${BACKTEST_PY}"       --pred_csv "${OUT_CSV}"       --csv_root "${CSV_ROOT}"       --horizon_days "${HORIZON_DAYS}"       --weighting equal       --out_dir "${BACKTEST_OUT_DIR}"
  } 2>&1 | tee "${LOG_PATH}"
}

# run_one_task 5 0
# run_one_task 5 1
# run_one_task 5 2
# run_one_task 20 0
# run_one_task 20 1
# run_one_task 20 2
# run_one_task 60 0
run_one_task 60 1
run_one_task 60 2
