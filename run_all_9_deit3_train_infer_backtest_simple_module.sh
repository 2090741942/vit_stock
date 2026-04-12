#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON_BIN="${PYTHON_BIN:-python}"

# ===== 路径配置 =====
PROJECT_ROOT="/workspace/vit_stock"
DATA_ROOT="/workspace/vit_stock_data"
CSV_ROOT="/workspace/stock_csv_8year"

TRAIN_MODULE="vit_stock.deit3.train_deit3_rect"
INFER_MODULE="vit_stock.deit3.infer_test_deit3_rect"

BACKTEST_PY=""
if [[ -f "${PROJECT_ROOT}/backtest/portfolio_backtest.py" ]]; then
  BACKTEST_PY="${PROJECT_ROOT}/backtest/portfolio_backtest.py"
elif [[ -f "${PROJECT_ROOT}/backtest/portfolio_backtest_with_costs_turnover_topn.py" ]]; then
  BACKTEST_PY="${PROJECT_ROOT}/backtest/portfolio_backtest_with_costs_turnover_topn.py"
else
  echo "[ERROR] backtest script not found under ${PROJECT_ROOT}/backtest"
  exit 1
fi

# ===== 训练超参数 =====
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-12}"
SEED="${SEED:-20260315}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
DROPOUT="${DROPOUT:-0.0}"
STEP_LOG_INTERVAL="${STEP_LOG_INTERVAL:-100}"
POOL_MODE="${POOL_MODE:-cls_mean_patch}"

# ===== 推理超参数 =====
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-128}"
INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-8}"

# ===== 模型名 =====
HF_MODEL_NAME="${HF_MODEL_NAME:-deit3_base_patch16_224.fb_in22k_ft_in1k}"

# ===== 输出目录 =====
EXP_ROOT="${DATA_ROOT}/exp_deit3"
PRED_ROOT="${DATA_ROOT}/predict_deit3"
BACKTEST_ROOT="${DATA_ROOT}/backtest_deit3"

mkdir -p "${EXP_ROOT}" "${PRED_ROOT}" "${BACKTEST_ROOT}"

# horizon_idx -> R days
declare -A HORIZON_DAYS
HORIZON_DAYS[0]=5
HORIZON_DAYS[1]=20
HORIZON_DAYS[2]=60

# image window -> meta/json
declare -A META_PATHS
META_PATHS[5]="${DATA_ROOT}/label_npz/meta_N5_clean.npz"
META_PATHS[20]="${DATA_ROOT}/label_npz/meta_N20_clean.npz"
META_PATHS[60]="${DATA_ROOT}/label_npz/meta_N60_clean.npz"

declare -A TAR_JSONS
TAR_JSONS[5]="${PROJECT_ROOT}/symbol_to_tar/symbol_to_tar_N5.json"
TAR_JSONS[20]="${PROJECT_ROOT}/symbol_to_tar/symbol_to_tar_N20.json"
TAR_JSONS[60]="${PROJECT_ROOT}/symbol_to_tar/symbol_to_tar_N60.json"

for IMG_WINDOW in 5 20 60; do
  META_PATH="${META_PATHS[$IMG_WINDOW]}"
  TAR_JSON="${TAR_JSONS[$IMG_WINDOW]}"

  if [[ ! -f "${META_PATH}" ]]; then
    echo "[ERROR] meta file not found: ${META_PATH}"
    exit 1
  fi
  if [[ ! -f "${TAR_JSON}" ]]; then
    echo "[ERROR] symbol_to_tar json not found: ${TAR_JSON}"
    exit 1
  fi

  for HIDX in 0 1 2; do
    R_DAYS="${HORIZON_DAYS[$HIDX]}"

    EXP_DIR="${EXP_ROOT}/deit3_I${IMG_WINDOW}_R${R_DAYS}"
    PRED_CSV="${PRED_ROOT}/preds_I${IMG_WINDOW}_R${R_DAYS}_deit3.csv"
    BT_DIR="${BACKTEST_ROOT}/backtest_I${IMG_WINDOW}_R${R_DAYS}_deit3_equal"

    echo "============================================================"
    echo "[RUN] Train I${IMG_WINDOW} -> R${R_DAYS}"
    echo "============================================================"

    "${PYTHON_BIN}" -m "${TRAIN_MODULE}" \
      --meta_path "${META_PATH}" \
      --symbol_to_tar_json "${TAR_JSON}" \
      --out_dir "${EXP_DIR}" \
      --hf_model_name "${HF_MODEL_NAME}" \
      --horizon_idx "${HIDX}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --num_workers "${NUM_WORKERS}" \
      --seed "${SEED}" \
      --warmup_ratio "${WARMUP_RATIO}" \
      --dropout "${DROPOUT}" \
      --pool_mode "${POOL_MODE}" \
      --step_log_interval "${STEP_LOG_INTERVAL}" \
      --save_last

    echo "============================================================"
    echo "[RUN] Infer I${IMG_WINDOW} -> R${R_DAYS}"
    echo "============================================================"

    "${PYTHON_BIN}" -m "${INFER_MODULE}" \
      --meta_path "${META_PATH}" \
      --symbol_to_tar_json "${TAR_JSON}" \
      --checkpoint "${EXP_DIR}/best.pt" \
      --hf_model_name "${HF_MODEL_NAME}" \
      --horizon_idx "${HIDX}" \
      --batch_size "${INFER_BATCH_SIZE}" \
      --num_workers "${INFER_NUM_WORKERS}" \
      --out_csv "${PRED_CSV}"

    echo "============================================================"
    echo "[RUN] Backtest I${IMG_WINDOW} -> R${R_DAYS}"
    echo "============================================================"

    "${PYTHON_BIN}" "${BACKTEST_PY}" \
      --pred_csv "${PRED_CSV}" \
      --csv_root "${CSV_ROOT}" \
      --out_dir "${BT_DIR}" \
      --horizon_days "${R_DAYS}" \
      --weighting equal \
      --top_n_by_cap 0 \
      --buy_cost_rate 0 \
      --sell_cost_rate 0 \
      --cap_col CirculatedMarketValue

  done
done

echo "============================================================"
echo "[DONE] All 9 DeiT III runs finished."
echo "Train outputs:   ${EXP_ROOT}"
echo "Prediction csvs: ${PRED_ROOT}"
echo "Backtests:       ${BACKTEST_ROOT}"
echo "============================================================"
