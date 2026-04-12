#!/usr/bin/env bash
set -euo pipefail

# =========================
# DINOv3: 训练 + 推理 + 回测（9组）
# 口径：
# - 全市场
# - 等权
# - 不考虑交易成本
# - 不取前500市值
# - 保留回测脚本原有输出逻辑
#
# 关键修改：
# - 统一从 /workspace 启动
# - 训练/推理改为 python -m vit_stock....
#   以适配 from vit_stock.xxx import ... 的绝对包导入
# =========================

cd /workspace

# ===== 路径配置 =====
PROJECT_ROOT="/workspace/vit_stock"
DATA_ROOT="/workspace/vit_stock_data"
CSV_ROOT="/workspace/stock_csv_8year"

TRAIN_MODULE="vit_stock.dinov3.train_dinov3_rect"
INFER_MODULE="vit_stock.dinov3.infer_test_dinov3_rect"

BACKTEST_SCRIPT=""
if [[ -f "/workspace/vit_stock/backtest/portfolio_backtest.py" ]]; then
  BACKTEST_SCRIPT="/workspace/vit_stock/backtest/portfolio_backtest.py"
elif [[ -f "/workspace/vit_stock/backtest/portfolio_backtest_with_costs_turnover_topn.py" ]]; then
  BACKTEST_SCRIPT="/workspace/vit_stock/backtest/portfolio_backtest_with_costs_turnover_topn.py"
else
  echo "[ERROR] backtest script not found under /workspace/vit_stock/backtest" >&2
  exit 1
fi

# ===== 可按需调整的训练参数 =====
HF_MODEL_NAME="facebook/dinov3-vits16-pretrain-lvd1689m"
EPOCHS=1
BATCH_SIZE_TRAIN=64
BATCH_SIZE_INFER=128
LR=1e-4
WEIGHT_DECAY=1e-4
NUM_WORKERS_TRAIN=12
NUM_WORKERS_INFER=8
SEED=20260315
SAVE_LAST=1
POOL_MODE="cls_mean_patch"

# ===== 固定参数 =====
declare -A HORIZON_IDX
HORIZON_IDX[5]=0
HORIZON_IDX[20]=1
HORIZON_IDX[60]=2

WINDOWS=(5 20 60)
HORIZONS=(5 20 60)

mkdir -p "${DATA_ROOT}/predict_dinov3"
mkdir -p "${DATA_ROOT}/backtest_dinov3"
mkdir -p "${DATA_ROOT}/exp_dinov3"

for WIN in "${WINDOWS[@]}"; do
  META_PATH="${DATA_ROOT}/label_npz/meta_N${WIN}_clean.npz"
  SYMBOL_TO_TAR_JSON="${PROJECT_ROOT}/symbol_to_tar/symbol_to_tar_N${WIN}.json"

  if [[ ! -f "${META_PATH}" ]]; then
    echo "[ERROR] meta file not found: ${META_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${SYMBOL_TO_TAR_JSON}" ]]; then
    echo "[ERROR] symbol_to_tar json not found: ${SYMBOL_TO_TAR_JSON}" >&2
    exit 1
  fi

  for HOR in "${HORIZONS[@]}"; do
    HIDX="${HORIZON_IDX[$HOR]}"

    EXP_DIR="${DATA_ROOT}/exp_dinov3/exp_I${WIN}_R${HOR}_dinov3"
    PRED_CSV="${DATA_ROOT}/predict_dinov3/preds_I${WIN}_R${HOR}_dinov3.csv"
    BACKTEST_DIR="${DATA_ROOT}/backtest_dinov3/backtest_I${WIN}_R${HOR}_dinov3_equal"

    mkdir -p "${EXP_DIR}"
    mkdir -p "${BACKTEST_DIR}"

    echo "============================================================"
    echo "[RUN] Train I${WIN} -> R${HOR}"
    echo "============================================================"

    python -m "${TRAIN_MODULE}" \
      --meta_path "${META_PATH}" \
      --symbol_to_tar_json "${SYMBOL_TO_TAR_JSON}" \
      --out_dir "${EXP_DIR}" \
      --horizon_idx "${HIDX}" \
      --hf_model_name "${HF_MODEL_NAME}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE_TRAIN}" \
      --lr "${LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --num_workers "${NUM_WORKERS_TRAIN}" \
      --seed "${SEED}" \
      --pool_mode "${POOL_MODE}" \
      $( [[ "${SAVE_LAST}" -eq 1 ]] && echo "--save_last" )

    BEST_CKPT="${EXP_DIR}/best.pt"
    if [[ ! -f "${BEST_CKPT}" ]]; then
      echo "[ERROR] checkpoint not found: ${BEST_CKPT}" >&2
      exit 1
    fi

    echo "============================================================"
    echo "[RUN] Infer I${WIN} -> R${HOR}"
    echo "============================================================"

    python -m "${INFER_MODULE}" \
      --meta_path "${META_PATH}" \
      --symbol_to_tar_json "${SYMBOL_TO_TAR_JSON}" \
      --checkpoint "${BEST_CKPT}" \
      --horizon_idx "${HIDX}" \
      --hf_model_name "${HF_MODEL_NAME}" \
      --batch_size "${BATCH_SIZE_INFER}" \
      --num_workers "${NUM_WORKERS_INFER}" \
      --pool_mode "${POOL_MODE}" \
      --out_csv "${PRED_CSV}"

    if [[ ! -f "${PRED_CSV}" ]]; then
      echo "[ERROR] prediction csv not found: ${PRED_CSV}" >&2
      exit 1
    fi

    echo "============================================================"
    echo "[RUN] Backtest I${WIN} -> R${HOR}"
    echo "============================================================"

    python "${BACKTEST_SCRIPT}" \
      --pred_csv "${PRED_CSV}" \
      --csv_root "${CSV_ROOT}" \
      --out_dir "${BACKTEST_DIR}" \
      --horizon_days "${HOR}" \
      --weighting equal \
      --top_n_by_cap 0 \
      --buy_cost_rate 0 \
      --sell_cost_rate 0 \
      --cap_col CirculatedMarketValue

    echo "[DONE] I${WIN} -> R${HOR}"
    echo
  done
done

echo "All 9 DINOv3 train + infer + backtest jobs finished."
