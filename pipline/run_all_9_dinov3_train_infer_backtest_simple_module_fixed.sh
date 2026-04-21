#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON_BIN="python"

PROJECT_ROOT="/workspace/vit_stock"
DATA_ROOT="/workspace/vit_stock_data"
CSV_ROOT="/workspace/stock_csv_8year"

TRAIN_MODULE="vit_stock.dinov3.train_dinov3_rect"
INFER_MODULE="vit_stock.dinov3.infer_test_dinov3_rect"
BACKTEST_SCRIPT="/workspace/vit_stock/backtest/portfolio_backtest.py"

if [[ ! -f "$BACKTEST_SCRIPT" ]]; then
  echo "[ERROR] backtest script not found: $BACKTEST_SCRIPT"
  exit 1
fi

WINDOWS=(5 20 60)
HORIZONS=(5 20 60)

# train/infer hyperparams
EPOCHS=1
BATCH_SIZE_TRAIN=64
BATCH_SIZE_INFER=128
LR=1e-4
WEIGHT_DECAY=1e-4
NUM_WORKERS_TRAIN=12
NUM_WORKERS_INFER=8
SEED=20260315
WARMUP_RATIO=0.05
DROPOUT=0.0
POOL_MODE="cls_mean_patch"

mkdir -p "$DATA_ROOT/predict_dinov3"
mkdir -p "$DATA_ROOT/backtest_dinov3"
mkdir -p "$DATA_ROOT/exp_dinov3"

horizon_to_idx() {
  case "$1" in
    5) echo 0 ;;
    20) echo 1 ;;
    60) echo 2 ;;
    *) echo "invalid horizon: $1" >&2; exit 1 ;;
  esac
}

for I in "${WINDOWS[@]}"; do
  META_PATH="$DATA_ROOT/label_npz/meta_N${I}_clean.npz"
  SYMBOL_TO_TAR_JSON="$PROJECT_ROOT/symbol_to_tar/symbol_to_tar_N${I}.json"

  if [[ ! -f "$META_PATH" ]]; then
    echo "[ERROR] meta not found: $META_PATH"
    exit 1
  fi
  if [[ ! -f "$SYMBOL_TO_TAR_JSON" ]]; then
    echo "[ERROR] symbol_to_tar json not found: $SYMBOL_TO_TAR_JSON"
    exit 1
  fi

  for R in "${HORIZONS[@]}"; do
    HORIZON_IDX="$(horizon_to_idx "$R")"
    EXP_DIR="$DATA_ROOT/exp_dinov3/exp_I${I}_R${R}_dinov3"
    PRED_CSV="$DATA_ROOT/predict_dinov3/preds_I${I}_R${R}_dinov3.csv"
    BACKTEST_OUT="$DATA_ROOT/backtest_dinov3/backtest_I${I}_R${R}_dinov3_equal"

    echo "============================================================"
    echo "[RUN] Train I${I} -> R${R}"
    echo "============================================================"

    "$PYTHON_BIN" -m "$TRAIN_MODULE" \
      --meta_path "$META_PATH" \
      --symbol_to_tar_json "$SYMBOL_TO_TAR_JSON" \
      --out_dir "$EXP_DIR" \
      --horizon_idx "$HORIZON_IDX" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE_TRAIN" \
      --lr "$LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --num_workers "$NUM_WORKERS_TRAIN" \
      --seed "$SEED" \
      --warmup_ratio "$WARMUP_RATIO" \
      --dropout "$DROPOUT" \
      --pool_mode "$POOL_MODE" \
      --save_last

    CKPT_PATH="$EXP_DIR/best.pt"
    if [[ ! -f "$CKPT_PATH" ]]; then
      echo "[ERROR] checkpoint not found: $CKPT_PATH"
      exit 1
    fi

    echo "============================================================"
    echo "[RUN] Infer I${I} -> R${R}"
    echo "============================================================"

    "$PYTHON_BIN" -m "$INFER_MODULE" \
      --meta_path "$META_PATH" \
      --symbol_to_tar_json "$SYMBOL_TO_TAR_JSON" \
      --checkpoint "$CKPT_PATH" \
      --horizon_idx "$HORIZON_IDX" \
      --batch_size "$BATCH_SIZE_INFER" \
      --num_workers "$NUM_WORKERS_INFER" \
      --out_csv "$PRED_CSV"

    echo "============================================================"
    echo "[RUN] Backtest I${I} -> R${R}"
    echo "============================================================"

    "$PYTHON_BIN" "$BACKTEST_SCRIPT" \
      --pred_csv "$PRED_CSV" \
      --csv_root "$CSV_ROOT" \
      --out_dir "$BACKTEST_OUT" \
      --horizon_days "$R" \
      --weighting equal \
      --top_n_by_cap 0 \
      --buy_cost_rate 0 \
      --sell_cost_rate 0 \
      --cap_col CirculatedMarketValue
  done
done

echo "All done."
