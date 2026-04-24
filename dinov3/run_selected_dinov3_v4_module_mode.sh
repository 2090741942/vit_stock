#!/usr/bin/env bash
set -euo pipefail

cd /workspace

###############################################################################
# 手动配置区：只在这里改
###############################################################################

PYTHON_BIN="python"

PROJECT_ROOT="/workspace/vit_stock"
DATA_ROOT="/workspace/vit_stock_data"
CSV_ROOT="/workspace/stock_csv_8year"

# 你的项目需要用 -m 模块导入方式运行
TRAIN_MODULE="vit_stock.dinov3.train_dinov3_final"
INFER_MODULE="vit_stock.dinov3.infer_test_dinov3_rect_fixed_v2"

BACKTEST_SCRIPT="${PROJECT_ROOT}/backtest/portfolio_backtest_summary_only_with_turnover_all_rows.py"

HF_MODEL_NAME="facebook/dinov3-vits16-pretrain-lvd1689m"
EPOCHS=10
BATCH_SIZE_TRAIN=32
BATCH_SIZE_INFER=64
LR=1e-4
WEIGHT_DECAY=1e-4
NUM_WORKERS_TRAIN=2
NUM_WORKERS_INFER=2
SEED=20260315
SAVE_LAST=1
POOL_MODE="cls_mean_patch"
DROPOUT=0.0
WARMUP_RATIO=0.05
STEP_LOG_INTERVAL=5000
VALIDATE_MEMBERS=0

# 选择要执行哪些阶段：1=执行，0=跳过
RUN_TRAIN=1
RUN_INFER=1
RUN_BACKTEST=1

# 是否跳过已经完成的 infer / backtest
# 训练阶段这里不做断点续跑逻辑：只要 RUN_TRAIN=1 就重新训练
SKIP_EXISTING_INFER=1
SKIP_EXISTING_BACKTEST=1

# 在这里直接写要跑的组合，空格分隔
RUN_CONFIGS=(
  "I20R60" "I60R5" "I60R20" "I60R60"
)

###############################################################################
# 固定逻辑区：一般不用改
###############################################################################

declare -A HORIZON_IDX
HORIZON_IDX[5]=0
HORIZON_IDX[20]=1
HORIZON_IDX[60]=2

mkdir -p "${DATA_ROOT}/predict_dinov3"
mkdir -p "${DATA_ROOT}/backtest_dinov3"
mkdir -p "${DATA_ROOT}/exp_dinov3"

if [[ ! -f "${BACKTEST_SCRIPT}" ]]; then
  echo "[ERROR] backtest script not found: ${BACKTEST_SCRIPT}" >&2
  exit 1
fi

parse_config() {
  local cfg="$1"
  if [[ "${cfg}" =~ ^I([0-9]+)R([0-9]+)$ ]]; then
    WIN="${BASH_REMATCH[1]}"
    HOR="${BASH_REMATCH[2]}"
  else
    echo "[ERROR] invalid config format: ${cfg}, expected like I20R60" >&2
    exit 1
  fi
}

for CFG in "${RUN_CONFIGS[@]}"; do
  parse_config "${CFG}"

  if [[ -z "${HORIZON_IDX[$HOR]:-}" ]]; then
    echo "[ERROR] unsupported horizon: ${HOR}" >&2
    exit 1
  fi

  HIDX="${HORIZON_IDX[$HOR]}"

  META_PATH="${DATA_ROOT}/label_npz/meta_N${WIN}_clean.npz"
  SYMBOL_TO_TAR_JSON="${PROJECT_ROOT}/symbol_to_tar/symbol_to_tar_N${WIN}.json"

  EXP_DIR="${DATA_ROOT}/exp_dinov3/exp_I${WIN}_R${HOR}_dinov3"
  PRED_CSV="${DATA_ROOT}/predict_dinov3/preds_I${WIN}_R${HOR}_dinov3.csv"
  BACKTEST_DIR="${DATA_ROOT}/backtest_dinov3/backtest_I${WIN}_R${HOR}_dinov3_equal"
  BACKTEST_DONE_FILE="${BACKTEST_DIR}/.done"

  BEST_CKPT="${EXP_DIR}/best.pt"

  mkdir -p "${EXP_DIR}"
  mkdir -p "${BACKTEST_DIR}"

  if [[ ! -f "${META_PATH}" ]]; then
    echo "[ERROR] meta file not found: ${META_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${SYMBOL_TO_TAR_JSON}" ]]; then
    echo "[ERROR] symbol_to_tar json not found: ${SYMBOL_TO_TAR_JSON}" >&2
    exit 1
  fi

  echo "============================================================"
  echo "[CONFIG] ${CFG}"
  echo "train_module : ${TRAIN_MODULE}"
  echo "infer_module : ${INFER_MODULE}"
  echo "meta_path    : ${META_PATH}"
  echo "exp_dir      : ${EXP_DIR}"
  echo "pred_csv     : ${PRED_CSV}"
  echo "backtest_dir : ${BACKTEST_DIR}"
  echo "============================================================"

  if [[ "${RUN_TRAIN}" -eq 1 ]]; then
    echo "============================================================"
    echo "[RUN] Train ${CFG}"
    echo "============================================================"

    TRAIN_CMD=(
      "${PYTHON_BIN}" -m "${TRAIN_MODULE}"
      --meta_path "${META_PATH}"
      --symbol_to_tar_json "${SYMBOL_TO_TAR_JSON}"
      --out_dir "${EXP_DIR}"
      --horizon_idx "${HIDX}"
      --hf_model_name "${HF_MODEL_NAME}"
      --epochs "${EPOCHS}"
      --batch_size "${BATCH_SIZE_TRAIN}"
      --lr "${LR}"
      --weight_decay "${WEIGHT_DECAY}"
      --num_workers "${NUM_WORKERS_TRAIN}"
      --seed "${SEED}"
      --warmup_ratio "${WARMUP_RATIO}"
      --dropout "${DROPOUT}"
      --pool_mode "${POOL_MODE}"
      --step_log_interval "${STEP_LOG_INTERVAL}"
    )

    if [[ "${SAVE_LAST}" -eq 1 ]]; then
      TRAIN_CMD+=(--save_last)
    fi
    if [[ "${VALIDATE_MEMBERS}" -eq 1 ]]; then
      TRAIN_CMD+=(--validate_members)
    fi

    "${TRAIN_CMD[@]}"

    if [[ ! -f "${BEST_CKPT}" ]]; then
      echo "[ERROR] checkpoint not found after training: ${BEST_CKPT}" >&2
      exit 1
    fi
  else
    echo "[SKIP] Train ${CFG}"
  fi

  if [[ "${RUN_INFER}" -eq 1 ]]; then
    if [[ "${SKIP_EXISTING_INFER}" -eq 1 && -f "${PRED_CSV}" ]]; then
      echo "[SKIP] Infer ${CFG} because prediction csv already exists: ${PRED_CSV}"
    else
      if [[ ! -f "${BEST_CKPT}" ]]; then
        echo "[ERROR] checkpoint not found for infer: ${BEST_CKPT}" >&2
        exit 1
      fi

      echo "============================================================"
      echo "[RUN] Infer ${CFG}"
      echo "============================================================"

      "${PYTHON_BIN}" -m "${INFER_MODULE}" \
        --meta_path "${META_PATH}" \
        --symbol_to_tar_json "${SYMBOL_TO_TAR_JSON}" \
        --checkpoint "${BEST_CKPT}" \
        --horizon_idx "${HIDX}" \
        --hf_model_name "${HF_MODEL_NAME}" \
        --batch_size "${BATCH_SIZE_INFER}" \
        --num_workers "${NUM_WORKERS_INFER}" \
        --pool_mode "${POOL_MODE}" \
        --dropout "${DROPOUT}" \
        --out_csv "${PRED_CSV}"

      if [[ ! -f "${PRED_CSV}" ]]; then
        echo "[ERROR] prediction csv not found after infer: ${PRED_CSV}" >&2
        exit 1
      fi
    fi
  else
    echo "[SKIP] Infer ${CFG}"
  fi

  if [[ "${RUN_BACKTEST}" -eq 1 ]]; then
    if [[ "${SKIP_EXISTING_BACKTEST}" -eq 1 && -f "${BACKTEST_DONE_FILE}" ]]; then
      echo "[SKIP] Backtest ${CFG} because done file already exists: ${BACKTEST_DONE_FILE}"
    else
      if [[ ! -f "${PRED_CSV}" ]]; then
        echo "[ERROR] pred csv not found for backtest: ${PRED_CSV}" >&2
        exit 1
      fi

      echo "============================================================"
      echo "[RUN] Backtest ${CFG}"
      echo "============================================================"

      rm -f "${BACKTEST_DONE_FILE}"

      "${PYTHON_BIN}" "${BACKTEST_SCRIPT}" \
        --pred_csv "${PRED_CSV}" \
        --csv_root "${CSV_ROOT}" \
        --out_dir "${BACKTEST_DIR}" \
        --horizon_days "${HOR}" \
        --weighting equal \
        --top_n_by_cap 0 \
        --buy_cost_rate 0 \
        --sell_cost_rate 0 \
        --cap_col CirculatedMarketValue

      touch "${BACKTEST_DONE_FILE}"
    fi
  else
    echo "[SKIP] Backtest ${CFG}"
  fi

  echo "[DONE] ${CFG}"
  echo
done

echo "Selected DINOv3 jobs finished."
