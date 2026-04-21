#!/usr/bin/env bash
set -euo pipefail

# =========================
# DINOv3: 训练 + 预测 + 回测（9个组合）
# 口径：
# 1) 全市场，不分市场
# 2) 等权
# 3) 不考虑交易成本
# 4) 不筛 top500 市值
# 5) 终端只打印 H-L 的年化收益和换手
# =========================

# 如需切换环境，可取消下一行注释
# source /workspace/vit_stock/env.sh

ROOT="/workspace"
PROJ_DIR="${ROOT}/vit_stock"
DATA_DIR="${ROOT}/vit_stock_data"

TRAIN_PY="${PROJ_DIR}/dinov3/train_dinov3_rect.py"
INFER_PY="${PROJ_DIR}/dinov3/infer_test_dinov3_rect.py"

# 兼容两种可能的回测脚本命名
if [[ -f "${PROJ_DIR}/backtest/portfolio_backtest_with_costs_turnover_topn.py" ]]; then
  BACKTEST_PY="${PROJ_DIR}/backtest/portfolio_backtest_with_costs_turnover_topn.py"
elif [[ -f "${PROJ_DIR}/backtest/portfolio_backtest.py" ]]; then
  BACKTEST_PY="${PROJ_DIR}/backtest/portfolio_backtest.py"
else
  echo "[ERROR] 未找到回测脚本。请确认以下任一路径存在："
  echo "  ${PROJ_DIR}/backtest/portfolio_backtest_with_costs_turnover_topn.py"
  echo "  ${PROJ_DIR}/backtest/portfolio_backtest.py"
  exit 1
fi

CSV_ROOT="${ROOT}/stock_csv_8year"
LABEL_DIR="${DATA_DIR}/label_npz"
SYMBOL_TAR_DIR="${PROJ_DIR}/symbol_to_tar"

EXP_ROOT="${DATA_DIR}/exp_dinov3"
PRED_ROOT="${DATA_DIR}/predict_dinov3"
BACKTEST_ROOT="${DATA_DIR}/backtest_dinov3"

mkdir -p "${EXP_ROOT}" "${PRED_ROOT}" "${BACKTEST_ROOT}"

# ===== 可调训练参数 =====
HF_MODEL_NAME="facebook/dinov3-vits16-pretrain-lvd1689m"
EPOCHS=1
BATCH_SIZE_TRAIN=64
BATCH_SIZE_INFER=128
LR=1e-4
WEIGHT_DECAY=1e-4
NUM_WORKERS_TRAIN=12
NUM_WORKERS_INFER=8
SEED=20260315
WARMUP_RATIO=0.05
SAVE_LAST=1
VALIDATE_MEMBERS=0
POOL_MODE="cls_mean"

# ===== 任务定义 =====
declare -A HIDX=( [5]=0 [20]=1 [60]=2 )
WINDOWS=(5 20 60)
HORIZONS=(5 20 60)

run_one() {
  local img_days="$1"
  local ret_days="$2"

  local horizon_idx="${HIDX[$ret_days]}"
  local meta_path="${LABEL_DIR}/meta_N${img_days}_clean.npz"
  local symbol_json="${SYMBOL_TAR_DIR}/symbol_to_tar_N${img_days}.json"

  local exp_dir="${EXP_ROOT}/I${img_days}_R${ret_days}_dinov3"
  local pred_csv="${PRED_ROOT}/preds_I${img_days}_R${ret_days}_dinov3.csv"
  local bt_dir="${BACKTEST_ROOT}/backtest_I${img_days}_R${ret_days}_dinov3_equal"
  local ckpt_path="${exp_dir}/best.pt"

  echo
  echo "============================================================"
  echo "[TASK] I${img_days} / R${ret_days}"
  echo "============================================================"

  if [[ ! -f "${meta_path}" ]]; then
    echo "[ERROR] meta 不存在: ${meta_path}"
    exit 1
  fi
  if [[ ! -f "${symbol_json}" ]]; then
    echo "[ERROR] symbol_to_tar 不存在: ${symbol_json}"
    exit 1
  fi

  echo "[1/3] Train"
  python "${TRAIN_PY}" \
    --meta_path "${meta_path}" \
    --symbol_to_tar_json "${symbol_json}" \
    --out_dir "${exp_dir}" \
    --hf_model_name "${HF_MODEL_NAME}" \
    --horizon_idx "${horizon_idx}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE_TRAIN}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --num_workers "${NUM_WORKERS_TRAIN}" \
    --seed "${SEED}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --pool_mode "${POOL_MODE}" \
    --repeat_to_3ch

  if [[ "${SAVE_LAST}" -eq 1 ]]; then
    python - <<PY
from pathlib import Path
p = Path(r"${exp_dir}")
last = p / "last.pt"
if last.exists():
    print(f"[INFO] last checkpoint exists: {last}")
PY
  fi

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[ERROR] 训练完成后未找到 best.pt: ${ckpt_path}"
    exit 1
  fi

  echo "[2/3] Infer"
  infer_cmd=(
    python "${INFER_PY}"
    --meta_path "${meta_path}"
    --symbol_to_tar_json "${symbol_json}"
    --checkpoint "${ckpt_path}"
    --hf_model_name "${HF_MODEL_NAME}"
    --horizon_idx "${horizon_idx}"
    --batch_size "${BATCH_SIZE_INFER}"
    --num_workers "${NUM_WORKERS_INFER}"
    --pool_mode "${POOL_MODE}"
    --repeat_to_3ch
    --out_csv "${pred_csv}"
  )
  if [[ "${VALIDATE_MEMBERS}" -eq 1 ]]; then
    infer_cmd+=(--validate_members)
  fi
  "${infer_cmd[@]}"

  if [[ ! -f "${pred_csv}" ]]; then
    echo "[ERROR] 推理完成后未找到预测文件: ${pred_csv}"
    exit 1
  fi

  echo "[3/3] Backtest"
  python "${BACKTEST_PY}" \
    --pred_csv "${pred_csv}" \
    --csv_root "${CSV_ROOT}" \
    --horizon_days "${ret_days}" \
    --weighting equal \
    --top_n_by_cap 0 \
    --buy_cost_rate 0 \
    --sell_cost_rate 0 \
    --out_dir "${bt_dir}"

  local summary_csv="${bt_dir}/summary_equal.csv"
  if [[ ! -f "${summary_csv}" ]]; then
    echo "[ERROR] 未找到回测汇总文件: ${summary_csv}"
    exit 1
  fi

  echo
  echo "[RESULT] I${img_days} / R${ret_days}  (仅显示 H-L 年化收益与换手)"
  python - <<PY
import pandas as pd
p = r"${summary_csv}"
df = pd.read_csv(p)
row = df.loc[df["group"].astype(str) == "H-L"].copy()
if row.empty:
    print(f"[WARN] {p} 中没有 H-L 行")
else:
    cols = [c for c in ["group", "AnnualRet", "AnnualRet_pct", "Turnover", "Turnover_pct"] if c in row.columns]
    print(row[cols].to_string(index=False))
PY
}

main() {
  echo "ROOT         = ${ROOT}"
  echo "PROJ_DIR     = ${PROJ_DIR}"
  echo "DATA_DIR     = ${DATA_DIR}"
  echo "TRAIN_PY     = ${TRAIN_PY}"
  echo "INFER_PY     = ${INFER_PY}"
  echo "BACKTEST_PY  = ${BACKTEST_PY}"
  echo "CSV_ROOT     = ${CSV_ROOT}"
  echo "LABEL_DIR    = ${LABEL_DIR}"
  echo "SYMBOL_TAR   = ${SYMBOL_TAR_DIR}"
  echo "EXP_ROOT     = ${EXP_ROOT}"
  echo "PRED_ROOT    = ${PRED_ROOT}"
  echo "BACKTESTROOT = ${BACKTEST_ROOT}"

  for img_days in "${WINDOWS[@]}"; do
    for ret_days in "${HORIZONS[@]}"; do
      run_one "${img_days}" "${ret_days}"
    done
  done

  echo
  echo "================ ALL DONE ================"
  echo "实验目录: ${EXP_ROOT}"
  echo "预测目录: ${PRED_ROOT}"
  echo "回测目录: ${BACKTEST_ROOT}"
}

main "$@"
