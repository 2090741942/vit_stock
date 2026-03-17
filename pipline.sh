# # 给出股票序号与tar的映射

# python /workspace/build_symbol_to_tar.py \
#     --tar_dir /workspace/ohlc_image_N20/packs \
#     --out_json /workspace/symbol_to_tar/symbol_to_tar_N20.json

# python /workspace/build_symbol_to_tar.py \
#     --tar_dir /workspace/ohlc_image_N5/packs \
#     --out_json /workspace/symbol_to_tar/symbol_to_tar_N5.json

# # 清除 meta数据中存在label的日期在 tar 包里找不到对应成员的样本，输出新的 clean npz。
# python clean_meta_by_tar.py \
#     --meta_path /workspace/label_npz/meta_N20.npz \
#     --symbol_to_tar_json /workspace/symbol_to_tar/symbol_to_tar_N20.json \
#     --out_meta_path /workspace/label_npz/meta_N20_clean.npz

# python clean_meta_by_tar.py \
#     --meta_path /workspace/label_npz/meta_N5.npz \
#     --symbol_to_tar_json /workspace/symbol_to_tar/symbol_to_tar_N5.json \
#     --out_meta_path /workspace/label_npz/meta_N5_clean.npz

# # 训练
# python train.py \
#   --meta_path /workspace/label_npz/meta_N20_clean.npz \
#   --symbol_to_tar_json /workspace/symbol_to_tar/symbol_to_tar_N20.json \
#   --horizon_idx 0 \
#   --hf_model_name google/vit-base-patch16-224-in21k \
#   --epochs 5 \
#   --batch_size 64 \
#   --lr 1e-4 \
#   --weight_decay 1e-4 \
#   --num_workers 12 \
#   --seed 20260315 \
#   --patch_size 16 \
#   --repeat_to_3ch \
#   --warmup_ratio 0.05 \
#   --step_log_interval 200 \
#   --step_log_file metrics_step.csv \
#   --out_dir /workspace/exp_I20_R5_vitb_in21k

# python train.py \
#   --meta_path /workspace/label_npz/meta_N5_clean.npz \
#   --symbol_to_tar_json /workspace/symbol_to_tar/symbol_to_tar_N5.json \
#   --horizon_idx 0 \
#   --hf_model_name google/vit-base-patch16-224-in21k \
#   --epochs 5 \
#   --batch_size 64 \
#   --lr 1e-4 \
#   --weight_decay 1e-4 \
#   --num_workers 12 \
#   --seed 20260315 \
#   --patch_size 16 \
#   --repeat_to_3ch \
#   --warmup_ratio 0.05 \
#   --step_log_interval 200 \
#   --step_log_file metrics_step.csv \
#   --out_dir /workspace/exp_I5_R5_vitb_in21k

# # 预测
# python infer_test.py \
#   --meta_path /workspace/label_npz/meta_N20_clean.npz \
#   --symbol_to_tar_json /workspace/symbol_to_tar/symbol_to_tar_N20.json \
#   --checkpoint /workspace/exp_I20_R5_vitb_in21k/best.pt \
#   --horizon_idx 0 \
#   --batch_size 128 \
#   --num_workers 12 \
#   --patch_size 16 \
#   --repeat_to_3ch \
#   --out_csv /workspace/predict/preds_I20_R5.csv

# python infer_test.py \
#   --meta_path /workspace/label_npz/meta_N5_clean.npz \
#   --symbol_to_tar_json /workspace/symbol_to_tar/symbol_to_tar_N5.json \
#   --checkpoint /workspace/exp_I5_R5_vitb_in21k/best.pt \
#   --horizon_idx 0 \
#   --batch_size 128 \
#   --num_workers 12 \
#   --patch_size 16 \
#   --repeat_to_3ch \
#   --out_csv /workspace/predict/preds_I5_R5.csv

# # 回测
# python portfolio_backtest.py \
#   --pred_csv /workspace/predict/preds_I20_R5.csv \
#   --csv_root /workspace/stock_data_csv \
#   --horizon_days 5 \
#   --weighting equal \
#   --out_dir /workspace/backtest_I20_R5_equal

# python portfolio_backtest.py \
#   --pred_csv /workspace/predict/preds_I5_R5.csv \
#   --csv_root /workspace/stock_data_csv \
#   --horizon_days 5 \
#   --weighting equal \
#   --out_dir /workspace/backtest_I5_R5_equal

#!/usr/bin/env bash
set -euo pipefail

# =========================
# 路径配置
# =========================
PYTHON_BIN="python"

SCRIPT_DIR="/workspace"

LABEL_DIR="/workspace/label_npz"
SYMBOL_TAR_DIR="/workspace/symbol_to_tar"
PRED_DIR="/workspace/predict"

CSV_ROOT="/workspace/stock_data_csv"

EXP_I20_R5="/workspace/exp_I20_R5_vitb_in21k"
EXP_I5_R5="/workspace/exp_I5_R5_vitb_in21k"

BACKTEST_I20_R5_EQUAL="/workspace/backtest_I20_R5_equal"
BACKTEST_I20_R5_VALUE="/workspace/backtest_I20_R5_value"
BACKTEST_I5_R5_EQUAL="/workspace/backtest_I5_R5_equal"
BACKTEST_I5_R5_VALUE="/workspace/backtest_I5_R5_value"

# =========================
# 创建目录
# =========================
# mkdir -p "${SYMBOL_TAR_DIR}"
# mkdir -p "${PRED_DIR}"

# 如果不想混入旧训练日志，先删旧实验目录
# rm -rf "${EXP_I20_R5}"
# rm -rf "${EXP_I5_R5}"

# 如果不想混入旧回测结果，也可以删掉
# rm -rf "${BACKTEST_I20_R5_EQUAL}"
# rm -rf "${BACKTEST_I20_R5_VALUE}"
# rm -rf "${BACKTEST_I5_R5_EQUAL}"
# rm -rf "${BACKTEST_I5_R5_VALUE}"

# =========================
# 1) 建立 symbol -> tar 映射
# =========================
echo "===== build symbol_to_tar for N20 ====="
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_symbol_to_tar.py" \
    --tar_dir /workspace/ohlc_image_N20/packs \
    --out_json "${SYMBOL_TAR_DIR}/symbol_to_tar_N20.json"

echo "===== build symbol_to_tar for N5 ====="
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_symbol_to_tar.py" \
    --tar_dir /workspace/ohlc_image_N5/packs \
    --out_json "${SYMBOL_TAR_DIR}/symbol_to_tar_N5.json"

# =========================
# 2) 清洗 meta，输出 clean npz
# =========================
echo "===== clean meta N20 ====="
"${PYTHON_BIN}" "${SCRIPT_DIR}/clean_meta_by_tar.py" \
    --meta_path "${LABEL_DIR}/meta_N20.npz" \
    --symbol_to_tar_json "${SYMBOL_TAR_DIR}/symbol_to_tar_N20.json" \
    --out_meta_path "${LABEL_DIR}/meta_N20_clean.npz"

echo "===== clean meta N5 ====="
"${PYTHON_BIN}" "${SCRIPT_DIR}/clean_meta_by_tar.py" \
    --meta_path "${LABEL_DIR}/meta_N5.npz" \
    --symbol_to_tar_json "${SYMBOL_TAR_DIR}/symbol_to_tar_N5.json" \
    --out_meta_path "${LABEL_DIR}/meta_N5_clean.npz"

# =========================
# 3) 训练 I20 / R5
# =========================
echo "===== train I20 / R5 ====="
"${PYTHON_BIN}" "${SCRIPT_DIR}/train.py" \
  --meta_path "${LABEL_DIR}/meta_N20_clean.npz" \
  --symbol_to_tar_json "${SYMBOL_TAR_DIR}/symbol_to_tar_N20.json" \
  --horizon_idx 0 \
  --hf_model_name google/vit-base-patch16-224-in21k \
  --epochs 5 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --num_workers 12 \
  --seed 20260315 \
  --patch_size 16 \
  --repeat_to_3ch \
  --warmup_ratio 0.05 \
  --step_log_interval 200 \
  --step_log_file metrics_step.csv \
  --out_dir "${EXP_I20_R5}"

# =========================
# 4) 训练 I5 / R5
# =========================
echo "===== train I5 / R5 ====="
"${PYTHON_BIN}" "${SCRIPT_DIR}/train.py" \
  --meta_path "${LABEL_DIR}/meta_N5_clean.npz" \
  --symbol_to_tar_json "${SYMBOL_TAR_DIR}/symbol_to_tar_N5.json" \
  --horizon_idx 0 \
  --hf_model_name google/vit-base-patch16-224-in21k \
  --epochs 5 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --num_workers 12 \
  --seed 20260315 \
  --patch_size 16 \
  --repeat_to_3ch \
  --warmup_ratio 0.05 \
  --step_log_interval 200 \
  --step_log_file metrics_step.csv \
  --out_dir "${EXP_I5_R5}"

# =========================
# 5) 推断 I20 / R5 的 test 预测
# =========================
echo "===== infer test I20 / R5 ====="
"${PYTHON_BIN}" "${SCRIPT_DIR}/infer_test.py" \
  --meta_path "${LABEL_DIR}/meta_N20_clean.npz" \
  --symbol_to_tar_json "${SYMBOL_TAR_DIR}/symbol_to_tar_N20.json" \
  --checkpoint "${EXP_I20_R5}/best.pt" \
  --horizon_idx 0 \
  --batch_size 128 \
  --num_workers 12 \
  --patch_size 16 \
  --repeat_to_3ch \
  --out_csv "${PRED_DIR}/preds_I20_R5.csv"

# =========================
# 6) 推断 I5 / R5 的 test 预测
# =========================
echo "===== infer test I5 / R5 ====="
"${PYTHON_BIN}" "${SCRIPT_DIR}/infer_test.py" \
  --meta_path "${LABEL_DIR}/meta_N5_clean.npz" \
  --symbol_to_tar_json "${SYMBOL_TAR_DIR}/symbol_to_tar_N5.json" \
  --checkpoint "${EXP_I5_R5}/best.pt" \
  --horizon_idx 0 \
  --batch_size 128 \
  --num_workers 12 \
  --patch_size 16 \
  --repeat_to_3ch \
  --out_csv "${PRED_DIR}/preds_I5_R5.csv"

# =========================
# 7) 回测 I20 / R5 - equal-weight
# =========================
echo "===== backtest I20 / R5 - equal-weight ====="
"${PYTHON_BIN}" "${SCRIPT_DIR}/portfolio_backtest.py" \
  --pred_csv "${PRED_DIR}/preds_I20_R5.csv" \
  --csv_root "${CSV_ROOT}" \
  --horizon_days 5 \
  --weighting equal \
  --out_dir "${BACKTEST_I20_R5_EQUAL}"

# =========================
# 8) 回测 I20 / R5 - value-weight
# =========================
# echo "===== backtest I20 / R5 - value-weight ====="
# "${PYTHON_BIN}" "${SCRIPT_DIR}/portfolio_backtest.py" \
#   --pred_csv "${PRED_DIR}/preds_I20_R5.csv" \
#   --csv_root "${CSV_ROOT}" \
#   --horizon_days 5 \
#   --weighting value \
#   --cap_col 日个股流通市值 \
#   --out_dir "${BACKTEST_I20_R5_VALUE}"

# =========================
# 9) 回测 I5 / R5 - equal-weight
# =========================
echo "===== backtest I5 / R5 - equal-weight ====="
"${PYTHON_BIN}" "${SCRIPT_DIR}/portfolio_backtest.py" \
  --pred_csv "${PRED_DIR}/preds_I5_R5.csv" \
  --csv_root "${CSV_ROOT}" \
  --horizon_days 5 \
  --weighting equal \
  --out_dir "${BACKTEST_I5_R5_EQUAL}"

# =========================
# 10) 回测 I5 / R5 - value-weight
# =========================
# echo "===== backtest I5 / R5 - value-weight ====="
# "${PYTHON_BIN}" "${SCRIPT_DIR}/portfolio_backtest.py" \
#   --pred_csv "${PRED_DIR}/preds_I5_R5.csv" \
#   --csv_root "${CSV_ROOT}" \
#   --horizon_days 5 \
#   --weighting value \
#   --cap_col 日个股流通市值 \
#   --out_dir "${BACKTEST_I5_R5_VALUE}"

echo "===== all done ====="