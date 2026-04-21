文件说明

1. board_utils.py
   公共板块工具，包含：
   - normalize_symbol
   - infer_board_from_symbol

2. build_symbol_to_board.py
   生成 symbol -> board 映射表
   可从 meta_N*_clean.npz 或预测 csv 提取 symbol

3. attach_board_and_split_predictions.py
   给全市场预测结果打板块标签，并切成五个板块子集 csv

4. run_board_backtests.sh
   调你现有的 portfolio_backtest_with_costs_and_turnover.py
   对五个板块分别跑回测

推荐流程

Step 1:
python build_symbol_to_board.py \
  --meta_path /workspace/label_npz/meta_N60_clean.npz \
  --out_csv /workspace/board_eval/symbol_to_board.csv

Step 2:
python attach_board_and_split_predictions.py \
  --pred_csv /workspace/predict/preds_I60_R5.csv \
  --symbol_to_board_csv /workspace/board_eval/symbol_to_board.csv \
  --out_with_board_csv /workspace/board_eval/preds_I60_R5_with_board.csv \
  --split_out_dir /workspace/board_eval/pred_splits

Step 3:
bash run_board_backtests.sh \
  --pred_split_dir /workspace/board_eval/pred_splits \
  --pred_stem preds_I60_R5 \
  --csv_root /workspace/stock_data_csv \
  --horizon_days 5 \
  --weighting equal \
  --backtest_py /workspace/vit_stock/portfolio_backtest_with_costs_and_turnover.py \
  --out_root /workspace/board_eval/backtests_I60_R5_equal
