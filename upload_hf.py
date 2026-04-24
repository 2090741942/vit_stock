from huggingface_hub import HfApi

api = HfApi()

# api.upload_large_folder(
#     folder_path="/workspace/vit_stock_data",          # 你要上传的本地文件夹
#     repo_id="2090741942justin/vit_stock_data",  # 替换为你的用户名和repo名
#     # repo_path="train_result",                      # 你想在repo中存储文件夹的路径
#     repo_type="dataset",                          # "model" / "dataset" / "space"
#     ignore_patterns=[
#         ".cache",
#         ".cache/*",
#         "**/.cache",
#         "**/.cache/*",
#         "__pycache__",
#         "__pycache__/*",
#         "**/__pycache__",
#         "**/__pycache__/*",
#         "*.pyc",
#         "*.tmp",
#         "*.log",
#         "exp_dinov3/**",
#     ],
# )

api.upload_folder(
    folder_path="/workspace/vit_stock_data/exp_dinov3",
    repo_id="2090741942justin/vit_stock_model",
    repo_type="model",
    path_in_repo="exp_dinov3",
)
