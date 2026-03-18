from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="/workspace/train_result",          # 你要上传的本地文件夹
    repo_id="2090741942justin/vit_stock",  # 替换为你的用户名和repo名
    repo_type="model"                          # "model" / "dataset" / "space"
)
