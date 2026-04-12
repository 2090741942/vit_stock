
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from vit_stock.data.dataset import build_dataloader, get_target_width


# =========================
# 工具函数
# =========================

def infer_window_from_meta(meta_path: str) -> int:
    meta = np.load(meta_path, allow_pickle=True)
    return int(meta["window"][0])


def normalize_for_cnn(images: torch.Tensor, pixel_mean: float, pixel_std: float) -> torch.Tensor:
    mean = torch.tensor([pixel_mean], device=images.device, dtype=images.dtype).view(1, 1, 1, 1)
    std = torch.tensor([pixel_std], device=images.device, dtype=images.dtype).view(1, 1, 1, 1)
    return (images - mean) / std


# =========================
# CNN 模型（与 train_cnn.py 保持一致）
# =========================

class CNNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        dilation: Tuple[int, int],
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()

        pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
        pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=(pad_h, pad_w),
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OHLCBinaryCNN(nn.Module):
    def __init__(
        self,
        window: int,
        input_height: int,
        input_width: int,
        num_classes: int = 2,
        dropout: float = 0.5,
        fc_hidden_dim: int = 0,
        negative_slope: float = 0.01,
        use_paper_first_layer: bool = True,
    ) -> None:
        super().__init__()

        if window not in (5, 20, 60):
            raise ValueError(f"window must be one of [5, 20, 60], got {window}")

        self.window = int(window)
        self.model_name = f"I{window}_cnn"

        if window == 5:
            channels = [64, 128]
        elif window == 20:
            channels = [64, 128, 256]
        else:
            channels = [64, 128, 256, 512]

        self.num_blocks = len(channels)

        if use_paper_first_layer:
            first_stride_h = {5: 1, 20: 3, 60: 3}[window]
            first_dilation_h = {5: 1, 20: 2, 60: 3}[window]
            first_kernel = (5, 3)
            first_stride = (first_stride_h, 1)
            first_dilation = (first_dilation_h, 1)
        else:
            first_kernel = (3, 3)
            first_stride = (1, 1)
            first_dilation = (1, 1)

        blocks = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            if i == 0:
                kernel_size = first_kernel
                stride = first_stride
                dilation = first_dilation
            else:
                kernel_size = (3, 3)
                stride = (1, 1)
                dilation = (1, 1)

            blocks.append(
                CNNBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    negative_slope=negative_slope,
                )
            )
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, input_width)
            feat = self.features(dummy)
            flat_dim = int(feat.flatten(1).shape[1])

        if fc_hidden_dim and fc_hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout),
                nn.Linear(flat_dim, fc_hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fc_hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout),
                nn.Linear(flat_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# 推理
# =========================

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader,
    device: torch.device,
    pixel_mean: float,
    pixel_std: float,
) -> pd.DataFrame:
    model.eval()

    rows: List[Dict] = []

    pbar = tqdm(loader, desc="Infer test", dynamic_ncols=True)
    for images, targets, metas in pbar:
        images = images.to(device, non_blocking=True)
        images = normalize_for_cnn(images, pixel_mean=pixel_mean, pixel_std=pixel_std)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        prob_down = probs[:, 0].detach().cpu().numpy()
        prob_up = probs[:, 1].detach().cpu().numpy()
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        target = targets.detach().cpu().numpy()

        symbols = [str(x) for x in metas["symbol"]]
        end_dates = [str(x) for x in metas["end_date"]]

        for s, d, t, p, pdn, pup in zip(symbols, end_dates, target, pred, prob_down, prob_up):
            rows.append(
                {
                    "symbol": s,
                    "end_date": d,
                    "target": int(t),
                    "pred": int(p),
                    "prob_down": float(pdn),
                    "prob_up": float(pup),
                }
            )

    df = pd.DataFrame(rows)
    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.sort_values(["end_date", "symbol"]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run CNN test inference and export prediction CSV.")

    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--symbol_to_tar_json", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="best.pt from CNN training")

    parser.add_argument("--horizon_idx", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--validate_members", action="store_true")

    parser.add_argument("--out_csv", type=str, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if torch.cuda.is_available():
        print(f"gpu = {torch.cuda.get_device_name(0)}")

    window = infer_window_from_meta(args.meta_path)
    target_width = get_target_width(window, patch_size=args.patch_size)
    print(f"window = {window}")
    print(f"target input shape = [1, 96, {target_width}]")

    loader, dataset = build_dataloader(
        meta_path=args.meta_path,
        symbol_to_tar_json=args.symbol_to_tar_json,
        split="test",
        horizon_idx=args.horizon_idx,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        shuffle=False,
        repeat_to_3ch=False,
        normalize_to_01=True,
        return_meta=True,
        validate_members=args.validate_members,
        validate_verbose=args.validate_members,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,
    )

    print(f"test dataset size = {len(dataset)}")

    ckpt = torch.load(args.checkpoint, map_location=device)

    config = ckpt.get("config", {})
    pixel_mean = float(ckpt["pixel_mean"])
    pixel_std = float(ckpt["pixel_std"])

    dropout = float(config.get("dropout", 0.5))
    fc_hidden_dim = int(config.get("fc_hidden_dim", 0))
    negative_slope = float(config.get("negative_slope", 0.01))
    use_paper_first_layer = not bool(config.get("disable_paper_first_layer", False))

    model = OHLCBinaryCNN(
        window=window,
        input_height=96,
        input_width=target_width,
        num_classes=2,
        dropout=dropout,
        fc_hidden_dim=fc_hidden_dim,
        negative_slope=negative_slope,
        use_paper_first_layer=use_paper_first_layer,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"loaded checkpoint from: {args.checkpoint}")
    print(f"model_name = {model.model_name}")
    print(f"num_blocks = {model.num_blocks}")
    print(f"pixel_mean = {pixel_mean:.8f}")
    print(f"pixel_std  = {pixel_std:.8f}")

    pred_df = run_inference(
        model=model,
        loader=loader,
        device=device,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"saved predictions to: {out_path}")
    print(pred_df.head())

    dataset.close()


if __name__ == "__main__":
    main()


"""
示例：

python infer_test_cnn.py \
  --meta_path /workspace/label_npz/meta_N60_clean.npz \
  --symbol_to_tar_json /workspace/symbol_to_tar/symbol_to_tar_N60.json \
  --checkpoint /workspace/outs/I60_R5_cnn/best.pt \
  --horizon_idx 0 \
  --batch_size 128 \
  --num_workers 8 \
  --patch_size 16 \
  --out_csv /workspace/predict/preds_I60_R5_cnn.csv
"""
