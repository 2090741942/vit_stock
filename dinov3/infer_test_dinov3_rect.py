from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import DINOv3ViTModel

from vit_stock.data.dataset import build_dataloader


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_dinov3(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def infer_window_from_meta(meta_path: str) -> int:
    meta = np.load(meta_path, allow_pickle=True)
    return int(meta["window"][0])


class HFDinoV3BinaryClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        num_labels: int = 2,
        dropout: float = 0.0,
        pool_mode: str = "cls_mean_patch",
    ) -> None:
        super().__init__()

        self.backbone = DINOv3ViTModel.from_pretrained(model_name)
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        self.patch_size = int(self.backbone.config.patch_size)
        self.pool_mode = str(pool_mode)

        if self.pool_mode == "cls":
            head_in_dim = self.hidden_size
        elif self.pool_mode == "cls_mean_patch":
            head_in_dim = self.hidden_size * 2
        else:
            raise ValueError(f"unsupported pool_mode: {self.pool_mode}")

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_in_dim, num_labels)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        x = outputs.last_hidden_state

        cls_token = x[:, 0, :]
        if self.pool_mode == "cls":
            feat = cls_token
        else:
            patch_tokens = x[:, 1 + self.num_register_tokens :, :]
            patch_mean = patch_tokens.mean(dim=1)
            feat = torch.cat([cls_token, patch_mean], dim=1)

        feat = self.dropout(feat)
        logits = self.classifier(feat)
        return logits


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()

    rows: List[Dict] = []

    pbar = tqdm(loader, desc="Infer test", dynamic_ncols=True)
    for images, targets, metas in pbar:
        images = images.to(device, non_blocking=True)
        images = normalize_for_dinov3(images)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        prob_down = probs[:, 0].detach().cpu().numpy()
        prob_up = probs[:, 1].detach().cpu().numpy()
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        target = targets.detach().cpu().numpy()

        symbols = [str(x).split(".")[0].zfill(6) for x in metas["symbol"]]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run test inference for DINOv3 and export prediction CSV.")

    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--symbol_to_tar_json", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="best.pt")
    parser.add_argument("--hf_model_name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")

    parser.add_argument("--horizon_idx", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pool_mode", type=str, default="cls_mean_patch", choices=["cls", "cls_mean_patch"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--out_csv", type=str, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = infer_window_from_meta(args.meta_path)

    print("=" * 80)
    print(f"device          : {device}")
    print(f"meta_path       : {args.meta_path}")
    print(f"window          : {window}")
    print(f"horizon_idx     : {args.horizon_idx}")
    print(f"hf_model_name   : {args.hf_model_name}")
    print(f"pool_mode       : {args.pool_mode}")
    print(f"checkpoint      : {args.checkpoint}")
    print(f"out_csv         : {args.out_csv}")
    print("=" * 80)

    loader = build_dataloader(
        meta_path=args.meta_path,
        split="test",
        horizon_idx=args.horizon_idx,
        symbol_to_tar_json=args.symbol_to_tar_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        return_meta=True,
    )

    model = HFDinoV3BinaryClassifier(
        model_name=args.hf_model_name,
        num_labels=2,
        dropout=args.dropout,
        pool_mode=args.pool_mode,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)

    df = run_inference(model=model, loader=loader, device=device)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"saved prediction csv to: {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
