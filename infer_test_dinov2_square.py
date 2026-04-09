from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import Dinov2Config, Dinov2ForImageClassification

from dataset import build_dataloader, get_target_width


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_dinov2(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def infer_window_from_meta(meta_path: str) -> int:
    meta = np.load(meta_path, allow_pickle=True)
    return int(meta["window"][0])


def ceil_to_multiple(x: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((x + multiple - 1) // multiple) * multiple


def pad_to_square(images: torch.Tensor, target_side: int) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"images must be [B,C,H,W], got {tuple(images.shape)}")

    _, _, h, w = images.shape
    if h > target_side or w > target_side:
        raise ValueError(f"current shape {(h, w)} is larger than target_side={target_side}")

    pad_h = target_side - h
    pad_w = target_side - w
    return F.pad(images, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


class HFDinoV2BinaryClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base-imagenet1k-1-layer",
        num_labels: int = 2,
        image_size: int = 224,
    ) -> None:
        super().__init__()

        config = Dinov2Config.from_pretrained(model_name)
        config.num_labels = num_labels
        config.image_size = int(image_size)

        self.model = Dinov2ForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        self.patch_size = int(config.patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader,
    device: torch.device,
    model_side: int,
) -> pd.DataFrame:
    model.eval()

    rows: List[Dict] = []

    pbar = tqdm(loader, desc="Infer test", dynamic_ncols=True)
    for images, targets, metas in pbar:
        images = images.to(device, non_blocking=True)
        images = pad_to_square(images, target_side=model_side)
        images = normalize_for_dinov2(images)

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
    parser = argparse.ArgumentParser(description="Run DINOv2 test inference and export prediction CSV (square-pad version).")

    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--symbol_to_tar_json", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="best.pt")
    parser.add_argument("--hf_model_name", type=str, default="facebook/dinov2-base-imagenet1k-1-layer")

    parser.add_argument("--horizon_idx", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pad_multiple", type=int, default=14)
    parser.add_argument("--repeat_to_3ch", action="store_true")
    parser.add_argument("--validate_members", action="store_true")

    parser.add_argument("--out_csv", type=str, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if torch.cuda.is_available():
        print(f"gpu = {torch.cuda.get_device_name(0)}")

    window = infer_window_from_meta(args.meta_path)
    dataset_width = get_target_width(window, patch_size=args.pad_multiple)
    dataset_width = ceil_to_multiple(dataset_width, args.pad_multiple)
    dataset_height = 96

    model_side = max(dataset_height, dataset_width)
    model_side = ceil_to_multiple(model_side, args.pad_multiple)

    print(f"window = {window}")
    print(f"dataset input shape = [3, {dataset_height}, {dataset_width}]")
    print(f"model input shape   = [3, {model_side}, {model_side}]")
    print(f"pad_multiple = {args.pad_multiple}")

    loader, dataset = build_dataloader(
        meta_path=args.meta_path,
        symbol_to_tar_json=args.symbol_to_tar_json,
        split="test",
        horizon_idx=args.horizon_idx,
        batch_size=args.batch_size,
        patch_size=args.pad_multiple,
        num_workers=args.num_workers,
        shuffle=False,
        repeat_to_3ch=args.repeat_to_3ch,
        normalize_to_01=True,
        return_meta=True,
        validate_members=args.validate_members,
        validate_verbose=args.validate_members,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,
    )

    print(f"test dataset size = {len(dataset)}")

    model = HFDinoV2BinaryClassifier(
        model_name=args.hf_model_name,
        num_labels=2,
        image_size=model_side,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"loaded checkpoint from: {args.checkpoint}")
    print(f"dinov2_patch_size = {model.patch_size}")

    pred_df = run_inference(
        model=model,
        loader=loader,
        device=device,
        model_side=model_side,
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"saved predictions to: {out_path}")
    print(pred_df.head())

    dataset.close()


if __name__ == "__main__":
    main()
