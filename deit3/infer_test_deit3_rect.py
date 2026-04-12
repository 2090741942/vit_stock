from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from tqdm.auto import tqdm

from vit_stock.data.dataset import build_dataloader, get_target_width


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_deit(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def infer_window_from_meta(meta_path: str) -> int:
    meta = np.load(meta_path, allow_pickle=True)
    return int(meta["window"][0])


class TIMMDeiT3BinaryClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "deit3_base_patch16_224.fb_in22k_ft_in1k",
        num_labels: int = 2,
        img_size: tuple[int, int] = (96, 192),
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_labels,
            img_size=img_size,
            in_chans=3,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)


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
        images = normalize_for_deit(images)

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
    parser = argparse.ArgumentParser(description="Run DeiT-III test inference and export prediction CSV (rectangular input version).")

    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--symbol_to_tar_json", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="best.pt")
    parser.add_argument("--hf_model_name", type=str, default="deit3_base_patch16_224.fb_in22k_ft_in1k")

    parser.add_argument("--horizon_idx", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--repeat_to_3ch", action="store_true")
    parser.add_argument("--validate_members", action="store_true")

    parser.add_argument("--out_csv", type=str, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if torch.cuda.is_available():
        print(f"gpu = {torch.cuda.get_device_name(0)}")

    window = infer_window_from_meta(args.meta_path)
    target_width = get_target_width(window, patch_size=args.patch_size)
    target_height = 96

    print(f"window = {window}")
    print(f"target input shape = [3, {target_height}, {target_width}]")

    loader, dataset = build_dataloader(
        meta_path=args.meta_path,
        symbol_to_tar_json=args.symbol_to_tar_json,
        split="test",
        horizon_idx=args.horizon_idx,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
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

    model = TIMMDeiT3BinaryClassifier(
        model_name=args.hf_model_name,
        num_labels=2,
        img_size=(target_height, target_width),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"loaded checkpoint from: {args.checkpoint}")

    pred_df = run_inference(model=model, loader=loader, device=device)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"saved predictions to: {out_path}")
    print(pred_df.head())

    dataset.close()


if __name__ == "__main__":
    main()
