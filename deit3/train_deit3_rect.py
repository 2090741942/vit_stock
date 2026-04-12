from __future__ import annotations

import os
import csv
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import timm

from transformers import get_cosine_schedule_with_warmup

from vit_stock.data.dataset import build_dataloader, get_target_width


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def infer_window_from_meta(meta_path: str) -> int:
    meta = np.load(meta_path, allow_pickle=True)
    return int(meta["window"][0])


def horizon_idx_to_days(horizon_idx: int) -> int:
    mapping = {0: 5, 1: 20, 2: 60}
    return mapping[horizon_idx]


def count_split_stats(meta_path: str, split: str, horizon_idx: int) -> Dict[str, int]:
    split_map = {"train": 0, "val": 1, "test": 2}
    if split not in split_map:
        raise ValueError(f"invalid split: {split}")
    if horizon_idx not in (0, 1, 2):
        raise ValueError(f"invalid horizon_idx: {horizon_idx}")

    meta = np.load(meta_path, allow_pickle=True)
    labels = meta["labels"]
    splits = meta["split"]

    y = labels[:, horizon_idx]
    split_mask = splits == split_map[split]

    raw_split_rows = int(split_mask.sum())
    valid_label_rows = int(((split_mask) & (y != -1)).sum())
    dropped_label_minus1 = raw_split_rows - valid_label_rows

    return {
        "raw_split_rows": raw_split_rows,
        "valid_label_rows": valid_label_rows,
        "dropped_label_minus1": dropped_label_minus1,
    }


def append_metrics_csv(csv_path: str, row: Dict) -> None:
    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    config: Dict,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "config": config,
        },
        ckpt_path,
    )


def normalize_for_deit(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


class TIMMDeiT3BinaryClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "deit3_base_patch16_224.fb_in22k_ft_in1k",
        num_labels: int = 2,
        img_size: tuple[int, int] = (96, 192),
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_labels,
            img_size=img_size,
            in_chans=3,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Eval",
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        images = normalize_for_deit(images)
        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

        running_loss = total_loss / max(total_samples, 1)
        running_acc = total_correct / max(total_samples, 1)
        pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}", bs=batch_size)

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
        "num_samples": total_samples,
        "num_steps": len(loader),
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    step_log_interval: int,
    step_csv_path: str,
    global_step_start: int,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", dynamic_ncols=True)
    for step, (images, targets) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        images = normalize_for_deit(images)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

        running_loss = total_loss / max(total_samples, 1)
        running_acc = total_correct / max(total_samples, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        global_step = global_step_start + step

        if step_log_interval > 0 and (step % step_log_interval == 0):
            append_metrics_csv(
                step_csv_path,
                {
                    "epoch": epoch,
                    "step_in_epoch": step,
                    "global_step": global_step,
                    "lr": current_lr,
                    "running_train_loss": running_loss,
                    "running_train_acc": running_acc,
                    "batch_size": batch_size,
                },
            )

        pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}", lr=f"{current_lr:.2e}", bs=batch_size)

    elapsed = time.time() - start_time
    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
        "num_samples": total_samples,
        "time_sec": elapsed,
        "lr": optimizer.param_groups[0]["lr"],
        "num_steps": len(loader),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeiT-III on OHLC images (rectangular input version).")

    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--symbol_to_tar_json", type=str, required=True)
    parser.add_argument("--horizon_idx", type=int, required=True, choices=[0, 1, 2])

    parser.add_argument("--hf_model_name", type=str, default="deit3_base_patch16_224.fb_in22k_ft_in1k")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260315)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--step_log_interval", type=int, default=100)
    parser.add_argument("--step_log_file", type=str, default="metrics_step.csv")

    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--repeat_to_3ch", action="store_true")
    parser.add_argument("--validate_members", action="store_true")
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--drop_path_rate", type=float, default=0.0)
    parser.add_argument("--save_last", action="store_true")

    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if torch.cuda.is_available():
        print(f"gpu = {torch.cuda.get_device_name(0)}")

    window = infer_window_from_meta(args.meta_path)
    horizon_days = horizon_idx_to_days(args.horizon_idx)
    target_width = get_target_width(window, patch_size=args.patch_size)
    target_height = 96

    exp_name = f"I{window}_R{horizon_days}_deit3"
    print(f"experiment = {exp_name}")
    print(f"window = {window}")
    print(f"horizon_days = {horizon_days}")
    print(f"target input shape = [3, {target_height}, {target_width}]")
    print("finetune mode = full finetuning")
    print("input normalization = ImageNet mean/std")

    config = vars(args).copy()
    config["window"] = window
    config["horizon_days"] = horizon_days
    config["target_width"] = target_width
    config["target_height"] = target_height
    config["device"] = str(device)
    config["normalization"] = "imagenet_mean_std"
    config["runtime_square_pad"] = False

    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    train_stats = count_split_stats(args.meta_path, "train", args.horizon_idx)
    val_stats = count_split_stats(args.meta_path, "val", args.horizon_idx)

    print("\n===== split stats before loader =====")
    for split_name, stats in [("train", train_stats), ("val", val_stats)]:
        print(
            f"{split_name}: "
            f"raw_split_rows={stats['raw_split_rows']}, "
            f"valid_label_rows={stats['valid_label_rows']}, "
            f"dropped_label_minus1={stats['dropped_label_minus1']}"
        )

    train_loader, train_set = build_dataloader(
        meta_path=args.meta_path,
        symbol_to_tar_json=args.symbol_to_tar_json,
        split="train",
        horizon_idx=args.horizon_idx,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        shuffle=True,
        repeat_to_3ch=args.repeat_to_3ch,
        normalize_to_01=True,
        return_meta=False,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,
        validate_members=args.validate_members,
        validate_verbose=args.validate_members,
    )

    val_loader, val_set = build_dataloader(
        meta_path=args.meta_path,
        symbol_to_tar_json=args.symbol_to_tar_json,
        split="val",
        horizon_idx=args.horizon_idx,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        shuffle=False,
        repeat_to_3ch=args.repeat_to_3ch,
        normalize_to_01=True,
        return_meta=False,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,
        validate_members=args.validate_members,
        validate_verbose=args.validate_members,
    )

    print("\n===== dataset size after loader filtering =====")
    print(f"train dataset size = {len(train_set)}")
    print(f"val dataset size   = {len(val_set)}")

    model = TIMMDeiT3BinaryClassifier(
        model_name=args.hf_model_name,
        num_labels=2,
        img_size=(target_height, target_width),
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n===== model stats =====")
    print(f"hf_model_name = {args.hf_model_name}")
    print(f"total params = {num_params:,}")
    print(f"trainable params = {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = len(train_loader)
    num_training_steps = args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    print("\n===== scheduler stats =====")
    print(f"num_update_steps_per_epoch = {num_update_steps_per_epoch}")
    print(f"num_training_steps = {num_training_steps}")
    print(f"num_warmup_steps = {num_warmup_steps}")

    metrics_epoch_csv = os.path.join(args.out_dir, "metrics_epoch.csv")
    metrics_step_csv = os.path.join(args.out_dir, args.step_log_file)
    best_ckpt = os.path.join(args.out_dir, "best.pt")
    last_ckpt = os.path.join(args.out_dir, "last.pt")

    best_val_acc = -1.0
    best_epoch = -1
    global_step = 0

    print("\n===== start training =====")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            step_log_interval=args.step_log_interval,
            step_csv_path=metrics_step_csv,
            global_step_start=global_step,
        )
        global_step += train_metrics["num_steps"]

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            desc=f"Val Epoch {epoch}",
        )

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": train_metrics["lr"],
            "train_num_samples": train_metrics["num_samples"],
            "val_num_samples": val_metrics["num_samples"],
            "train_time_sec": train_metrics["time_sec"],
        }
        append_metrics_csv(metrics_epoch_csv, epoch_row)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['acc']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            save_checkpoint(best_ckpt, model, optimizer, epoch, best_val_acc, config)
            print(f"[BEST] saved to {best_ckpt} (epoch={epoch}, val_acc={best_val_acc:.4f})")

        if args.save_last:
            save_checkpoint(last_ckpt, model, optimizer, epoch, best_val_acc, config)

    print("\n===== training finished =====")
    print(f"best_epoch = {best_epoch}")
    print(f"best_val_acc = {best_val_acc:.6f}")

    train_set.close()
    val_set.close()


if __name__ == "__main__":
    main()
