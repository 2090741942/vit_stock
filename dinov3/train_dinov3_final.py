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
from torch.utils.data import DataLoader

from transformers import DINOv3ViTModel, get_cosine_schedule_with_warmup

from vit_stock.data.dataset import build_dataloader, get_target_width


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
    valid_label_rows = int((split_mask & (y != -1)).sum())
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


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_dinov3(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


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

        for p in self.backbone.parameters():
            p.requires_grad = True

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
        return self.classifier(feat)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
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

        images = normalize_for_dinov3(images)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

        running_loss = total_loss / max(total_samples, 1)
        running_acc = total_correct / max(total_samples, 1)
        pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
        "n": total_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DINOv3 binary classifier on OHLC images.")

    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--symbol_to_tar_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--hf_model_name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--horizon_idx", type=int, required=True, choices=[0, 1, 2])

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260315)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pool_mode", type=str, default="cls_mean_patch", choices=["cls", "cls_mean_patch"])
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument("--step_log_interval", type=int, default=5000)
    parser.add_argument("--validate_members", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    window = infer_window_from_meta(args.meta_path)
    horizon_days = horizon_idx_to_days(args.horizon_idx)

    target_width = get_target_width(window, patch_size=16)
    if 96 % 16 != 0:
        raise ValueError("当前实现假定高度 96 能被 patch_size=16 整除")

    print("=" * 80)
    print(f"device          : {device}")
    print(f"meta_path       : {args.meta_path}")
    print(f"window          : {window}")
    print(f"horizon_idx     : {args.horizon_idx}")
    print(f"horizon_days    : {horizon_days}")
    print(f"target_width    : {target_width}")
    print(f"hf_model_name   : {args.hf_model_name}")
    print(f"pool_mode       : {args.pool_mode}")
    print("=" * 80)

    train_stats = count_split_stats(args.meta_path, split="train", horizon_idx=args.horizon_idx)
    val_stats = count_split_stats(args.meta_path, split="val", horizon_idx=args.horizon_idx)
    test_stats = count_split_stats(args.meta_path, split="test", horizon_idx=args.horizon_idx)
    print("split stats:")
    print(json.dumps({"train": train_stats, "val": val_stats, "test": test_stats}, indent=2, ensure_ascii=False))

    train_loader, train_dataset = build_dataloader(
        meta_path=args.meta_path,
        split="train",
        horizon_idx=args.horizon_idx,
        symbol_to_tar_json=args.symbol_to_tar_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        return_meta=False,
        validate_members=args.validate_members,
    )

    val_loader, val_dataset = build_dataloader(
        meta_path=args.meta_path,
        split="val",
        horizon_idx=args.horizon_idx,
        symbol_to_tar_json=args.symbol_to_tar_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        return_meta=False,
        validate_members=args.validate_members,
    )

    print(f"len(train_dataset) = {len(train_dataset)}")
    print(f"len(val_dataset)   = {len(val_dataset)}")
    print(f"len(train_loader)  = {len(train_loader)}")
    print(f"len(val_loader)    = {len(val_loader)}")

    model = HFDinoV3BinaryClassifier(
        model_name=args.hf_model_name,
        num_labels=2,
        dropout=args.dropout,
        pool_mode=args.pool_mode,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_train_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    metrics_csv = os.path.join(args.out_dir, "metrics_step.csv")
    metrics_epoch_csv = os.path.join(args.out_dir, "metrics_epoch.csv")
    best_ckpt = os.path.join(args.out_dir, "best.pt")
    last_ckpt = os.path.join(args.out_dir, "last.pt")
    config_json = os.path.join(args.out_dir, "train_config.json")

    config = vars(args).copy()
    config.update(
        {
            "window": window,
            "horizon_days": horizon_days,
            "target_width": target_width,
            "device": str(device),
        }
    )

    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    best_val_acc = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_samples = 0

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        epoch_start = time.time()

        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            images = normalize_for_dinov3(images)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size = images.size(0)
            epoch_loss_sum += loss.item() * batch_size
            epoch_correct += (logits.argmax(dim=1) == targets).sum().item()
            epoch_samples += batch_size
            global_step += 1

            train_loss = epoch_loss_sum / max(epoch_samples, 1)
            train_acc = epoch_correct / max(epoch_samples, 1)
            lr_now = optimizer.param_groups[0]["lr"]

            pbar.set_postfix(loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}", lr=f"{lr_now:.2e}")

            if args.step_log_interval > 0 and global_step % args.step_log_interval == 0:
                append_metrics_csv(
                    metrics_csv,
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "split": "train",
                        "loss": train_loss,
                        "acc": train_acc,
                        "lr": lr_now,
                    },
                )

        train_epoch_loss = epoch_loss_sum / max(epoch_samples, 1)
        train_epoch_acc = epoch_correct / max(epoch_samples, 1)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            desc=f"Val epoch {epoch}/{args.epochs}",
        )

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        append_metrics_csv(
            metrics_csv,
            {
                "global_step": global_step,
                "epoch": epoch,
                "split": "train_epoch",
                "loss": train_epoch_loss,
                "acc": train_epoch_acc,
                "lr": lr_now,
            },
        )
        append_metrics_csv(
            metrics_csv,
            {
                "global_step": global_step,
                "epoch": epoch,
                "split": "val",
                "loss": val_metrics["loss"],
                "acc": val_metrics["acc"],
                "lr": lr_now,
            },
        )
        append_metrics_csv(
            metrics_epoch_csv,
            {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_epoch_loss,
                "train_acc": train_epoch_acc,
                "train_num_samples": epoch_samples,
                "train_time_sec": elapsed,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_num_samples": val_metrics["n"],
            },
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            save_checkpoint(
                ckpt_path=best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                config=config,
            )
            print(f"[best] epoch={epoch}, val_acc={best_val_acc:.6f}, saved to {best_ckpt}")

        if args.save_last:
            save_checkpoint(
                ckpt_path=last_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                config=config,
            )

        print(
            f"epoch {epoch}/{args.epochs} | "
            f"train_loss={train_epoch_loss:.6f}, train_acc={train_epoch_acc:.6f} | "
            f"val_loss={val_metrics['loss']:.6f}, val_acc={val_metrics['acc']:.6f} | "
            f"time={elapsed:.1f}s"
        )

    print("training finished.")
    print(f"best_val_acc = {best_val_acc:.6f}")
    print(f"best checkpoint = {best_ckpt}")


if __name__ == "__main__":
    main()
