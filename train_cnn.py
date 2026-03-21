
from __future__ import annotations

import os
import csv
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from dataset import build_dataloader, get_target_width


# =========================
# 工具函数
# =========================

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
    split_mask = (splits == split_map[split])

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
    pixel_mean: float,
    pixel_std: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "config": config,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
        },
        ckpt_path,
    )


# =========================
# CNN 标准化
# =========================

@torch.no_grad()
def estimate_train_pixel_stats(
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> Tuple[float, float]:
    """
    估计训练集单通道像素均值和标准差。
    要求 dataset 输出已经是 [0, 1] 范围的 float tensor。
    """
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Estimate pixel stats", leave=False), start=1):
        images = batch[0].to(device, non_blocking=True)  # [B,1,H,W]
        if images.ndim != 4 or images.size(1) != 1:
            raise ValueError(f"Expected images with shape [B,1,H,W], got {tuple(images.shape)}")

        total_sum += images.sum().item()
        total_sq_sum += (images * images).sum().item()
        total_count += images.numel()

        if max_batches is not None and batch_idx >= max_batches:
            break

    if total_count == 0:
        raise RuntimeError("No pixels were seen when estimating pixel stats.")

    mean = total_sum / total_count
    var = max(total_sq_sum / total_count - mean * mean, 1e-12)
    std = var ** 0.5
    return float(mean), float(std)


def normalize_for_cnn(images: torch.Tensor, pixel_mean: float, pixel_std: float) -> torch.Tensor:
    mean = torch.tensor([pixel_mean], device=images.device, dtype=images.dtype).view(1, 1, 1, 1)
    std = torch.tensor([pixel_std], device=images.device, dtype=images.dtype).view(1, 1, 1, 1)
    return (images - mean) / std


# =========================
# 模型
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
    """
    单通道灰度图 CNN。
    I5 / I20 / I60 分别使用 2 / 3 / 4 个 blocks。
    """

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

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# 训练与评估
# =========================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pixel_mean: float,
    pixel_std: float,
    desc: str = "Eval",
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(
        loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        images = normalize_for_cnn(images, pixel_mean=pixel_mean, pixel_std=pixel_std)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

        running_loss = total_loss / max(total_samples, 1)
        running_acc = total_correct / max(total_samples, 1)

        pbar.set_postfix(
            loss=f"{running_loss:.4f}",
            acc=f"{running_acc:.4f}",
            bs=batch_size,
        )

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)

    return {
        "loss": avg_loss,
        "acc": avg_acc,
        "num_samples": total_samples,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    pixel_mean: float,
    pixel_std: float,
    epoch: int,
    total_epochs: int,
    step_log_interval: int,
    step_csv_path: str,
    global_step_start: int,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    start_time = time.time()

    pbar = tqdm(
        loader,
        desc=f"Train {epoch}/{total_epochs}",
        leave=True,
        dynamic_ncols=True,
    )

    for step, (images, targets) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        images = normalize_for_cnn(images, pixel_mean=pixel_mean, pixel_std=pixel_std)

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
            step_row = {
                "epoch": epoch,
                "step_in_epoch": step,
                "global_step": global_step,
                "lr": current_lr,
                "running_train_loss": running_loss,
                "running_train_acc": running_acc,
                "batch_size": batch_size,
            }
            append_metrics_csv(step_csv_path, step_row)

        pbar.set_postfix(
            loss=f"{running_loss:.4f}",
            acc=f"{running_acc:.4f}",
            lr=f"{current_lr:.2e}",
            bs=batch_size,
        )

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)

    return {
        "loss": avg_loss,
        "acc": avg_acc,
        "num_samples": total_samples,
        "time_sec": elapsed,
        "lr": optimizer.param_groups[0]["lr"],
        "num_steps": len(loader),
    }


# =========================
# 主函数
# =========================

def main():
    parser = argparse.ArgumentParser(description="Train CNN on OHLC images with ViT-like training flow.")

    # 数据
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--symbol_to_tar_json", type=str, required=True)
    parser.add_argument("--horizon_idx", type=int, required=True, choices=[0, 1, 2])

    # 训练参数
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--step_log_interval", type=int, default=100)
    parser.add_argument("--step_log_file", type=str, default="metrics_step.csv")

    # 输入设置
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--validate_members", action="store_true", help="是否再次检查 tar 成员。正式训练一般不开。")
    parser.add_argument("--disable_paper_first_layer", action="store_true", help="关闭第一层特殊 stride/dilation")

    # 模型设置
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fc_hidden_dim", type=int, default=0, help="0 表示直接线性分类头")
    parser.add_argument("--negative_slope", type=float, default=0.01)

    # 统计方式
    parser.add_argument(
        "--stats_mode",
        type=str,
        default="full",
        choices=["full", "approx", "manual"],
        help="CNN 单通道像素统计的获取方式",
    )
    parser.add_argument("--stats_batches", type=int, default=50, help="stats_mode=approx 时使用前多少个 batch")
    parser.add_argument("--pixel_mean", type=float, default=None, help="stats_mode=manual 时传入")
    parser.add_argument("--pixel_std", type=float, default=None, help="stats_mode=manual 时传入")

    # 输出
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

    exp_name = f"I{window}_R{horizon_days}_cnn"
    print(f"experiment = {exp_name}")
    print(f"window = {window}")
    print(f"horizon_days = {horizon_days}")
    print(f"target input shape = [1, 96, {target_width}]")
    print("input normalization = train-set grayscale mean/std")
    print("repeat_to_3ch = False")

    config = vars(args).copy()
    config["window"] = window
    config["horizon_days"] = horizon_days
    config["target_width"] = target_width
    config["device"] = str(device)
    config["normalization"] = "train_grayscale_mean_std"

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
        repeat_to_3ch=False,
        normalize_to_01=True,
        return_meta=False,
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
        repeat_to_3ch=False,
        normalize_to_01=True,
        return_meta=False,
        validate_members=args.validate_members,
        validate_verbose=args.validate_members,
    )

    print("\n===== dataset size after loader filtering =====")
    print(f"train dataset size = {len(train_set)}")
    print(f"val dataset size   = {len(val_set)}")

    if args.stats_mode == "manual":
        if args.pixel_mean is None or args.pixel_std is None:
            raise ValueError("stats_mode=manual 时，必须同时提供 --pixel_mean 和 --pixel_std")
        pixel_mean = float(args.pixel_mean)
        pixel_std = float(args.pixel_std)
    else:
        max_batches = None if args.stats_mode == "full" else int(args.stats_batches)
        pixel_mean, pixel_std = estimate_train_pixel_stats(
            loader=train_loader,
            device=device,
            max_batches=max_batches,
        )

    print("\n===== pixel stats =====")
    print(f"pixel_mean = {pixel_mean:.8f}")
    print(f"pixel_std  = {pixel_std:.8f}")

    model = OHLCBinaryCNN(
        window=window,
        input_height=96,
        input_width=target_width,
        num_classes=2,
        dropout=args.dropout,
        fc_hidden_dim=args.fc_hidden_dim,
        negative_slope=args.negative_slope,
        use_paper_first_layer=(not args.disable_paper_first_layer),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n===== model stats =====")
    print(f"model_name = {model.model_name}")
    print(f"num_blocks = {model.num_blocks}")
    print(f"total params = {num_params:,}")
    print(f"trainable params = {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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

    print("\n===== start training =====")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            epoch=epoch,
            total_epochs=args.epochs,
            step_log_interval=args.step_log_interval,
            step_csv_path=metrics_step_csv,
            global_step_start=global_step,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            desc=f"Val {epoch}/{args.epochs}",
        )

        row = {
            "epoch": epoch,
            "lr": train_metrics["lr"],
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_num_samples": train_metrics["num_samples"],
            "train_time_sec": train_metrics["time_sec"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_num_samples": val_metrics["num_samples"],
        }
        append_metrics_csv(metrics_epoch_csv, row)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = float(val_metrics["acc"])
            best_epoch = int(epoch)
            save_checkpoint(
                ckpt_path=best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                config=config,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
            )
            print(f"saved new best checkpoint to: {best_ckpt}")

        save_checkpoint(
            ckpt_path=last_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc,
            config=config,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )

        global_step += train_metrics["num_steps"]

    print("\n===== training finished =====")
    print(f"best_epoch = {best_epoch}")
    print(f"best_val_acc = {best_val_acc:.6f}")

    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
        "train_dataset_size": len(train_set),
        "val_dataset_size": len(val_set),
        "train_split_stats": train_stats,
        "val_split_stats": val_stats,
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    train_set.close()
    val_set.close()


if __name__ == "__main__":
    main()


"""
示例：

python train_cnn.py \
  --meta_path /workspace/label_npz/meta_N60_clean.npz \
  --symbol_to_tar_json /workspace/symbol_to_tar/symbol_to_tar_N60.json \
  --horizon_idx 0 \
  --epochs 30 \
  --batch_size 128 \
  --lr 1e-5 \
  --num_workers 8 \
  --out_dir /workspace/outs/I60_R5_cnn
"""
