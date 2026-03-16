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

from transformers import ViTForImageClassification
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


# =========================
# ImageNet 标准化
# =========================

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_vit(images: torch.Tensor) -> torch.Tensor:
    """
    images: [B, 3, H, W], 取值范围应为 [0, 1]
    """
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


# =========================
# 模型
# =========================

class HFViTBinaryClassifier(nn.Module):
    """
    基于 Hugging Face Google ViT 权重的二分类模型。

    关键点：
    1. 权重导入位置就在 from_pretrained(...)
    2. 默认全量微调，不冻结 backbone
    3. 前向时使用 interpolate_pos_encoding=True，
       以适配你的非 224x224 输入
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        num_labels: int = 2,
    ) -> None:
        super().__init__()

        # 这里就是导入 HF 上 Google ViT 预训练权重的核心代码
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,   # 允许把原分类头改成2类
        )

        # 默认全量微调：这里不冻结任何层
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
        )
        return outputs.logits


# =========================
# 训练与评估
# =========================

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

    pbar = tqdm(
        loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        images = normalize_for_vit(images)

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

# @torch.no_grad()
# def evaluate(
#     model: nn.Module,
#     loader: DataLoader,
#     criterion: nn.Module,
#     device: torch.device,
# ) -> Dict[str, float]:
#     model.eval()

#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0

#     for images, targets in loader:
#         images = images.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         images = normalize_for_vit(images)

#         logits = model(images)
#         loss = criterion(logits, targets)

#         batch_size = images.size(0)
#         total_loss += loss.item() * batch_size
#         total_correct += (logits.argmax(dim=1) == targets).sum().item()
#         total_samples += batch_size

#     avg_loss = total_loss / max(total_samples, 1)
#     avg_acc = total_correct / max(total_samples, 1)

#     return {
#         "loss": avg_loss,
#         "acc": avg_acc,
#         "num_samples": total_samples,
#     }


# def train_one_epoch(
#     model: nn.Module,
#     loader: DataLoader,
#     criterion: nn.Module,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
#     epoch: int,
#     log_interval: int = 100,
# ) -> Dict[str, float]:
#     model.train()

#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0

#     start_time = time.time()

#     for step, (images, targets) in enumerate(loader, start=1):
#         images = images.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         images = normalize_for_vit(images)

#         optimizer.zero_grad(set_to_none=True)

#         logits = model(images)
#         loss = criterion(logits, targets)
#         loss.backward()
#         optimizer.step()

#         batch_size = images.size(0)
#         total_loss += loss.item() * batch_size
#         total_correct += (logits.argmax(dim=1) == targets).sum().item()
#         total_samples += batch_size

#         if step % log_interval == 0:
#             running_loss = total_loss / max(total_samples, 1)
#             running_acc = total_correct / max(total_samples, 1)
#             print(
#                 f"[epoch {epoch:03d}] step {step:05d}/{len(loader):05d} | "
#                 f"loss={running_loss:.6f} acc={running_acc:.4f}"
#             )

#     elapsed = time.time() - start_time
#     avg_loss = total_loss / max(total_samples, 1)
#     avg_acc = total_correct / max(total_samples, 1)

#     return {
#         "loss": avg_loss,
#         "acc": avg_acc,
#         "num_samples": total_samples,
#         "time_sec": elapsed,
#     }

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
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

        images = normalize_for_vit(images)

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
    parser = argparse.ArgumentParser(description="Train Google ViT-B/16 on OHLC images.")

    # 数据
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--symbol_to_tar_json", type=str, required=True)
    parser.add_argument("--horizon_idx", type=int, required=True, choices=[0, 1, 2])

    # 模型
    parser.add_argument("--hf_model_name", type=str, default="google/vit-base-patch16-224-in21k")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260315)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--step_log_interval", type=int, default=100)
    parser.add_argument("--step_log_file", type=str, default="metrics_step.csv")

    # 输入设置
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--repeat_to_3ch", action="store_true", help="用原版ViT时建议打开")
    parser.add_argument("--validate_members", action="store_true", help="是否再次检查tar成员。正式训练一般不开。")

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

    exp_name = f"I{window}_R{horizon_days}_vit_b16_in21k"
    print(f"experiment = {exp_name}")
    print(f"window = {window}")
    print(f"horizon_days = {horizon_days}")
    print(f"target input shape = [3, 96, {target_width}]")
    print("finetune mode = full finetuning")
    print("peft/lora = not used")
    print("input normalization = ImageNet mean/std")

    config = vars(args).copy()
    config["window"] = window
    config["horizon_days"] = horizon_days
    config["target_width"] = target_width
    config["device"] = str(device)
    config["finetune_mode"] = "full"
    config["normalization"] = "imagenet_mean_std"

    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 样本统计
    # train_stats = count_split_stats(args.meta_path, "train", args.horizon_idx)
    # val_stats = count_split_stats(args.meta_path, "val", args.horizon_idx)
    # test_stats = count_split_stats(args.meta_path, "test", args.horizon_idx)

    # print("\n===== split stats before loader =====")
    # for split_name, stats in [("train", train_stats), ("val", val_stats), ("test", test_stats)]:
    #     print(
    #         f"{split_name}: "
    #         f"raw_split_rows={stats['raw_split_rows']}, "
    #         f"valid_label_rows={stats['valid_label_rows']}, "
    #         f"dropped_label_minus1={stats['dropped_label_minus1']}"
    #     )

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

    # DataLoader
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
        validate_members=args.validate_members,
        validate_verbose=args.validate_members,
    )

    # test_loader, test_set = build_dataloader(
    #     meta_path=args.meta_path,
    #     symbol_to_tar_json=args.symbol_to_tar_json,
    #     split="test",
    #     horizon_idx=args.horizon_idx,
    #     batch_size=args.batch_size,
    #     patch_size=args.patch_size,
    #     num_workers=args.num_workers,
    #     shuffle=False,
    #     repeat_to_3ch=args.repeat_to_3ch,
    #     normalize_to_01=True,
    #     return_meta=False,
    #     validate_members=args.validate_members,
    #     validate_verbose=args.validate_members,
    # )

    print("\n===== dataset size after loader filtering =====")
    print(f"train dataset size = {len(train_set)}")
    print(f"val dataset size   = {len(val_set)}")
    # print(f"test dataset size  = {len(test_set)}")

    model = HFViTBinaryClassifier(
        model_name=args.hf_model_name,
        num_labels=2,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n===== model stats =====")
    print(f"hf_model_name = {args.hf_model_name}")
    print(f"total params = {num_params:,}")
    print(f"trainable params = {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
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
        # train_metrics = train_one_epoch(
        #     model=model,
        #     loader=train_loader,
        #     criterion=criterion,
        #     optimizer=optimizer,
        #     device=device,
        #     epoch=epoch,
        #     log_interval=args.log_interval,
        # )

        # val_metrics = evaluate(
        #     model=model,
        #     loader=val_loader,
        #     criterion=criterion,
        #     device=device,
        # )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
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
            desc=f"Val {epoch}/{args.epochs}",
        )

        # row = {
        #     "epoch": epoch,
        #     "train_loss": train_metrics["loss"],
        #     "train_acc": train_metrics["acc"],
        #     "train_num_samples": train_metrics["num_samples"],
        #     "train_time_sec": train_metrics["time_sec"],
        #     "val_loss": val_metrics["loss"],
        #     "val_acc": val_metrics["acc"],
        #     "val_num_samples": val_metrics["num_samples"],
        # }
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
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            save_checkpoint(
                ckpt_path=best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                config=config,
            )
            print(f"saved new best checkpoint to: {best_ckpt}")

        save_checkpoint(
            ckpt_path=last_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc,
            config=config,
        )

        global_step += train_metrics["num_steps"]

    # print("\n===== training finished =====")
    # print(f"best_epoch = {best_epoch}")
    # print(f"best_val_acc = {best_val_acc:.6f}")

    print("\n===== training finished =====")
    print(f"best_epoch = {best_epoch}")
    print(f"best_val_acc = {best_val_acc:.6f}")

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "train_dataset_size": len(train_set),
                "val_dataset_size": len(val_set),
                "train_split_stats": train_stats,
                "val_split_stats": val_stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # print("\n===== evaluate best checkpoint on test =====")
    # ckpt = torch.load(best_ckpt, map_location=device)
    # model.load_state_dict(ckpt["model_state_dict"])

    # test_metrics = evaluate(
    #     model=model,
    #     loader=test_loader,
    #     criterion=criterion,
    #     device=device,
    #     desc="Test",
    # )

    # print(
    #     f"test_loss={test_metrics['loss']:.6f} "
    #     f"test_acc={test_metrics['acc']:.4f} "
    #     f"test_num_samples={test_metrics['num_samples']}"
    # )

    # with open(os.path.join(args.out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
    #     json.dump(
    #         {
    #             "best_epoch": best_epoch,
    #             "best_val_acc": best_val_acc,
    #             "test_loss": test_metrics["loss"],
    #             "test_acc": test_metrics["acc"],
    #             "test_num_samples": test_metrics["num_samples"],
    #             "train_dataset_size": len(train_set),
    #             "val_dataset_size": len(val_set),
    #             "test_dataset_size": len(test_set),
    #             "train_split_stats": train_stats,
    #             "val_split_stats": val_stats,
    #             "test_split_stats": test_stats,
    #         },
    #         f,
    #         indent=2,
    #         ensure_ascii=False,
    #     )

    train_set.close()
    val_set.close()
    # test_set.close()


if __name__ == "__main__":
    main()

"""
python train.py \
  --meta_path /workspace/label_npz/meta_N60_clean.npz \
  --symbol_to_tar_json /workspace/symbol_to_tar/symbol_to_tar_N60.json \
  --horizon_idx 0 \
  --hf_model_name google/vit-base-patch16-224-in21k \
  --epochs 5 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --num_workers 12 \
  --seed 20260315 \
  --patch_size 16 \
  --repeat_to_3ch \
  --warmup_ratio 0.05 \
  --step_log_interval 200 \
  --step_log_file metrics_step.csv \
  --out_dir /workspace/exp_I60_R5_vitb_in21k
"""