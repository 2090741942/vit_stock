from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =========================
# 基础数据结构
# =========================

@dataclass
class SampleRecord:
    """
    单个样本的索引信息。
    """
    symbol: str
    end_date: str
    target: int
    tar_path: str
    member_name: str


# =========================
# 工具函数
# =========================

def ceil_to_multiple(x: int, multiple: int) -> int:
    """
    向上取到 multiple 的整数倍。
    例如:
        ceil_to_multiple(15, 16) -> 16
        ceil_to_multiple(60, 16) -> 64
        ceil_to_multiple(180, 16) -> 192
    """
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((x + multiple - 1) // multiple) * multiple


def get_raw_width(window: int) -> int:
    """
    原始图片宽度：一天 3 个像素。
    """
    return window * 3


def get_target_width(window: int, patch_size: int = 16) -> int:
    """
    计算 pad 后目标宽度，仅对宽度做最小 pad。
    高度固定 96，不处理。
    """
    raw_width = get_raw_width(window)
    return ceil_to_multiple(raw_width, patch_size)


def symmetric_pad_width(
    x: torch.Tensor,
    target_width: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    对图像张量做“仅宽度方向”的对称 pad。

    参数
    ----
    x : torch.Tensor
        形状 [C, H, W]
    target_width : int
        pad 后目标宽度
    pad_value : float
        pad 的填充值，默认 0（黑色）

    返回
    ----
    torch.Tensor
        pad 后张量，形状 [C, H, target_width]
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [C, H, W], got shape={tuple(x.shape)}")

    _, _, w = x.shape
    if w > target_width:
        raise ValueError(
            f"current width {w} is larger than target_width {target_width}"
        )

    total_pad = target_width - w
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left

    # 对 CHW 图像，pad 顺序为 (left, right, top, bottom)
    return F.pad(x, pad=(pad_left, pad_right, 0, 0), mode="constant", value=pad_value)


def build_member_name(window: int, symbol: str, end_date: str) -> str:
    """
    根据 tar 内部成员命名规则构造成员路径。

    当前按你的 tar 结构默认使用：
        {symbol}/{symbol}_end{end_date}_N{window}.png

    例如：
        000001/000001_end20180206_N60.png

    如果你后面确认 tar 内部其实还有一层 ohlc_image_N{N}/，
    只需要改这里，不用改别处。
    """
    return f"{symbol}/{symbol}_end{end_date}_N{window}.png"


def load_symbol_to_tar(json_path: str) -> Dict[str, str]:
    """
    读取 symbol -> tar_path 映射表。
    """
    json_path = str(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    return {str(k): str(v) for k, v in mapping.items()}


# =========================
# 从 meta 读取样本索引
# =========================

def load_records_from_meta(
    meta_path: str,
    split: str,
    horizon_idx: int,
    symbol_to_tar: Dict[str, str],
) -> Tuple[List[SampleRecord], int]:
    """
    从 meta_N*.npz 中读取并筛选当前任务对应的样本记录。

    参数
    ----
    meta_path : str
        例如 meta_N5.npz / meta_N20.npz / meta_N60.npz
    split : str
        'train' / 'val' / 'test'
    horizon_idx : int
        0 -> future 5d
        1 -> future 20d
        2 -> future 60d
    symbol_to_tar : Dict[str, str]
        symbol 到 tar 路径的映射

    返回
    ----
    records : List[SampleRecord]
        当前 split 且当前 horizon label != -1 的样本
    window : int
        当前 meta 对应的窗口长度 N
    """
    split_map = {"train": 0, "val": 1, "test": 2}

    if split not in split_map:
        raise ValueError(f"split must be one of {list(split_map.keys())}, got {split}")
    if horizon_idx not in (0, 1, 2):
        raise ValueError(f"horizon_idx must be 0/1/2, got {horizon_idx}")

    meta = np.load(meta_path, allow_pickle=True)

    labels = meta["labels"]                  # [M, 3]
    symbols = meta["symbol"].astype(str)     # [M]
    end_dates = meta["end_date"].astype(str) # [M]
    splits = meta["split"]                   # [M]
    window = int(meta["window"][0])          # 当前 N

    y = labels[:, horizon_idx].astype(np.int8)

    # 当前 split 且标签有效
    mask = (splits == split_map[split]) & (y != -1)

    symbols = symbols[mask]
    end_dates = end_dates[mask]
    y = y[mask]

    missing_symbols = []
    records: List[SampleRecord] = []

    for symbol, end_date, target in zip(symbols, end_dates, y):
        tar_path = symbol_to_tar.get(symbol)
        if tar_path is None:
            missing_symbols.append(symbol)
            continue

        records.append(
            SampleRecord(
                symbol=symbol,
                end_date=end_date,
                target=int(target),
                tar_path=tar_path,
                member_name=build_member_name(window, symbol, end_date),
            )
        )

    if missing_symbols:
        uniq_missing = sorted(set(missing_symbols))
        preview = uniq_missing[:20]
        raise KeyError(
            f"{len(uniq_missing)} symbols are missing in symbol_to_tar mapping. "
            f"Examples: {preview}"
        )

    return records, window

def filter_records_by_existing_members(
    records: List[SampleRecord],
    verbose: bool = True,
) -> List[SampleRecord]:
    """
    按 tar 分组，批量检查 member_name 是否真实存在于 tar 中。
    不存在的样本会被剔除。

    这样可以处理：
    - label/index 能拼出路径，但 tar 中实际没有该图片
    - 少量图片与 label 对不上的情况

    参数
    ----
    records : List[SampleRecord]
        初始样本列表
    verbose : bool
        是否打印过滤统计信息

    返回
    ----
    List[SampleRecord]
        过滤后的样本列表
    """
    from collections import defaultdict

    grouped: Dict[str, List[SampleRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.tar_path].append(rec)

    kept_records: List[SampleRecord] = []
    missing_count = 0
    bad_tar_count = 0

    for tar_path, recs in grouped.items():
        try:
            with tarfile.open(tar_path, mode="r") as tf:
                member_names = {
                    m.name for m in tf
                    if m.isfile()
                }
        except Exception as e:
            bad_tar_count += len(recs)
            if verbose:
                print(f"[WARN] 无法读取 tar，跳过其中全部样本: {tar_path}\n       {e}")
            continue

        for rec in recs:
            if rec.member_name in member_names:
                kept_records.append(rec)
            else:
                missing_count += 1

    if verbose:
        print(
            f"[filter_records_by_existing_members] "
            f"input={len(records)}, kept={len(kept_records)}, "
            f"missing_member={missing_count}, bad_tar_dropped={bad_tar_count}"
        )

    return kept_records


# =========================
# 主 Dataset
# =========================

class TarOHLCImageDataset(Dataset):
    """
    从 tar 包中动态读取 OHLC 图片，并根据 meta_N*.npz 提供标签与切分。

    处理流程：
    1. 从 tar 中按成员名读图片
    2. 转成灰度图
    3. 变成 [1, H, W] tensor
    4. 宽度方向做最小对称 pad
    5. 如需要，复制成 3 通道 [3, H, W]
    """

    def __init__(
        self,
        records: List[SampleRecord],
        window: int,
        patch_size: int = 16,
        repeat_to_3ch: bool = True,
        normalize_to_01: bool = True,
        return_meta: bool = False,
    ) -> None:
        super().__init__()

        self.records = records
        self.window = int(window)
        self.patch_size = int(patch_size)
        self.repeat_to_3ch = bool(repeat_to_3ch)
        self.normalize_to_01 = bool(normalize_to_01)
        self.return_meta = bool(return_meta)

        self.target_width = get_target_width(self.window, self.patch_size)

        # 每个 worker 进程内部缓存 tar 句柄，避免每个样本反复打开/关闭 tar
        self._tar_handles: Dict[str, tarfile.TarFile] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _get_tar_handle(self, tar_path: str) -> tarfile.TarFile:
        """
        懒加载 tar 句柄，并在当前 worker 内缓存。
        """
        handle = self._tar_handles.get(tar_path)
        if handle is None:
            handle = tarfile.open(tar_path, mode="r")
            self._tar_handles[tar_path] = handle
        return handle

    def _read_image_from_tar(self, tar_path: str, member_name: str) -> Image.Image:
        """
        从 tar 中读取单张图片，返回 PIL.Image（灰度）。
        """
        tf = self._get_tar_handle(tar_path)

        try:
            extracted = tf.extractfile(member_name)
        except KeyError as e:
            raise FileNotFoundError(
                f"Member not found in tar.\n"
                f"tar_path: {tar_path}\n"
                f"member_name: {member_name}"
            ) from e

        if extracted is None:
            raise FileNotFoundError(
                f"extractfile returned None.\n"
                f"tar_path: {tar_path}\n"
                f"member_name: {member_name}"
            )

        data = extracted.read()
        img = Image.open(io.BytesIO(data)).convert("L")  # 灰度图，单通道
        return img

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """
        PIL 灰度图 -> torch tensor [1, H, W]
        """
        arr = np.array(img, dtype=np.uint8)      # [H, W]
        x = torch.from_numpy(arr).unsqueeze(0)   # [1, H, W], uint8

        if self.normalize_to_01:
            x = x.float().div(255.0)
        else:
            x = x.float()

        return x

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        预处理步骤：
        1. 宽度方向最小对称 pad
        2. 必要时复制为 3 通道
        """
        x = symmetric_pad_width(
            x,
            target_width=self.target_width,
            pad_value=0.0,
        )

        if self.repeat_to_3ch:
            x = x.repeat(3, 1, 1)

        return x

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        img = self._read_image_from_tar(rec.tar_path, rec.member_name)
        x = self._pil_to_tensor(img)
        x = self._preprocess(x)

        y = torch.tensor(rec.target, dtype=torch.long)

        if self.return_meta:
            meta = {
                "symbol": rec.symbol,
                "end_date": rec.end_date,
                "tar_path": rec.tar_path,
                "member_name": rec.member_name,
                "window": self.window,
                "target_width": self.target_width,
            }
            return x, y, meta

        return x, y

    def close(self) -> None:
        """
        手动关闭已打开的 tar 句柄。
        """
        for handle in self._tar_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._tar_handles.clear()

    def __del__(self):
        self.close()


# =========================
# DataLoader 构造
# =========================

def build_dataset(
    meta_path: str,
    symbol_to_tar_json: str,
    split: str,
    horizon_idx: int,
    patch_size: int = 16,
    repeat_to_3ch: bool = True,
    normalize_to_01: bool = True,
    return_meta: bool = False,
    validate_members: bool = False,
    validate_verbose: bool = True,
) -> TarOHLCImageDataset:
    """
    构造单个 Dataset。
    """
    symbol_to_tar = load_symbol_to_tar(symbol_to_tar_json)

    records, window = load_records_from_meta(
        meta_path=meta_path,
        split=split,
        horizon_idx=horizon_idx,
        symbol_to_tar=symbol_to_tar,
    )

    if validate_members:
        records = filter_records_by_existing_members(
            records,
            verbose=validate_verbose,
        )

    dataset = TarOHLCImageDataset(
        records=records,
        window=window,
        patch_size=patch_size,
        repeat_to_3ch=repeat_to_3ch,
        normalize_to_01=normalize_to_01,
        return_meta=return_meta,
    )
    return dataset


def build_dataloader(
    meta_path: str,
    symbol_to_tar_json: str,
    split: str,
    horizon_idx: int,
    batch_size: int,
    patch_size: int = 16,
    num_workers: int = 8,
    shuffle: Optional[bool] = None,
    repeat_to_3ch: bool = True,
    normalize_to_01: bool = True,
    return_meta: bool = False,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: int = 4,
    drop_last: bool = False,
    validate_members: bool = False,
    validate_verbose: bool = True,
) -> Tuple[DataLoader, TarOHLCImageDataset]:
    """
    构造 DataLoader，并返回 (loader, dataset)

    参数
    ----
    split : str
        'train' / 'val' / 'test'
    horizon_idx : int
        0 -> future 5d
        1 -> future 20d
        2 -> future 60d
    """
    if shuffle is None:
        shuffle = (split == "train")

    if persistent_workers is None:
        persistent_workers = (num_workers > 0)

    dataset = build_dataset(
        meta_path=meta_path,
        symbol_to_tar_json=symbol_to_tar_json,
        split=split,
        horizon_idx=horizon_idx,
        patch_size=patch_size,
        repeat_to_3ch=repeat_to_3ch,
        normalize_to_01=normalize_to_01,
        return_meta=return_meta,
        validate_members=validate_members,
        validate_verbose=validate_verbose,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=drop_last,
    )

    return loader, dataset


# =========================
# 调试辅助
# =========================

def inspect_one_batch(
    meta_path: str,
    symbol_to_tar_json: str,
    split: str,
    horizon_idx: int,
    batch_size: int = 4,
    patch_size: int = 16,
    num_workers: int = 0,
    validate_members: bool = False,
    validate_verbose: bool = True,
) -> None:
    """
    用于快速检查 batch 形状、标签、meta 信息，以及可选的图片成员有效性过滤。
    """
    loader, dataset = build_dataloader(
        meta_path=meta_path,
        symbol_to_tar_json=symbol_to_tar_json,
        split=split,
        horizon_idx=horizon_idx,
        batch_size=batch_size,
        patch_size=patch_size,
        num_workers=num_workers,
        shuffle=False,
        repeat_to_3ch=True,
        normalize_to_01=True,
        return_meta=True,
        validate_members=validate_members,
        validate_verbose=validate_verbose,
    )

    print("dataset size =", len(dataset))

    batch = next(iter(loader))
    images, targets, metas = batch

    print("images.shape =", images.shape)
    print("targets.shape =", targets.shape)
    print("targets[:10] =", targets[:10])

    print("meta keys =", metas.keys())
    print("symbols[:5] =", metas["symbol"][:5])
    print("end_dates[:5] =", metas["end_date"][:5])

    dataset.close()


if __name__ == "__main__":
    # 这里只做一个轻量示例，不会真的运行，按需改路径再手动测试。
    #
    inspect_one_batch(
        meta_path="/workspace/label_npz/meta_N60.npz",
        symbol_to_tar_json="/workspace/symbol_to_tar/symbol_to_tar_N60.json",
        split="train",
        horizon_idx=0,
        batch_size=4,
        patch_size=16,
        num_workers=0,
    )
    #
    print("dataset.py loaded successfully.")