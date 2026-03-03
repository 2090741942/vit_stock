from __future__ import annotations

import os
import sys
import glob
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image


@dataclass
class ImageSpec:
    """
    用一个“配置对象”集中管理生成图片的参数。

    这里用到了 @dataclass（数据类）：
    - Python 会自动帮你生成 __init__（构造函数）、__repr__（打印展示）、__eq__（比较）等方法
    - 你只要声明字段名和默认值，就能像普通类一样 ImageSpec(...) 创建对象

    各字段含义：
    - height：输出图片总高度（像素）
    - ohlc_frac：总高度中分配给 OHLC+均线 的比例，剩下的用于成交量柱
    - day_px：每个交易日占用的水平像素宽度（这里固定 3 像素：open / 中线 / close）
    - draw_ma：是否绘制移动平均线
    - draw_volume：是否绘制成交量柱状图
    """

    height: int = 100
    ohlc_frac: float = 0.8
    day_px: int = 3
    draw_ma: bool = True
    draw_volume: bool = True


def _clip_int(x: int, lo: int, hi: int) -> int:
    """
    将整数 x 截断到 [lo, hi] 之间，避免越界。
    常用于像素坐标计算（超出图像范围会报错或写不到正确位置）。
    """
    return int(max(lo, min(hi, x)))


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Bresenham 直线算法：在像素网格上，用整数步进画一条“尽量连贯”的直线。

    返回值是一个点列表 [(x, y), ...]，表示从 (x0,y0) 到 (x1,y1) 之间
    每个应该点亮的像素坐标。用于画均线连接段。
    """
    pts = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return pts


def _to_row(val: float, vmin: float, vmax: float, ohlc_h: int) -> int:
    """
    把一个“数值 val”（比如某天的归一化价格）映射到 OHLC 区域的 y 像素坐标。

    约定：
    - y=0 在上方，y 增大向下（这是图像坐标系）
    - vmax 映射到顶部（0）
    - vmin 映射到底部（ohlc_h - 1）

    若遇到 NaN/inf 或 vmax<=vmin（无法归一化）则返回 0（兜底）。
    """
    if not np.isfinite(val) or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0
    t = (vmax - val) / (vmax - vmin)
    y = int(round(t * (ohlc_h - 1)))
    return _clip_int(y, 0, ohlc_h - 1)


def build_window_image(
    win: pd.DataFrame,
    n: int,
    spec: ImageSpec,
    price_cols: Tuple[str, str, str, str] = ("OpenPrice", "HighPrice", "LowPrice", "ClosePrice"),
    volume_col: str = "Volume",
) -> Optional[Image.Image]:
    """
    用一个长度为 n 的窗口 win（同一只股票的连续 n 天数据）生成一张图片。

    返回：
    - 成功：返回 PIL.Image 对象
    - 失败：返回 None（比如窗口起止 close 非法、整体无法归一化等）

    关键思路：
    1) 先用 Close 构造“相对价格序列 p”，从 1.0 开始累乘收益率
       这样不同股票/不同价格量级可以统一到相同尺度（相对走势）
    2) 再把 O/H/L 相对于 C 缩放到同一相对尺度：o = p*(O/C), h = p*(H/C), l = p*(L/C), c = p
    3) 将这些值映射到像素坐标，画 OHLC（每一天 3 像素：open点 | high-low竖线 | close点）
    4) 可选：画均线（rolling mean），并用 Bresenham 把相邻点连起来
    5) 可选：画成交量柱（按窗口内最大成交量归一化）
    """
    if len(win) != n:
        raise ValueError(f"window length must be {n}, got {len(win)}")

    op, hi, lo, cl = price_cols
    C = win[cl].astype(float)

    # 要求窗口第一天和最后一天的 close 必须是有效数，否则相对价格无法稳定定义
    if not np.isfinite(C.iloc[0]) or not np.isfinite(C.iloc[-1]):
        return None

    # 构造相对价格序列 p：p[0]=1.0，之后根据 close 的涨跌累计
    # 这里 p 是一个 Series，index 与 win 一致，便于后续对齐运算
    p = pd.Series(np.nan, index=win.index, dtype=float)

    # first_valid_index：返回第一个非 NaN 的索引标签；若全是 NaN 则返回 None
    first_valid = C.first_valid_index()
    if first_valid is None:
        return None

    # get_loc：把“索引标签”转换为“位置下标”
    # 这里强制要求窗口第 0 个位置就应该有有效 close，否则认为窗口开头有缺失，不画
    first_pos = win.index.get_loc(first_valid)
    if first_pos != 0:
        return None

    p.iloc[0] = 1.0
    last_close = float(C.iloc[0])
    last_p = 1.0

    for i in range(1, n):
        c_now = C.iloc[i]
        if np.isfinite(c_now) and c_now > 0 and np.isfinite(last_close) and last_close > 0:
            ret = float(c_now) / last_close - 1.0
            last_p = (1.0 + ret) * last_p
            p.iloc[i] = last_p
            last_close = float(c_now)
        else:
            # 如果这一天 close 不合法，就把相对价格设为 NaN
            # 后续绘图时会跳过这一天
            p.iloc[i] = np.nan

    # 读取 O/H/L，并换算到相对尺度（与 p 同尺度）
    O = win[op].astype(float)
    H = win[hi].astype(float)
    L = win[lo].astype(float)

    # 这几行是“逐元素对齐运算”：
    # - pandas 会按 index 对齐后做乘法/除法
    # - o/h/l/c 都是 Series，长度 n
    o = p * (O / C)
    h = p * (H / C)
    l = p * (L / C)
    c = p

    ma = None
    if spec.draw_ma:
        # rolling(window=n, min_periods=1).mean()：
        # - rolling：滑动窗口
        # - window=n：最多用最近 n 个点算均值
        # - min_periods=1：窗口里只要有 1 个有效点就会输出均值（否则一开始会全是 NaN）
        ma = c.rolling(window=n, min_periods=1).mean()
        # 如果当天 c 是 NaN，则均线也强制 NaN，避免“缺交易日但均线还连着”
        ma[c.isna()] = np.nan

    # 把需要参与归一化的序列拼到一起，统一求 vmax/vmin
    parts = [o, h, l, c]
    if spec.draw_ma and ma is not None:
        parts.append(ma)
    arr = pd.concat(parts, axis=1).to_numpy()

    vmax = np.nanmax(arr)
    vmin = np.nanmin(arr)
    if not np.isfinite(vmax) or not np.isfinite(vmin) or vmax <= vmin:
        return None

    # 计算图像尺寸与 OHLC/成交量区域高度
    H_total = spec.height
    ohlc_h = int(round(H_total * spec.ohlc_frac))
    vol_h = H_total - ohlc_h
    width = n * spec.day_px

    # PIL 模式 "L"：8-bit 灰度图，像素值 0=黑，255=白
    img = Image.new("L", (width, H_total), 0)

    # img.load() 返回一个“像素访问对象”，可以用 px[x, y] = value 直接写像素
    px = img.load()

    # 绘制 OHLC：每一天占 3 像素宽
    # x0：open 点；x1：high-low 竖线；x2：close 点
    for d in range(n):
        x0 = d * spec.day_px
        x1 = x0 + 1
        x2 = x0 + 2

        # 如果当天 o/h/l/c 任意一个为 NaN，就跳过
        if not (np.isfinite(o.iloc[d]) and np.isfinite(h.iloc[d]) and np.isfinite(l.iloc[d]) and np.isfinite(c.iloc[d])):
            continue

        yo = _to_row(float(o.iloc[d]), vmin, vmax, ohlc_h)
        yh = _to_row(float(h.iloc[d]), vmin, vmax, ohlc_h)
        yl = _to_row(float(l.iloc[d]), vmin, vmax, ohlc_h)
        yc = _to_row(float(c.iloc[d]), vmin, vmax, ohlc_h)

        # sorted([yh, yl])：返回升序后的两个值，保证 y_top <= y_bot
        y_top, y_bot = sorted([yh, yl])

        # 画 high-low 的竖线：把 x1 这一列从 y_top 到 y_bot 的像素点亮
        for yy in range(y_top, y_bot + 1):
            px[x1, yy] = 255

        # open 与 close 用单点表示
        px[x0, yo] = 255
        px[x2, yc] = 255

    # 绘制均线：每一天取中间像素 x1 作为均线点，并把相邻点连线
    if spec.draw_ma and ma is not None:
        prev = None
        for d in range(n):
            if not np.isfinite(ma.iloc[d]):
                prev = None
                continue
            x = d * spec.day_px + 1
            y = _to_row(float(ma.iloc[d]), vmin, vmax, ohlc_h)
            px[x, y] = 255

            if prev is not None:
                x0, y0 = prev
                for xx, yy in _bresenham_line(x0, y0, x, y):
                    if 0 <= xx < width and 0 <= yy < ohlc_h:
                        px[xx, yy] = 255

            prev = (x, y)

    # 绘制成交量柱：在底部 vol_h 区域画竖直柱
    # 做法：用窗口内最大成交量 vmax_v 做归一化，柱高 ∝ v / vmax_v
    if spec.draw_volume and volume_col in win.columns and vol_h > 0:
        V = win[volume_col].astype(float)
        vmax_v = np.nanmax(V.to_numpy())
        if np.isfinite(vmax_v) and vmax_v > 0:
            for d in range(n):
                v = V.iloc[d]
                if not np.isfinite(v) or v <= 0:
                    continue
                x = d * spec.day_px + 1

                bar_h = int(round((v / vmax_v) * (vol_h - 1)))
                bar_h = _clip_int(bar_h, 0, vol_h - 1)

                # 从图像最底部往上点亮 bar_h 高度
                for yy in range(H_total - 1, H_total - 1 - bar_h - 1, -1):
                    px[x, yy] = 255

    return img


def generate_images(
    df: pd.DataFrame,
    out_dir: str,
    windows: Iterable[int] = (5, 20, 60),
    spec: ImageSpec = ImageSpec(),
    symbol_col: str = "Symbol",
    date_col: str = "TradingDate",
    price_cols: Tuple[str, str, str, str] = ("OpenPrice", "HighPrice", "LowPrice", "ClosePrice"),
    volume_col: str = "Volume",
    trading_calendar: Optional[pd.DatetimeIndex] = None,
    step: int = 1,
) -> None:
    """
    从全量 df 生成图片文件，并写到 out_dir。

    主要流程：
    1) 按股票代码分组（groupby symbol）
    2) 可选：按 trading_calendar 重新对齐索引（reindex），把缺失交易日补成 NaN 行
       这样“节假日缺失”会在窗口中显式出现，并且 build_window_image 会跳过缺失日，避免连起来
    3) 对每个窗口长度 n（例如 60/20/5），滑动窗口生成图片
    4) 文件名包含：股票代码、窗口结束日期、窗口长度

    一些语法点：
    - os.makedirs(out_dir, exist_ok=True)：如果目录不存在就创建；存在也不报错
    - groupby(..., sort=False)：保持原数据中 symbol 出现的顺序（不额外排序），通常更快也更可控
    - tqdm(range(...), desc=..., file=sys.stdout)：显示进度条；输出到 stdout 兼容部分 IDE
    """
    os.makedirs(out_dir, exist_ok=True)

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values([symbol_col, date_col], inplace=True)

    # 先在外面算好总股票数，让外层 tqdm 能显示总进度
    total_syms = df[symbol_col].nunique()

    # 外层：股票维度进度条
    for sym, g in tqdm(
        df.groupby(symbol_col, sort=False),
        total=total_syms,
        desc="Processing symbols",
        file=sys.stdout,
    ):
        g = g.sort_values(date_col).set_index(date_col)

        op, hi, lo, cl = price_cols

        if volume_col in g.columns:
            v0 = g[volume_col].astype(float).fillna(0) <= 0
        else:
            v0 = False

        o_ = g[op].astype(float)
        h_ = g[hi].astype(float)
        l_ = g[lo].astype(float)
        c_ = g[cl].astype(float)

        no_trade = v0 & (o_ == h_) & (h_ == l_) & (l_ == c_)

        g.loc[no_trade, list(price_cols)] = np.nan
        if volume_col in g.columns:
            g.loc[no_trade, volume_col] = np.nan

        if trading_calendar is not None:
            g = g.reindex(trading_calendar)

        # 第二层：窗口长度（60/20/5）正常循环，不加 tqdm（避免太多条）
        for n in windows:
            if len(g) < n:
                continue

            window_out_dir = os.path.join(out_dir, f"N{n}")
            os.makedirs(window_out_dir, exist_ok=True)

            idx = g.index

            # 内层：日期滑动窗口进度条（对每只股票 + 每个 n）
            for end_i in tqdm(
                range(n - 1, len(g), step),
                desc=f"{sym} | N{n}",
                file=sys.stdout,
                leave=False,   # 关键：内层结束后不留下，界面不会刷屏
            ):
                start_i = end_i - n + 1
                win = g.iloc[start_i:end_i + 1]
                if len(win) != n:
                    continue

                img = build_window_image(
                    win=win,
                    n=n,
                    spec=spec,
                    price_cols=price_cols,
                    volume_col=volume_col,
                )
                if img is None:
                    continue

                end_date = idx[end_i].strftime("%Y%m%d")
                fname = f"{sym}_end{end_date}_N{n}.png"
                img.save(os.path.join(window_out_dir, fname))

    # for sym, g in df.groupby(symbol_col, sort=False):
    #     g = g.sort_values(date_col).set_index(date_col)

    #     # 识别“非交易日行”：volume<=0 且 O=H=L=C
    #     # 这些行通常是“用前收填充”或“停牌/缺失被填”的假数据
    #     op, hi, lo, cl = price_cols

    #     if volume_col in g.columns:
    #         v0 = g[volume_col].astype(float).fillna(0) <= 0
    #     else:
    #         v0 = False

    #     o_ = g[op].astype(float)
    #     h_ = g[hi].astype(float)
    #     l_ = g[lo].astype(float)
    #     c_ = g[cl].astype(float)

    #     no_trade = v0 & (o_ == h_) & (h_ == l_) & (l_ == c_)

    #     # 把非交易日的 OHLC 和 volume 置为 NaN
    #     g.loc[no_trade, list(price_cols)] = np.nan
    #     if volume_col in g.columns:
    #         g.loc[no_trade, volume_col] = np.nan

    #     if trading_calendar is not None:
    #         # reindex：把索引强行对齐到 calendar
    #         # - 原来没有的日期会新增一行，并且各列都是 NaN
    #         g = g.reindex(trading_calendar)

    #     for n in windows:
    #         if len(g) < n:
    #             continue

    #         window_out_dir = os.path.join(out_dir, f"N{n}")
    #         os.makedirs(window_out_dir, exist_ok=True)

    #         idx = g.index

    #         # end_i 是窗口的结束位置（包含）
    #         # step=1 表示每天滑动一步；如果你想减少图片数量可以把 step 调大
    #         for end_i in tqdm(range(n - 1, len(g), step), desc=f"{n} day image generating", file=sys.stdout):
    #             start_i = end_i - n + 1
    #             win = g.iloc[start_i:end_i + 1]
    #             if len(win) != n:
    #                 continue

    #             img = build_window_image(
    #                 win=win,
    #                 n=n,
    #                 spec=spec,
    #                 price_cols=price_cols,
    #                 volume_col=volume_col,
    #             )
    #             if img is None:
    #                 continue

    #             end_date = idx[end_i].strftime("%Y%m%d")

    #             # zfill(6)：把股票代码左侧补 0 到 6 位（如 1 -> "000001"）
    #             # sym_str = str(sym).zfill(6)

    #             # f-string：在字符串里用 {变量} 直接插入值
    #             fname = f"{sym}_end{end_date}_N{n}.png"
    #             img.save(os.path.join(window_out_dir, fname))


if __name__ == "__main__":
    """
    只有当这个文件作为“脚本直接运行”时，才会执行这里的代码。
    如果你把它 import 到别的文件中，这一段不会执行。
    """

    # r"..." 是 raw string（原始字符串），反斜杠不会被当作转义字符
    # Windows 路径里经常用到，避免 \n \t 之类被解释
    DATA_ROOT = r"/workspace/data"
    OUT_DIR = "/workspace/ohlc_images"

    # glob 模式：** 表示递归子目录匹配
    pattern = os.path.join(DATA_ROOT, "**", "TRD_BwardQuotation*.csv")

    # glob.glob(..., recursive=True) 返回所有匹配的路径
    # sorted(...) 用于排序，保证处理顺序稳定
    # 这里排除路径里含 "[DES]" 的文件（通常是描述文件或不需要的版本）
    csv_files = sorted(p for p in glob.glob(pattern, recursive=True) if "[DES]" not in p)
    csv_files = csv_files[6:]
    # total = 0
    # for p in csv_files:
    #     df_i = pd.read_csv(p)
    #     r = len(df_i)
    #     total += r
    #     print(r, p)

    # print("逐文件行数求和:", total)
    # for i in csv_files:
    #     print(i)
    if not csv_files:
        raise FileNotFoundError(f"No csv files matched: {pattern}")

    # 逐文件读入，再 concat 合并成一个大表
    frames = [pd.read_csv(p, dtype={"Symbol": str}) for p in csv_files]

    # ignore_index=True：合并后重新从 0..N-1 生成新索引
    df = pd.concat(frames, ignore_index=True)

    # print(df["Symbol"].unique())

    # print(df["Symbol"].head(20))
    # print(df["Symbol"].dtype)
    # print(df["Symbol"].nunique())
    # print(df["Symbol"].min(), df["Symbol"].max())

    # 用数据里出现过的 TradingDate 构造一个“交易日历”
    # 注意：如果你希望用官方交易日历（包含未来/严格缺口），应改用外部日历源
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(df["TradingDate"]).unique()))

    # 自定义图片规格：这里把总高度设为 100，OHLC 占 80%，每个交易日 3 像素
    spec = ImageSpec(height=100, ohlc_frac=0.8, day_px=3, draw_ma=True, draw_volume=True)

    generate_images(
        df=df,
        out_dir=OUT_DIR,
        windows=(60, 20, 5),
        spec=spec,
        symbol_col="Symbol",
        date_col="TradingDate",
        price_cols=("OpenPrice", "HighPrice", "LowPrice", "ClosePrice"),
        volume_col="Volume",
        trading_calendar=calendar,
        step=1,
    )

    print("Done.")