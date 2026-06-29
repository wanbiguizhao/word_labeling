"""
规则 vs 模型 切割对比可视化工具

功能：对指定的行图像生成五行对比图
  第1行：原始行图像
  第2行：规则切割结果（原始间隔）
  第3行：规则切割结果（后处理合并后）
  第4行：模型预测结果
  第5行：模型概率热力图

用法：
  python visualize_comparison.py <line_id> --data_base_path /path/to/datahome
  python visualize_comparison.py <line_id> --data_base_path /path/to/datahome --model_path /path/to/model.pth --save_dir ./output
"""

import math
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import click
from PIL import Image, ImageDraw

# 添加项目根目录到路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from ai_model.inference.infer import CharSegmentPredictor


# ===================== 可视化配置 =====================
SEP_LINE_COLOR = (128, 0, 128)    # 紫色分隔线
SEP_LINE_WIDTH = 2                 # 分隔线宽度
CUT_LINE_WIDTH = 1                 # 切割线宽度
START_COLOR = (255, 0, 0)          # 红色 = 字符起点
END_COLOR = (0, 255, 0)            # 绿色 = 字符终点
PROB_HEIGHT = 51                   # 概率图高度（像素）
PROB_MAX_PIXEL = 50                # 概率条最大高度


# ===================== 合并后处理默认参数 =====================
DEFAULT_MERGE_MIN_GAP = 3          # 最小间隙（与 segment_config.py 保持一致）
DEFAULT_MERGE_SINGLE_RATIO = 0.7   # 单字符宽高比上限
DEFAULT_MERGE_MIN_RATIO = 0.5      # 合并后宽高比下限
DEFAULT_MERGE_MAX_RATIO = 1.5      # 合并后宽高比上限


# ===================== 合并后处理函数 =====================

def merge_rule_intervals(
    intervals: List[Tuple[int, int]],
    image_height: int,
    min_gap: int = DEFAULT_MERGE_MIN_GAP,
    single_ratio: float = DEFAULT_MERGE_SINGLE_RATIO,
    min_ratio: float = DEFAULT_MERGE_MIN_RATIO,
    max_ratio: float = DEFAULT_MERGE_MAX_RATIO
) -> List[Tuple[int, int]]:
    """
    模拟 postprocess_merge_chars 的合并逻辑

    合并条件（全部满足才合并）：
      1. 间隙 < min_gap
      2. 两个字符都窄（width/height < single_ratio）
      3. 合并后的宽高比在 [min_ratio, max_ratio] 范围内

    Args:
        intervals: [(start, end), ...] 按列排序的区间
        image_height: 行图像高度（作为字符高度）
        min_gap: 最小间隙像素数
        single_ratio: 单字符宽高比上限
        min_ratio: 合并后宽高比下限
        max_ratio: 合并后宽高比上限

    Returns:
        合并后的区间列表
    """
    if len(intervals) < 2:
        return intervals[:]

    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]
        gap = current[0] - last[1] - 1

        if gap >= 0 and gap < min_gap:
            w1 = last[1] - last[0]
            w2 = current[1] - current[0]
            r1 = w1 / image_height
            r2 = w2 / image_height

            # 条件2：两个字符都窄
            if r1 < single_ratio and r2 < single_ratio:
                merged_w = current[1] - last[0]
                merged_r = merged_w / image_height

                # 条件3：合并后比例合理
                if min_ratio < merged_r < max_ratio:
                    merged[-1] = (last[0], current[1])
                    continue

        merged.append(current)

    return merged


def load_rule_intervals(rule_json_path: Path) -> List[Tuple[int, int]]:
    """
    从规则切割 JSON 中提取字符区间

    Args:
        rule_json_path: 规则切割 JSON 路径

    Returns:
        [(start, end), ...] 字符区间列表
    """
    with open(rule_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    intervals = []
    for char in data.get('chars', []):
        start = char.get('col_start', 0)
        end = char.get('col_end', 0)
        if end > start:
            intervals.append((start, end))

    return intervals


def compute_prob_map(pred_prob: np.ndarray, orig_width: int, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    将模型概率映射回原始图像宽度

    Args:
        pred_prob: 模型输出的概率数组（缩放后宽度）
        orig_width: 原始图像宽度
        scale: 缩放比例

    Returns:
        (prob_map, bar_heights) 映射到原始宽度的概率和条高度
    """
    inv_scale = 1.0 / scale if scale > 0 else 1.0
    resized_w = len(pred_prob)

    prob_map = np.zeros(orig_width, dtype=np.float32)
    count_map = np.zeros(orig_width, dtype=np.int32)

    for resized_col in range(resized_w):
        prob = pred_prob[resized_col]
        orig_col = int(round(resized_col * inv_scale))
        orig_col = min(max(orig_col, 0), orig_width - 1)
        prob_map[orig_col] += prob
        count_map[orig_col] += 1

    # 平均处理：多个缩放列映射到同一原始列时取平均
    mask = count_map > 0
    prob_map[mask] /= count_map[mask]

    bar_heights = np.ceil(prob_map * PROB_MAX_PIXEL).astype(np.int32)

    return prob_map, bar_heights


def draw_comparison(
    line_img: np.ndarray,
    rule_intervals: List[Tuple[int, int]],
    model_intervals: List[Tuple[int, int]],
    pred_prob: np.ndarray,
    scale: float,
    save_path: Path,
    rule_intervals_merged: Optional[List[Tuple[int, int]]] = None
) -> None:
    """
    绘制五层对比可视化图

    第1行：原始行图像
    第2行：原图 + 规则切割线（原始间隔）
    第3行：原图 + 规则切割线（后处理合并后）
    第4行：原图 + 模型切割线
    第5行：模型概率热力图

    Args:
        line_img: 原始灰度行图像 (H × W)
        rule_intervals: 规则切割区间（合并前）[(start, end), ...]
        model_intervals: 模型预测区间 [(start, end), ...]
        pred_prob: 模型预测概率数组（缩放后尺寸）
        scale: 缩放比例
        save_path: 保存路径
        rule_intervals_merged: 规则切割区间（合并后），不传则只画4行
    """
    H, W = line_img.shape
    sep = SEP_LINE_WIDTH

    # 计算总行数和总高度
    has_merged = rule_intervals_merged is not None and len(rule_intervals_merged) > 0
    num_image_rows = 4 if has_merged else 3  # 原始+规则+规则合并后+模型 / 原始+规则+模型
    total_height = H * num_image_rows + sep * (num_image_rows - 1) + PROB_HEIGHT

    # 转 PIL 彩色图像
    base_img = Image.fromarray(line_img).convert("RGB")

    # 创建画布
    canvas = Image.new("RGB", (W, total_height), (255, 255, 255))

    # =====================================
    # 第1行：原始图像
    # =====================================
    current_y = 0
    canvas.paste(base_img, (0, current_y))
    current_y += H

    draw = ImageDraw.Draw(canvas)

    # 紫色分隔线 1
    draw.line([(0, current_y), (W, current_y)], fill=SEP_LINE_COLOR, width=sep)
    current_y += sep

    # =====================================
    # 第2行：原图 + 规则切割线（原始间隔）
    # =====================================
    rule_img = base_img.copy()
    d_rule = ImageDraw.Draw(rule_img)
    for s, e in rule_intervals:
        d_rule.line([(s, 0), (s, H)], fill=START_COLOR, width=CUT_LINE_WIDTH)
        d_rule.line([(e, 0), (e, H)], fill=END_COLOR, width=CUT_LINE_WIDTH)
    canvas.paste(rule_img, (0, current_y))
    current_y += H

    # 紫色分隔线 2
    draw.line([(0, current_y), (W, current_y)], fill=SEP_LINE_COLOR, width=sep)
    current_y += sep

    if has_merged:
        # =====================================
        # 第3行：原图 + 规则切割线（后处理合并后）
        # =====================================
        rule_merge_img = base_img.copy()
        d_rule_merge = ImageDraw.Draw(rule_merge_img)
        for s, e in rule_intervals_merged:
            d_rule_merge.line([(s, 0), (s, H)], fill=START_COLOR, width=CUT_LINE_WIDTH)
            d_rule_merge.line([(e, 0), (e, H)], fill=END_COLOR, width=CUT_LINE_WIDTH)
        canvas.paste(rule_merge_img, (0, current_y))
        current_y += H

        # 紫色分隔线 3
        draw.line([(0, current_y), (W, current_y)], fill=SEP_LINE_COLOR, width=sep)
        current_y += sep

    # =====================================
    # 第4行（无合并时第3行）：原图 + 模型切割线
    # =====================================
    model_img = base_img.copy()
    d_model = ImageDraw.Draw(model_img)
    for s, e in model_intervals:
        d_model.line([(s, 0), (s, H)], fill=START_COLOR, width=CUT_LINE_WIDTH)
        d_model.line([(e, 0), (e, H)], fill=END_COLOR, width=CUT_LINE_WIDTH)
    canvas.paste(model_img, (0, current_y))
    current_y += H

    # 紫色分隔线（最后一条）
    draw.line([(0, current_y), (W, current_y)], fill=SEP_LINE_COLOR, width=sep)

    # =====================================
    # 最后行：概率热力图
    # =====================================
    prob_map, bar_heights = compute_prob_map(pred_prob, W, scale)

    for col in range(W):
        bar_height = bar_heights[col]
        if bar_height > 0:
            y_start = total_height - bar_height
            draw.line([(col, y_start), (col, total_height)], fill=(255, 255, 0), width=1)

    # 概率图顶部的红色参考线
    draw.line([(0, current_y + sep), (W, current_y + sep)], fill=(255, 0, 0), width=1)

    # 保存
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)
    print(f"[INFO] 对比可视化已保存：{save_path}")


@click.command("compare")
@click.argument("line_id", type=str)
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录")
@click.option("--model-path", type=click.Path(exists=True), default=None,
              help="模型权重路径")
@click.option("--image-path", type=click.Path(exists=True), default=None,
              help="直接指定行图像路径")
@click.option("--rule-json-path", type=click.Path(exists=True), default=None,
              help="直接指定规则 JSON 路径")
@click.option("--save-dir", type=str, default=None,
              help="可视化结果保存目录")
@click.option("--max-gap", type=int, default=2, show_default=True,
              help="模型合并间隙（特征像素，设为 -1 禁用合并）")
@click.option("--merge-min-gap", type=int, default=DEFAULT_MERGE_MIN_GAP,
              show_default=True, help="规则后处理合并：最小间隙")
@click.option("--merge-single-ratio", type=float, default=DEFAULT_MERGE_SINGLE_RATIO,
              show_default=True, help="规则后处理合并：单字符宽高比上限")
@click.option("--merge-min-ratio", type=float, default=DEFAULT_MERGE_MIN_RATIO,
              show_default=True, help="规则后处理合并：合并后宽高比下限")
@click.option("--merge-max-ratio", type=float, default=DEFAULT_MERGE_MAX_RATIO,
              show_default=True, help="规则后处理合并：合并后宽高比上限")
def cli(line_id, data_base_path, model_path, image_path, rule_json_path, save_dir,
        max_gap, merge_min_gap, merge_single_ratio, merge_min_ratio, merge_max_ratio):
    """
    对比规则与模型的切割结果

    生成多层对比可视化图：
    第1行：原始行图像
    第2行：规则切割结果（原始间隔）
    第3行：规则切割结果（后处理合并后）
    第4行：模型预测结果
    第5行：模型概率热力图

    规则后处理合并条件（与 segment_manager 一致）：
    1. 两个字符间隙 < --merge-min-gap 像素
    2. 两个字符都窄（width/height < --merge-single-ratio）
    3. 合并后宽高比在 [--merge-min-ratio, --merge-max-ratio] 范围
    """
    # 确定文件路径
    if image_path:
        line_path = Path(image_path)
    else:
        data_path = Path(data_base_path) if data_base_path else BASE_DIR / "datahome"
        line_path = data_path / "lines" / f"{line_id}.png"

    if not line_path.exists():
        click.echo(f"[ERROR] 行图像不存在: {line_path}", err=True)
        sys.exit(1)

    if rule_json_path:
        rule_json = Path(rule_json_path)
    else:
        data_path = Path(data_base_path) if data_base_path else BASE_DIR / "datahome"
        rule_json = data_path / "rule_jsons" / f"{line_id}_rule.json"

    rule_intervals = []
    rule_intervals_merged = []
    if rule_json.exists():
        rule_intervals = load_rule_intervals(rule_json)
        click.echo(f"[INFO] 规则切割区间（合并前）: {len(rule_intervals)} 个字符")

        # 读取行图像（用于合并计算和后续模型推理共用）
        img = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            click.echo(f"[ERROR] 无法读取图像: {line_path}", err=True)
            sys.exit(1)

        # 计算后处理合并后的区间
        rule_intervals_merged = merge_rule_intervals(
            rule_intervals,
            image_height=img.shape[0],
            min_gap=merge_min_gap,
            single_ratio=merge_single_ratio,
            min_ratio=merge_min_ratio,
            max_ratio=merge_max_ratio
        )
        click.echo(f"[INFO] 规则切割区间（合并后）: {len(rule_intervals_merged)} 个字符"
                   f"（合并了 {len(rule_intervals) - len(rule_intervals_merged)} 对）")
    else:
        click.echo(f"[WARN] 规则 JSON 不存在: {rule_json}")
        # 没有规则JSON时仍需读取图像用于模型推理
        img = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            click.echo(f"[ERROR] 无法读取图像: {line_path}", err=True)
            sys.exit(1)

    # 加载模型
    model_file = Path(model_path) if model_path else BASE_DIR / "ai_model" / "models" / "char_segment_1d_unet_best.pth"
    if not model_file.exists():
        click.echo(f"[ERROR] 模型文件不存在: {model_file}", err=True)
        sys.exit(1)

    click.echo(f"[INFO] 加载模型: {model_file}")
    predictor = CharSegmentPredictor(model_file)
    predictor.max_gap = max_gap

    model_intervals, pred_prob, scale = predictor.predict(img)
    click.echo(f"[INFO] 模型预测区间: {len(model_intervals)} 个字符")

    save_dir_path = Path(save_dir) if save_dir else line_path.parent.parent / "visualization"
    save_path = save_dir_path / f"{line_path.stem}_comparison.png"
    draw_comparison(img, rule_intervals, model_intervals, pred_prob, scale, save_path,
                    rule_intervals_merged=rule_intervals_merged)

    click.echo(f"\n{'='*50}")
    click.echo("切割对比统计")
    click.echo(f"{'='*50}")
    click.echo(f"  规则切割（合并前）: {len(rule_intervals)} 字符")
    if rule_intervals_merged:
        click.echo(f"  规则切割（合并后）: {len(rule_intervals_merged)} 字符"
                   f"（合并 {len(rule_intervals) - len(rule_intervals_merged)} 对）")
    click.echo(f"  模型预测: {len(model_intervals)} 字符")
    click.echo(f"  概率平均值: {np.mean(pred_prob):.4f}")


if __name__ == "__main__":
    cli()
