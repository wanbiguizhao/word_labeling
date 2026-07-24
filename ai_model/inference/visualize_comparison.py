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
from ai_model.common.cut_line_converter import (
    intervals_to_positions,
    START_COLOR_RGB,
    END_COLOR_RGB,
    SHARED_COLOR_RGB
)


# ===================== 可视化配置 =====================
SEP_LINE_COLOR = (128, 0, 128)    # 紫色分隔线
SEP_LINE_WIDTH = 2                 # 分隔线宽度
CUT_LINE_WIDTH = 1                 # 切割线宽度
START_COLOR = START_COLOR_RGB      # 红色 = 字符起点
END_COLOR = END_COLOR_RGB          # 绿色 = 字符终点
SHARED_COLOR = SHARED_COLOR_RGB    # 紫色 = 共享边界
PROB_HEIGHT = 51                   # 概率图高度（像素）
PROB_MAX_PIXEL = 50                # 概率条最大高度





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


def load_lineage_intervals(lineage_path: Path, line_id: str) -> List[Tuple[int, int]]:
    """
    从 lineage.json 中读取后处理合并后的字符区间

    Args:
        lineage_path: lineage.json 文件路径
        line_id: 行ID

    Returns:
        [(start, end), ...] 字符区间列表（已合并后的数据）
    """
    if not lineage_path.exists():
        return []

    with open(lineage_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lines = data.get('lines', {})
    chars = data.get('chars', {})

    line_info = lines.get(line_id)
    if not line_info:
        return []

    char_ids = line_info.get('chars', [])
    intervals = []
    for cid in char_ids:
        char_info = chars.get(cid)
        if char_info:
            start = char_info.get('col_start', 0)
            end = char_info.get('col_end', 0)
            if end > start:
                intervals.append((start, end))

    return intervals


def compute_prob_map(pred_prob: np.ndarray, orig_width: int, scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将模型概率映射回原始图像宽度（用于可视化，无阈值截断）

    Args:
        pred_prob: softmax后的概率数组（3通道：[空白概率, 边界概率, 内部概率]）
        orig_width: 原始图像宽度
        scale: 缩放比例

    Returns:
        (max_prob_map, class_map, bar_heights, blank_prob_map) 
            最大概率图、类别图、条高度、空白概率图
            class_map: 0=空白, 1=边界, 2=内部
    """
    inv_scale = 1.0 / scale if scale > 0 else 1.0
    n_channels, resized_w = pred_prob.shape

    max_prob_map = np.zeros(orig_width, dtype=np.float32)
    class_map = np.zeros(orig_width, dtype=np.int32)
    count_map = np.zeros(orig_width, dtype=np.int32)

    for resized_col in range(resized_w):
        max_prob = np.max(pred_prob[:, resized_col])
        max_class = np.argmax(pred_prob[:, resized_col])
        
        orig_col = int(round(resized_col * inv_scale))
        orig_col = min(max(orig_col, 0), orig_width - 1)
        
        max_prob_map[orig_col] += max_prob
        class_map[orig_col] = max_class
        count_map[orig_col] += 1

    mask = count_map > 0
    max_prob_map[mask] /= count_map[mask]

    bar_heights = np.maximum(1, np.round(max_prob_map * PROB_MAX_PIXEL)).astype(np.int32)

    return max_prob_map, class_map, bar_heights


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
    
    rule_starts, rule_ends, rule_shared = intervals_to_positions(rule_intervals)
    for s in rule_starts:
        d_rule.line([(s, 0), (s, H)], fill=START_COLOR, width=CUT_LINE_WIDTH)
    for e in rule_ends:
        d_rule.line([(e, 0), (e, H)], fill=END_COLOR, width=CUT_LINE_WIDTH)
    for s in rule_shared:
        d_rule.line([(s, 0), (s, H)], fill=SHARED_COLOR, width=CUT_LINE_WIDTH)
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
        
        rm_starts, rm_ends, rm_shared = intervals_to_positions(rule_intervals_merged)
        for s in rm_starts:
            d_rule_merge.line([(s, 0), (s, H)], fill=START_COLOR, width=CUT_LINE_WIDTH)
        for e in rm_ends:
            d_rule_merge.line([(e, 0), (e, H)], fill=END_COLOR, width=CUT_LINE_WIDTH)
        for s in rm_shared:
            d_rule_merge.line([(s, 0), (s, H)], fill=SHARED_COLOR, width=CUT_LINE_WIDTH)
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
    
    model_starts, model_ends, model_shared = intervals_to_positions(model_intervals)
    for s in model_starts:
        d_model.line([(s, 0), (s, H)], fill=START_COLOR, width=CUT_LINE_WIDTH)
    for e in model_ends:
        d_model.line([(e, 0), (e, H)], fill=END_COLOR, width=CUT_LINE_WIDTH)
    for s in model_shared:
        d_model.line([(s, 0), (s, H)], fill=SHARED_COLOR, width=CUT_LINE_WIDTH)
    canvas.paste(model_img, (0, current_y))
    current_y += H

    # 紫色分隔线（最后一条）
    draw.line([(0, current_y), (W, current_y)], fill=SEP_LINE_COLOR, width=sep)

    # =====================================
    # 最后行：概率热力图（使用softmax后的真实概率值，无阈值截断）
    # 颜色映射：内部=红色, 边界=黄色, 空白=蓝色, 高度=概率大小
    # =====================================
    max_prob_map, class_map, bar_heights = compute_prob_map(pred_prob, W, scale)

    for col in range(W):
        max_prob = max_prob_map[col]
        pred_class = class_map[col]
        bar_height = bar_heights[col]
        y_start = total_height - bar_height
        y_end = total_height
        
        if pred_class == 2:
            r = int(255 * max_prob)
            g = 0
            b = 0
        elif pred_class == 1:
            r = int(255 * max_prob)
            g = int(255 * max_prob)
            b = 0
        else:
            r = 0
            g = 0
            b = int(255 * max_prob)
        
        draw.line([(col, y_start), (col, y_end)], fill=(r, g, b), width=1)

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
@click.option("--threshold", type=float, default=0.3, show_default=True,
              help="模型预测概率阈值")
def cli(line_id, data_base_path, model_path, image_path, rule_json_path, save_dir,
        max_gap, threshold):
    """
    对比规则与模型的切割结果

    生成多层对比可视化图：
    第1行：原始行图像
    第2行：规则切割结果（原始间隔）
    第3行：规则切割结果（后处理合并后，从 lineage.json 读取）
    第4行：模型预测结果
    第5行：模型概率热力图

    规则后处理合并数据直接从 lineage.json 读取，不自行进行后处理操作。
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
    else:
        click.echo(f"[WARN] 规则 JSON 不存在: {rule_json}")

    # 从 lineage.json 读取后处理合并后的区间
    data_path = Path(data_base_path) if data_base_path else BASE_DIR / "datahome"
    lineage_path = data_path / "lineage.json"
    rule_intervals_merged = load_lineage_intervals(lineage_path, line_id)
    if rule_intervals_merged:
        click.echo(f"[INFO] 规则切割区间（合并后，来自 lineage.json）: {len(rule_intervals_merged)} 个字符")
    else:
        click.echo(f"[WARN] lineage.json 中未找到 {line_id} 的后处理数据")

    # 读取行图像（用于模型推理）
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
    predictor.threshold = threshold

    model_intervals, pred_prob, pred_logits, scale = predictor.predict(img)
    click.echo(f"[INFO] 模型预测区间: {len(model_intervals)} 个字符")
    
    click.echo(f"[DEBUG] 概率统计:")
    click.echo(f"  空白类平均概率: {np.mean(pred_prob[0]):.4f}")
    click.echo(f"  边界类平均概率: {np.mean(pred_prob[1]):.4f}")
    click.echo(f"  内部类平均概率: {np.mean(pred_prob[2]):.4f}")
    click.echo(f"  缩放比例: {scale:.4f}")
    click.echo(f"  原始宽度: {img.shape[1]}, 缩放后宽度: {pred_prob.shape[1]}")

    save_dir_path = Path(save_dir) if save_dir else line_path.parent.parent / "visualization"
    save_path = save_dir_path / f"{line_path.stem}_comparison.png"
    draw_comparison(img, rule_intervals, model_intervals, pred_prob, scale, save_path,
                    rule_intervals_merged=rule_intervals_merged)

    click.echo(f"\n{'='*50}")
    click.echo("切割对比统计")
    click.echo(f"{'='*50}")
    click.echo(f"  规则切割（原始）: {len(rule_intervals)} 字符")
    if rule_intervals_merged:
        click.echo(f"  规则切割（后处理，来自 lineage.json）: {len(rule_intervals_merged)} 字符")
        if rule_intervals:
            click.echo(f"  （较原始减少 {len(rule_intervals) - len(rule_intervals_merged)} 个）")
    click.echo(f"  模型预测: {len(model_intervals)} 字符")
    click.echo(f"  字符类平均概率: {np.mean(pred_prob[1:]):.4f}")


if __name__ == "__main__":
    cli()
