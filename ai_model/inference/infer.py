import math
import numpy as np
import cv2
import torch
import click
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageDraw
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from ai_model.models.unet1d import UNet1D
from ai_model.data.dataset import FeatureExtractor


class SequenceDecoder:
    @staticmethod
    def decode(pred_class: np.ndarray, pred_prob: Optional[np.ndarray] = None, boundary_threshold: float = 0.3) -> List[Tuple[int, int]]:
        """
        从3类序列预测结果解码字符区间（支持共享边界）
        
        标签定义：
            0 = 空白（字符外部）
            1 = 边界（字符起始/结束，支持共享边界）
            2 = 字符内部
        
        解码逻辑（支持共享边界）：
            1. 首先基于内部类(2)区域确定字符核心
            2. 每个内部类区域向左扩展到最近的边界点(1)或0
            3. 每个内部类区域向右扩展到最近的边界点(1)或图像宽度-1
            4. 处理没有内部类的边界点对（宽度1或2的字符）
            5. 支持共享边界：位置N同时是上一个字符的结束和下一个字符的开始
            6. 后处理：过滤无效区间、合并重叠区间、去除重复边界
            7. 概率加权去重：在密集边界区域，保留概率最高的边界点
        
        示例：
            正常字符：[1,2,2,1] → 内部区域[1-2] → 扩展到边界(0,3)
            共享边界：字符A[1,2,1] + 字符B[1,2,1] → 内部区域[1], [3] → 扩展到(0,2), (2,4)
            宽度1字符：[1,0,1,2,2,1] → 内部区域[3-4] → 扩展到(2,5)，单独边界点(0) → (0,0)
        
        Args:
            pred_class: 预测类别数组（0/1/2），长度=图像宽度
            pred_prob: softmax后概率数组（3通道：[空白概率, 边界概率, 内部概率]），可选，用于概率加权去重
            boundary_threshold: 边界置信度阈值，低于此阈值的边界点将被过滤
        
        Returns:
            intervals: [(start_col, end_col), ...] 字符区间列表
        """
        boundary_cols = np.where(pred_class == 1)[0]
        internal_cols = np.where(pred_class == 2)[0]
        
        if pred_prob is not None and pred_prob.shape[0] >= 2:
            boundary_probs = pred_prob[1, :]
            boundary_cols = SequenceDecoder._dedupe_boundaries(boundary_cols, boundary_probs, threshold=boundary_threshold)
        else:
            boundary_cols = SequenceDecoder._dedupe_boundaries(boundary_cols)
        
        if len(boundary_cols) == 0:
            return SequenceDecoder._decode_from_internal(pred_class)
        
        intervals = []
        
        if len(internal_cols) > 0:
            internal_regions = []
            start = internal_cols[0]
            prev = internal_cols[0]
            for col in internal_cols[1:]:
                if col == prev + 1:
                    prev = col
                else:
                    internal_regions.append((start, prev))
                    start = col
                    prev = col
            internal_regions.append((start, prev))
            
            n_regions = len(internal_regions)
            for i, (region_start, region_end) in enumerate(internal_regions):
                left_boundary = boundary_cols[boundary_cols <= region_start]
                start_col = left_boundary[-1] if len(left_boundary) > 0 else region_start
                
                right_boundary = boundary_cols[boundary_cols >= region_end]
                end_col = right_boundary[0] if len(right_boundary) > 0 else region_end
                
                if i > 0:
                    prev_region_end = internal_regions[i-1][1]
                    gap_start = prev_region_end + 1
                    gap_end = region_start - 1
                    if gap_start <= gap_end:
                        has_blank = np.any(pred_class[gap_start:gap_end+1] == 0)
                        if has_blank:
                            start_col = max(start_col, gap_end + 1)
                        else:
                            prev_end_boundary = boundary_cols[boundary_cols >= prev_region_end]
                            if len(prev_end_boundary) > 0:
                                start_col = prev_end_boundary[0]
                
                if i < n_regions - 1:
                    next_region_start = internal_regions[i+1][0]
                    gap_start = region_end + 1
                    gap_end = next_region_start - 1
                    if gap_start <= gap_end:
                        has_blank = np.any(pred_class[gap_start:gap_end+1] == 0)
                        if has_blank:
                            end_col = min(end_col, gap_start - 1)
                
                intervals.append((start_col, end_col))
        
        used_boundaries = set()
        for start, end in intervals:
            used_boundaries.add(start)
            used_boundaries.add(end)
        
        n = len(boundary_cols)
        i = 0
        while i < n:
            if boundary_cols[i] in used_boundaries:
                i += 1
                continue
            
            if i + 1 < n and boundary_cols[i + 1] not in used_boundaries:
                start = boundary_cols[i]
                end = boundary_cols[i + 1]
                
                if end - start == 1:
                    intervals.append((start, end))
                    used_boundaries.add(start)
                    used_boundaries.add(end)
                    i += 2
                else:
                    gap_has_blank = False
                    for col in range(start + 1, end):
                        if pred_class[col] == 0:
                            gap_has_blank = True
                            break
                    
                    if not gap_has_blank:
                        intervals.append((start, end))
                        used_boundaries.add(start)
                        used_boundaries.add(end)
                        i += 2
                    else:
                        intervals.append((start, start))
                        used_boundaries.add(start)
                        i += 1
            else:
                intervals.append((boundary_cols[i], boundary_cols[i]))
                used_boundaries.add(boundary_cols[i])
                i += 1
        
        intervals.sort(key=lambda x: x[0])
        
        return SequenceDecoder._clean_intervals(intervals)
    
    @staticmethod
    def _dedupe_boundaries(boundary_cols: np.ndarray, boundary_probs: Optional[np.ndarray] = None, 
                          min_dist: int = 1, threshold: float = 0.3) -> np.ndarray:
        """
        概率加权去除重复的边界点
        
        当模型在相邻列预测多个边界(1)时，根据概率加权保留最可靠的边界点。
        相邻边界点(间距=1)是合法的共享边界，表示两个字符之间的分界，不应被过滤。
        只有完全相同的位置(间距=0)才需要去重。
        
        处理流程：
            1. 过滤概率低于阈值的边界点（如果提供了概率）
            2. 按概率降序排序
            3. 贪心选择：保留概率最高的边界点，保证间距 >= min_dist(默认1)
        
        Args:
            boundary_cols: 边界点列索引数组
            boundary_probs: 边界点概率数组，长度与 boundary_cols 对应，可选
            min_dist: 边界点之间的最小距离，默认1(允许相邻边界点)
            threshold: 边界置信度阈值，低于此阈值的边界点将被过滤
        
        Returns:
            deduplicated_cols: 去重后的边界点数组（已排序）
        """
        if len(boundary_cols) == 0:
            return boundary_cols
        
        filtered_cols = boundary_cols.copy()
        filtered_probs = None
        
        if boundary_probs is not None:
            filtered_probs = boundary_probs[boundary_cols]
            mask = filtered_probs >= threshold
            filtered_cols = filtered_cols[mask]
            filtered_probs = filtered_probs[mask]
        
        if len(filtered_cols) == 0:
            return np.array([])
        
        if filtered_probs is not None:
            sorted_idx = np.argsort(filtered_probs)[::-1]
            sorted_cols = filtered_cols[sorted_idx]
            
            result = []
            for col in sorted_cols:
                if all(abs(col - r) >= min_dist for r in result):
                    result.append(col)
            
            return np.array(sorted(result))
        else:
            result = [filtered_cols[0]]
            for col in filtered_cols[1:]:
                if col - result[-1] >= min_dist:
                    result.append(col)
            return np.array(result)
    
    @staticmethod
    def _clean_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        后处理：清理区间列表
        
        处理步骤：
            1. 过滤无效区间（start >= end）
            2. 合并重叠的区间（start <= current_end）
            3. 保留相邻区间（start == current_end + 1）作为独立字符，这是合法的共享边界
        
        注意：相邻区间不合并，因为相邻边界点表示两个字符之间的共享边界。
        例如：区间(534, 619)和(620, 661)表示两个独立字符，不应合并。
        
        Args:
            intervals: 原始区间列表
        
        Returns:
            cleaned_intervals: 清理后的区间列表
        """
        if len(intervals) == 0:
            return []
        
        intervals.sort(key=lambda x: x[0])
        
        cleaned = []
        current_start, current_end = intervals[0]
        
        for start, end in intervals[1:]:
            if start > current_end + 1:
                if current_end >= current_start:
                    cleaned.append((current_start, current_end))
                current_start, current_end = start, end
            elif start == current_end + 1 or start == current_end:
                cleaned.append((current_start, current_end))
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        
        if current_end >= current_start:
            cleaned.append((current_start, current_end))
        
        return cleaned
    
    @staticmethod
    def _filter_small_intervals(intervals: List[Tuple[int, int]], min_width: int = 2, 
                                global_char_width: float = 0) -> List[Tuple[int, int]]:
        """
        过滤宽度过小的区间
        
        模型可能预测出宽度为0或1的噪声区间，这些区间应该被过滤掉。
        如果提供了全局字符宽度，则使用动态阈值（全局字符宽度的20%）。
        
        Args:
            intervals: 区间列表
            min_width: 最小宽度阈值（当global_char_width为0时使用）
            global_char_width: 全局字符宽度，用于计算动态阈值
        
        Returns:
            filtered_intervals: 过滤后的区间列表
        """
        if len(intervals) == 0:
            return []
        
        effective_min_width = min_width
        if global_char_width > 0:
            effective_min_width = max(min_width, int(global_char_width * 0.2))
        
        filtered = []
        for start, end in intervals:
            width = end - start
            if width >= effective_min_width:
                filtered.append((start, end))
        
        return filtered
    
    @staticmethod
    def _decode_from_internal(pred_class: np.ndarray) -> List[Tuple[int, int]]:
        """
        当没有边界点时，从内部区域解码
        
        Args:
            pred_class: 预测类别数组（0/1/2）
        
        Returns:
            intervals: [(start_col, end_col), ...] 字符区间列表
        """
        internal_cols = np.where(pred_class == 2)[0]
        
        if len(internal_cols) == 0:
            return []
        
        intervals = []
        start = internal_cols[0]
        prev = internal_cols[0]
        
        for col in internal_cols[1:]:
            if col == prev + 1:
                prev = col
            else:
                intervals.append((start, prev))
                start = col
                prev = col
        
        intervals.append((start, prev))
        
        return intervals


class CharSegmentPredictor:
    def __init__(self, model_path: Path, device: str = 'auto', global_char_width: float = 0):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = UNet1D(n_channels=6, n_classes=3).to(self.device)
        
        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=True)
        model_state_dict = self.model.state_dict()
        filtered_checkpoint = {}
        
        for key in checkpoint:
            if key in model_state_dict and checkpoint[key].shape == model_state_dict[key].shape:
                filtered_checkpoint[key] = checkpoint[key]
        
        self.model.load_state_dict(filtered_checkpoint, strict=False)
        
        if len(filtered_checkpoint) < len(checkpoint):
            print(f"[WARNING] 跳过 {len(checkpoint) - len(filtered_checkpoint)} 个不匹配的参数（输出层）")
        
        self.model.eval()
        
        self.max_width = 2048
        
        self.prob_height = 51
        self.prob_max_pixel = 50
        self.threshold = 0.5
        self.global_char_width = global_char_width
    
    def predict(self, line_img: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, float]:
        """
        预测行图像的字符区间
        
        Args:
            line_img: H × W 灰度图像 (0-255)
        
        Returns:
            intervals_orig: [(start_col, end_col), ...] - 原始图像坐标
            pred_prob: softmax后概率数组（缩放后尺寸，3通道），用于可视化
            pred_logits: 原始logits数组（缩放后尺寸），用于可视化
            scale: 缩放比例
        """
        original_w = line_img.shape[1]
        features, resized_w, scale = FeatureExtractor.extract(line_img)
        
        width = resized_w
        if width < self.max_width:
            pad_width = self.max_width - width
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        
        features = features.transpose(1, 0)
        features = np.expand_dims(features, axis=0)
        
        features_tensor = torch.from_numpy(features.astype(np.float32)).to(self.device)
        
        with torch.no_grad():
            output = self.model(features_tensor)
            pred_logits = output.squeeze().cpu().numpy()[:, :width]
            pred_prob = torch.softmax(output, dim=1).squeeze().cpu().numpy()[:, :width]
        
        pred_class = np.argmax(pred_prob, axis=0)
        
        intervals = SequenceDecoder.decode(pred_class, pred_prob)
        
        intervals = SequenceDecoder._filter_small_intervals(intervals, min_width=2, 
                                                           global_char_width=self.global_char_width)
        
        inv_scale = 1.0 / scale if scale > 0 else 1.0
        intervals_orig = [
            (int(round(start * inv_scale)), int(round(end * inv_scale)))
            for start, end in intervals
        ]
        
        return intervals_orig, pred_prob, pred_logits, scale
    
    def predict_from_path(self, line_path: Path) -> Optional[Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, float]]:
        """
        从文件路径预测
        
        Args:
            line_path: 行图像路径
        
        Returns:
            (intervals, pred_prob, pred_logits, scale) 或 None
        """
        if not line_path.exists():
            return None
        
        img = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        return self.predict(img)
    
    def draw_boundaries_with_prob(self, line_img: np.ndarray, intervals: List[Tuple[int, int]], 
                                   pred_prob: np.ndarray, scale: float, save_path: Path):
        """
        绘制字符边界 + 概率可视化（使用softmax后的真实概率值，无阈值截断）
        
        Args:
            line_img: 原始灰度图像 (H × W)
            intervals: 字符区间列表 [(start_col, end_col), ...]
            pred_prob: softmax后的概率数组（3通道：[空白概率, 边界概率, 内部概率]）
            scale: 缩放比例
            save_path: 保存路径
        """
        H, W = line_img.shape
        
        img_pil = Image.fromarray(line_img).convert("RGB")
        
        new_img = Image.new("RGB", (W, H + self.prob_height), color=(255, 255, 255))
        new_img.paste(img_pil, (0, 0))
        
        draw = ImageDraw.Draw(new_img)
        
        for x_start, x_end in intervals:
            draw.line([(x_start, 0), (x_start, H)], fill=(255, 0, 0), width=1)
            draw.line([(x_end, 0), (x_end, H)], fill=(0, 255, 0), width=1)
        
        resized_w = pred_prob.shape[1]
        inv_scale = 1.0 / scale if scale > 0 else 1.0
        
        for resized_col in range(resized_w):
            boundary_prob = pred_prob[1, resized_col]
            internal_prob = pred_prob[2, resized_col]
            
            orig_col = int(round(resized_col * inv_scale))
            
            if 0 <= orig_col < W:
                char_prob = max(boundary_prob, internal_prob)
                bar_height = max(1, int(round(char_prob * self.prob_max_pixel)))
                
                y_start = H + self.prob_height - bar_height
                y_end = H + self.prob_height
                
                r = int(255 * char_prob)
                g = int(255 * internal_prob)
                b = int(255 * (1 - char_prob))
                
                draw.line([(orig_col, y_start), (orig_col, y_end)], fill=(r, g, b), width=1)
        
        draw.line([(0, H), (W, H)], fill=(255, 0, 0), width=1)
        
        new_img.save(save_path)
        print(f"✅ 带概率可视化的结果已保存：{save_path}")
    
    def infer_whole_line(self, line_img_path: Path, save_path: Path = None) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        完整推理流程：预测 + 可视化
        
        Args:
            line_img_path: 行图像路径
            save_path: 保存路径（可选）
        
        Returns:
            (intervals, pred_prob)
        """
        result = self.predict_from_path(line_img_path)
        
        if result is None:
            print(f"[ERROR] 无法读取图像: {line_img_path}")
            return [], np.array([])
        
        intervals, pred_prob, pred_logits, scale = result
        
        if save_path is not None:
            img = cv2.imread(str(line_img_path), cv2.IMREAD_GRAYSCALE)
            self.draw_boundaries_with_prob(img, intervals, pred_prob, scale, save_path)
        
        return intervals, pred_prob


@click.command("predict")
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--model-path", type=click.Path(exists=True), default=None, help="模型权重路径")
@click.option("--output", type=str, default=None, help="可视化结果保存路径")
@click.option("--threshold", type=float, default=0.5, show_default=True, help="字符概率阈值")
@click.option("--max-gap", type=int, default=2, show_default=True, help="合并间隙（特征像素）")
def cli(image_path, model_path, output, threshold, max_gap):
    """
    对单行图像进行字符分割推理

    输入一行文本图像，输出每个字符的起止列坐标和可视化结果。
    """
    import numpy as np

    base_dir = Path(__file__).resolve().parent.parent / "models"
    model_file = Path(model_path) if model_path else base_dir / "char_segment_1d_unet_best.pth"

    if not model_file.exists():
        click.echo(f"[ERROR] 模型文件不存在: {model_file}", err=True)
        sys.exit(1)

    click.echo(f"[INFO] 加载模型: {model_file}")
    predictor = CharSegmentPredictor(model_file)
    predictor.threshold = threshold
    predictor.max_gap = max_gap

    line_path = Path(image_path)
    intervals, pred_prob = predictor.infer_whole_line(
        line_path,
        save_path=Path(output) if output else None
    )

    click.echo(f"[INFO] 识别到 {len(intervals)} 个字符")
    click.echo(f"[INFO] 字符区间: {intervals}")

    if len(pred_prob) > 0:
        click.echo(f"\n[INFO] 概率统计:")
        click.echo(f"  空白类平均概率: {np.mean(pred_prob[0]):.4f}")
        click.echo(f"  边界类平均概率: {np.mean(pred_prob[1]):.4f}")
        click.echo(f"  内部类平均概率: {np.mean(pred_prob[2]):.4f}")


if __name__ == "__main__":
    cli()