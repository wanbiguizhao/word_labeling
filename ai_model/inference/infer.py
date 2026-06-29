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
from ai_model.data.dataset import FeatureExtractor, IntervalExtractor


class CharSegmentPredictor:
    def __init__(self, model_path: Path, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = UNet1D(n_channels=6, n_classes=1).to(self.device)
        self.model.load_state_dict(torch.load(str(model_path), map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.max_width = 2048
        
        self.prob_height = 51
        self.prob_max_pixel = 50
        self.threshold = 0.5
        self.max_gap = 2
    
    def predict(self, line_img: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray, float]:
        """
        预测行图像的字符区间
        
        Args:
            line_img: H × W 灰度图像 (0-255)
        
        Returns:
            intervals_orig: [(start_col, end_col), ...] - 原始图像坐标
            pred_prob: 预测概率数组（缩放后尺寸）
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
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()[:width]
        
        intervals = IntervalExtractor.extract(pred_prob, self.threshold, self.max_gap)
        
        inv_scale = 1.0 / scale if scale > 0 else 1.0
        intervals_orig = [
            (int(round(start * inv_scale)), int(round(end * inv_scale)))
            for start, end in intervals
        ]
        
        return intervals_orig, pred_prob, scale
    
    def predict_from_path(self, line_path: Path) -> Optional[Tuple[List[Tuple[int, int]], np.ndarray, float]]:
        """
        从文件路径预测
        
        Args:
            line_path: 行图像路径
        
        Returns:
            (intervals, pred_prob, scale) 或 None
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
        绘制字符边界 + 概率可视化
        
        Args:
            line_img: 原始灰度图像 (H × W)
            intervals: 字符区间列表 [(start_col, end_col), ...]
            pred_prob: 预测概率数组
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
        
        resized_w = len(pred_prob)
        inv_scale = 1.0 / scale if scale > 0 else 1.0
        
        for resized_col in range(resized_w):
            prob = pred_prob[resized_col]
            orig_col = int(round(resized_col * inv_scale))
            
            if 0 <= orig_col < W:
                prob_percent = prob * 100
                bar_height = math.ceil(prob_percent * 0.5)
                
                if bar_height > 0:
                    y_start = H + self.prob_height - bar_height
                    y_end = H + self.prob_height
                    draw.line([(orig_col, y_start), (orig_col, y_end)], fill=(255, 255, 0), width=1)
        
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
        
        intervals, pred_prob, scale = result
        
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
        click.echo(f"  平均概率: {np.mean(pred_prob):.4f}")
        click.echo(f"  最大概率: {np.max(pred_prob):.4f}")
        click.echo(f"  最小概率: {np.min(pred_prob):.4f}")


if __name__ == "__main__":
    cli()