import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset


def resize_image(line_img: np.ndarray, target_height: int = 64) -> np.ndarray:
    h, w = line_img.shape
    
    if h == 0:
        return line_img
    
    if h > target_height:
        scale = target_height / h
        new_h = target_height
        new_w = int(w * scale)
        resized_img = cv2.resize(line_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
        new_h = h
        new_w = w
        resized_img = line_img.copy()
    
    canvas = np.ones((target_height, new_w), dtype=np.uint8) * 255
    
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset + new_h, :new_w] = resized_img
    
    return canvas


class FeatureExtractor:
    @staticmethod
    def extract(line_img: np.ndarray) -> Tuple[np.ndarray, int, float]:
        original_w = line_img.shape[1]
        line_img = resize_image(line_img)
        
        h, w = line_img.shape
        scale = w / original_w if original_w > 0 else 1.0
        
        line_img_norm = line_img / 255.0
        
        proj_mean = np.mean(line_img_norm, axis=0)
        proj_var = np.var(line_img_norm, axis=0)
        proj_nonzero = np.count_nonzero(line_img_norm > 0.1, axis=0) / h
        
        pad_img = np.pad(line_img_norm, ((0, 0), (1, 1)), mode='constant')
        local_proj = np.zeros(w)
        for i in range(w):
            local_proj[i] = np.mean(pad_img[:, i:i+3])
        
        grad_x = cv2.Sobel(line_img_norm, cv2.CV_64F, 1, 0, ksize=3)
        grad_proj = np.mean(np.abs(grad_x), axis=0)
        
        pad_img5 = np.pad(line_img_norm, ((0, 0), (2, 2)), mode='constant')
        local_proj5 = np.zeros(w)
        for i in range(w):
            local_proj5[i] = np.mean(pad_img5[:, i:i+5])
        
        features = np.stack([
            proj_mean, 
            proj_var, 
            proj_nonzero, 
            local_proj, 
            grad_proj,
            local_proj5
        ], axis=-1)
        
        return features, w, scale


# ===================== 合并参数（与 segment_config.py 保持一致）=====================
MERGE_MIN_GAP = 3              # 最小间隙像素（原始尺寸）
MERGE_MAX_HEIGHT_DIFF = 0.2    # 最大高度差比例
MERGE_SINGLE_ASPECT_RATIO = 0.7  # 单字符宽高比上限
MERGE_MIN_ASPECT_RATIO = 0.5    # 合并后宽高比下限
MERGE_MAX_ASPECT_RATIO = 1.5    # 合并后宽高比上限


class LabelGenerator:
    @staticmethod
    def generate(
        char_segments: List[Dict],
        image_width: int,
        image_height: int,
        scale: float = 1.0,
        merge_enabled: bool = True
    ) -> np.ndarray:
        """
        从字符段生成标签数组
        
        Args:
            char_segments: 字符段列表，每个包含 col_start, col_end, width, height
            image_width: 图像宽度（缩放后）
            image_height: 图像高度（缩放后）
            scale: 缩放比例
            merge_enabled: 是否启用合并逻辑（与 postprocess_merge_chars 一致）
        
        Returns:
            label: 二值标签数组，字符区域为1.0，间隙为0.0
        """
        label = np.zeros(image_width, dtype=np.float32)
        
        if not char_segments:
            return label
        
        # 提取区间并转换到缩放后的坐标
        intervals = []
        for char in char_segments:
            start = int(round(char.get('col_start', 0) * scale))
            end = int(round(char.get('col_end', 0) * scale))
            start = max(0, min(start, image_width - 1))
            end = max(0, min(end, image_width - 1))
            intervals.append((start, end))
        
        # 执行合并逻辑
        if merge_enabled and len(intervals) >= 2:
            intervals = LabelGenerator._merge_narrow_chars(
                intervals, image_height,
                min_gap=int(MERGE_MIN_GAP * scale),
                max_height_diff=MERGE_MAX_HEIGHT_DIFF,
                single_ratio=MERGE_SINGLE_ASPECT_RATIO,
                min_ratio=MERGE_MIN_ASPECT_RATIO,
                max_ratio=MERGE_MAX_ASPECT_RATIO
            )
        
        # 生成标签
        for start, end in intervals:
            if end >= start:
                label[start:end+1] = 1.0
        
        return label
    
    @staticmethod
    def _merge_narrow_chars(
        intervals: List[Tuple[int, int]],
        image_height: int,
        min_gap: int = 3,
        max_height_diff: float = 0.2,
        single_ratio: float = 0.7,
        min_ratio: float = 0.5,
        max_ratio: float = 1.5
    ) -> List[Tuple[int, int]]:
        """
        合并间隙较小的窄字符段（模拟 postprocess_merge_chars）
        
        合并条件（全部满足）：
        1. 间隙 < min_gap
        2. 两个字符高度差 < 平均高度 * max_height_diff
        3. 两个字符都窄（width/height < single_ratio）
        4. 合并后宽高比在 [min_ratio, max_ratio] 范围
        
        Args:
            intervals: [(start, end), ...] 已排序的区间列表
            image_height: 图像高度
            min_gap: 最小间隙像素数
            max_height_diff: 最大高度差比例
            single_ratio: 单字符宽高比上限
            min_ratio: 合并后宽高比下限
            max_ratio: 合并后宽高比上限
        
        Returns:
            合并后的区间列表
        """
        if len(intervals) < 2:
            return intervals
        
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            last = merged[-1]
            gap = current[0] - last[1] - 1
            
            # 条件1：间隙足够小
            if gap >= 0 and gap < min_gap:
                w1 = last[1] - last[0]
                w2 = current[1] - current[0]
                r1 = w1 / image_height
                r2 = w2 / image_height
                
                # 条件2：高度差足够小（假设所有字符高度 = image_height）
                # 条件3：两个字符都窄
                if r1 < single_ratio and r2 < single_ratio:
                    merged_w = current[1] - last[0]
                    merged_r = merged_w / image_height
                    
                    # 条件4：合并后宽高比合理
                    if min_ratio < merged_r < max_ratio:
                        merged[-1] = (last[0], current[1])
                        continue
            
            merged.append(current)
        
        return merged


class IntervalExtractor:
    @staticmethod
    def extract(pred_prob: np.ndarray, threshold: float = 0.5, max_gap: int = 2) -> List[Tuple[int, int]]:
        binary = (pred_prob > threshold).astype(np.int32)
        char_cols = np.where(binary == 1)[0]
        
        if len(char_cols) == 0:
            return []
        
        intervals = []
        start = char_cols[0]
        prev = char_cols[0]
        
        for col in char_cols[1:]:
            if col == prev + 1:
                prev = col
            else:
                intervals.append((start, prev))
                start = col
                prev = col
        
        intervals.append((start, prev))
        
        if max_gap >= 0:
            intervals = IntervalExtractor._merge_small_gaps(intervals, max_gap=max_gap)
        
        return intervals
    
    @staticmethod
    def _merge_small_gaps(intervals: List[Tuple[int, int]], max_gap: int = 2) -> List[Tuple[int, int]]:
        if len(intervals) < 2:
            return intervals
        
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            last = merged[-1]
            gap = current[0] - last[1] - 1
            
            if gap <= max_gap:
                merged[-1] = (last[0], current[1])
            else:
                merged.append(current)
        
        return merged


class CharSegmentDataset(Dataset):
    def __init__(
        self, 
        data_base_path: Path, 
        line_ids: List[str],
        char_width_stats: Optional[Dict] = None
    ):
        self.data_base_path = data_base_path
        self.line_ids = line_ids
        self.lines_dir = data_base_path / "lines"
        self.rule_jsons_dir = data_base_path / "rule_jsons"
        
        # 预加载所有 rule_json 到内存缓存，避免每个 epoch 重复读取
        self._rule_cache: Dict[str, dict] = {}
        # 预计算每行的中位字符宽度
        self._char_width_cache: Dict[str, float] = {}
        # 全局中位字符宽度
        self._global_char_width: float = 0.0
        
        self._load_rule_cache()
        
        if char_width_stats is not None:
            self._load_char_width_stats(char_width_stats)
        else:
            self._compute_char_widths()
    
    def _load_rule_cache(self):
        """批量加载所有 rule_json 到内存"""
        for line_id in self.line_ids:
            json_path = self.rule_jsons_dir / f"{line_id}_rule.json"
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self._rule_cache[line_id] = json.load(f)
            except Exception:
                pass
    
    def _load_char_width_stats(self, char_width_stats: Dict):
        """从外部传入的统计信息加载字符宽度"""
        self._global_char_width = char_width_stats.get('global_median_char_width', 16.0)
        self._char_width_cache = char_width_stats.get('line_median_char_widths', {})
    
    def _compute_char_widths(self):
        """计算每行和全局的中位字符宽度（后备方案）"""
        all_widths = []
        
        for line_id, rule_data in self._rule_cache.items():
            chars = rule_data.get('chars', [])
            if not chars:
                continue
            
            widths = []
            for char in chars:
                w = char.get('width', 0)
                if 3 <= w <= 100:
                    widths.append(w)
            
            if widths:
                median_width = float(np.median(widths))
                self._char_width_cache[line_id] = median_width
                all_widths.extend(widths)
        
        if all_widths:
            self._global_char_width = float(np.median(all_widths))
        else:
            self._global_char_width = 16.0
    
    def get_char_width(self, line_id: str) -> float:
        """获取指定行的中位字符宽度"""
        return self._char_width_cache.get(line_id, self._global_char_width)
    
    @property
    def global_char_width(self) -> float:
        """获取全局中位字符宽度"""
        return self._global_char_width
    
    def __len__(self):
        return len(self.line_ids)
    
    def __getitem__(self, idx):
        line_id = self.line_ids[idx]
        
        line_path = self.lines_dir / f"{line_id}.png"
        img = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
        
        rule_data = self._rule_cache.get(line_id)
        if rule_data is None:
            json_path = self.rule_jsons_dir / f"{line_id}_rule.json"
            with open(json_path, 'r', encoding='utf-8') as f:
                rule_data = json.load(f)
        
        features, resized_w, scale = FeatureExtractor.extract(img)
        
        # 使用合并后的标签（与 postprocess_merge_chars 一致）
        image_height = int(rule_data.get('image_height', img.shape[0]) * scale)
        label = LabelGenerator.generate(
            rule_data['chars'], 
            resized_w, 
            image_height,
            scale,
            merge_enabled=True
        )
        
        features = features.transpose(1, 0)
        
        char_width = self.get_char_width(line_id) * scale
        
        return {
            'line_id': line_id,
            'features': features.astype(np.float32),
            'label': label.astype(np.float32),
            'width': resized_w,
            'height': 64,
            'scale': scale,
            'char_width': char_width
        }


def collate_fn(batch):
    max_width = max(item['width'] for item in batch)
    batch_size = len(batch)
    
    # 预分配连续数组，避免多次 np.pad + np.array 的开销
    n_channels = batch[0]['features'].shape[0]
    features_arr = np.zeros((batch_size, n_channels, max_width), dtype=np.float32)
    labels_arr = np.zeros((batch_size, max_width), dtype=np.float32)
    line_ids = [None] * batch_size
    char_widths = np.zeros(batch_size, dtype=np.float32)
    
    for i, item in enumerate(batch):
        w = item['width']
        features_arr[i, :, :w] = item['features']
        labels_arr[i, :w] = item['label']
        line_ids[i] = item['line_id']
        char_widths[i] = item.get('char_width', 0.0)
    
    return {
        'line_ids': line_ids,
        'features': features_arr,
        'labels': labels_arr,
        'char_widths': char_widths
    }


def load_all_line_ids(data_base_path: Path) -> List[str]:
    rule_jsons_dir = data_base_path / "rule_jsons"
    json_files = sorted(rule_jsons_dir.glob("*_rule.json"))
    
    line_ids = []
    for json_file in json_files:
        line_id = json_file.stem.replace("_rule", "")
        line_path = data_base_path / "lines" / f"{line_id}.png"
        if line_path.exists():
            line_ids.append(line_id)
    
    return line_ids