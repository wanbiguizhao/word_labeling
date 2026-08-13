import numpy as np
import json
import click
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from ai_model.inference.infer import CharSegmentPredictor
from ai_model.data.dataset import load_all_line_ids, FeatureExtractor, IntervalExtractor


class ActiveLearner:
    def __init__(self, model_path: Path, data_base_path: Path):
        self.predictor = CharSegmentPredictor(model_path)
        self.data_base_path = data_base_path
        self.rule_jsons_dir = data_base_path / "rule_jsons"
    
    def load_rule_intervals(self, line_id: str) -> Optional[List[Tuple[int, int]]]:
        json_path = self.rule_jsons_dir / f"{line_id}_rule.json"
        if not json_path.exists():
            return None
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        intervals = []
        for char in data.get('chars', []):
            start = char.get('col_start', 0)
            end = char.get('col_end', 0)
            intervals.append((start, end))
        
        return intervals
    
    def uncertainty_score(self, pred_prob: np.ndarray) -> float:
        distances = np.minimum(pred_prob, 1 - pred_prob) * 2
        return float(np.mean(distances))
    
    def entropy_score(self, pred_prob: np.ndarray) -> float:
        eps = 1e-10
        entropy = -np.mean(pred_prob * np.log(pred_prob + eps) + (1 - pred_prob) * np.log(1 - pred_prob + eps))
        return float(entropy)
    
    def disagreement_score(self, rule_intervals: List[Tuple[int, int]], 
                           model_intervals: List[Tuple[int, int]], 
                           image_width: int) -> float:
        if not rule_intervals or not model_intervals:
            return 1.0
        
        rule_mask = np.zeros(image_width, dtype=np.float32)
        for start, end in rule_intervals:
            start = max(0, min(start, image_width - 1))
            end = max(0, min(end, image_width - 1))
            rule_mask[start:end+1] = 1.0
        
        model_mask = np.zeros(image_width, dtype=np.float32)
        for start, end in model_intervals:
            start = max(0, min(start, image_width - 1))
            end = max(0, min(end, image_width - 1))
            model_mask[start:end+1] = 1.0
        
        intersection = np.sum(rule_mask * model_mask)
        union = np.sum(rule_mask) + np.sum(model_mask) - intersection
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return 1.0 - float(iou)
    
    def interval_count_diff_score(self, rule_intervals: List[Tuple[int, int]], 
                                  model_intervals: List[Tuple[int, int]]) -> float:
        rule_count = len(rule_intervals)
        model_count = len(model_intervals)
        max_count = max(rule_count, model_count, 1)
        return abs(rule_count - model_count) / max_count
    
    def boundary_displacement_score(self, rule_intervals: List[Tuple[int, int]], 
                                     model_intervals: List[Tuple[int, int]]) -> float:
        if len(rule_intervals) != len(model_intervals):
            return 1.0
        
        total_displacement = 0.0
        total_width = 0.0
        
        for (r_start, r_end), (m_start, m_end) in zip(rule_intervals, model_intervals):
            total_displacement += abs(r_start - m_start) + abs(r_end - m_end)
            total_width += (r_end - r_start) + (m_end - m_start)
        
        if total_width == 0:
            return 0.0
        
        avg_width = total_width / (2 * len(rule_intervals))
        avg_displacement = total_displacement / (2 * len(rule_intervals))
        
        return min(avg_displacement / avg_width, 1.0)
    
    def compute_al_score(self, line_id: str) -> Dict:
        line_path = self.data_base_path / "lines" / f"{line_id}.png"
        if not line_path.exists():
            return None
        
        result = self.predictor.predict_from_path(line_path)
        if result is None:
            return None
        
        model_intervals, pred_prob, pred_logits, scale = result
        
        rule_intervals = self.load_rule_intervals(line_id)
        if rule_intervals is None:
            return None
        
        img_arr = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
        image_width = img_arr.shape[1]
        
        uncertainty = self.uncertainty_score(pred_prob)
        entropy = self.entropy_score(pred_prob)
        disagreement = self.disagreement_score(rule_intervals, model_intervals, image_width)
        count_diff = self.interval_count_diff_score(rule_intervals, model_intervals)
        boundary_diff = self.boundary_displacement_score(rule_intervals, model_intervals)
        
        weights = {
            'uncertainty': 0.2,
            'entropy': 0.2,
            'disagreement': 0.3,
            'count_diff': 0.15,
            'boundary_diff': 0.15
        }
        
        al_score = (
            weights['uncertainty'] * uncertainty +
            weights['entropy'] * entropy +
            weights['disagreement'] * disagreement +
            weights['count_diff'] * count_diff +
            weights['boundary_diff'] * boundary_diff
        )
        
        return {
            'line_id': line_id,
            'al_score': al_score,
            'uncertainty': uncertainty,
            'entropy': entropy,
            'disagreement': disagreement,
            'count_diff': count_diff,
            'boundary_diff': boundary_diff,
            'rule_interval_count': len(rule_intervals),
            'model_interval_count': len(model_intervals),
            'image_width': image_width
        }
    
    def rank_lines(self, top_n: int = 100) -> List[Dict]:
        line_ids = load_all_line_ids(self.data_base_path)
        print(f"[INFO] 总共有 {len(line_ids)} 个行数据")
        
        scores = []
        for i, line_id in enumerate(line_ids):
            if (i + 1) % 100 == 0:
                print(f"[INFO] 已处理 {i+1}/{len(line_ids)} 行")
            
            result = self.compute_al_score(line_id)
            if result is not None:
                scores.append(result)
        
        scores.sort(key=lambda x: x['al_score'], reverse=True)
        
        return scores[:top_n]
    
    def _extract_features_for_line(self, line_id: str) -> Optional[Dict]:
        line_path = self.data_base_path / "lines" / f"{line_id}.png"
        if not line_path.exists():
            return None
        
        rule_intervals = self.load_rule_intervals(line_id)
        if rule_intervals is None:
            return None
        
        img_arr = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
        if img_arr is None:
            return None
        
        image_width = img_arr.shape[1]
        features, resized_w, scale = FeatureExtractor.extract(img_arr)
        
        return {
            'line_id': line_id,
            'features': features,
            'resized_w': resized_w,
            'scale': scale,
            'image_width': image_width,
            'rule_intervals': rule_intervals
        }
    
    def _compute_al_score_from_result(self, pred_result: Dict) -> Dict:
        line_id = pred_result['line_id']
        model_intervals = pred_result['model_intervals']
        pred_prob = pred_result['pred_prob']
        image_width = pred_result['image_width']
        rule_intervals = pred_result['rule_intervals']
        
        uncertainty = self.uncertainty_score(pred_prob)
        entropy = self.entropy_score(pred_prob)
        disagreement = self.disagreement_score(rule_intervals, model_intervals, image_width)
        count_diff = self.interval_count_diff_score(rule_intervals, model_intervals)
        boundary_diff = self.boundary_displacement_score(rule_intervals, model_intervals)
        
        weights = {
            'uncertainty': 0.2,
            'entropy': 0.2,
            'disagreement': 0.3,
            'count_diff': 0.15,
            'boundary_diff': 0.15
        }
        
        al_score = (
            weights['uncertainty'] * uncertainty +
            weights['entropy'] * entropy +
            weights['disagreement'] * disagreement +
            weights['count_diff'] * count_diff +
            weights['boundary_diff'] * boundary_diff
        )
        
        return {
            'line_id': line_id,
            'al_score': al_score,
            'uncertainty': uncertainty,
            'entropy': entropy,
            'disagreement': disagreement,
            'count_diff': count_diff,
            'boundary_diff': boundary_diff,
            'rule_interval_count': len(rule_intervals),
            'model_interval_count': len(model_intervals),
            'image_width': image_width
        }
    
    def rank_lines_batched(self, top_n: int = 100, batch_size: int = 32, 
                           num_workers: int = 4, chunk_size: int = 1000) -> List[Dict]:
        import torch
        
        line_ids = load_all_line_ids(self.data_base_path)
        total_lines = len(line_ids)
        print(f"[INFO] 总共有 {total_lines} 个行数据")
        
        chunks = [line_ids[i:i+chunk_size] for i in range(0, total_lines, chunk_size)]
        num_chunks = len(chunks)
        print(f"[INFO] 分成 {num_chunks} 个chunk，每chunk {chunk_size} 个")
        
        device = self.predictor.device
        model = self.predictor.model
        threshold = self.predictor.threshold
        max_gap = self.predictor.max_gap
        
        global_top_scores = []
        
        for chunk_idx, chunk_line_ids in enumerate(chunks):
            print(f"\n{'='*60}")
            print(f"[INFO] 处理第 {chunk_idx + 1}/{num_chunks} 个chunk（{len(chunk_line_ids)} 行）")
            print(f"{'='*60}")
            
            line_data_list = []
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self._extract_features_for_line, line_id): line_id 
                          for line_id in chunk_line_ids}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="特征提取"):
                    result = future.result()
                    if result is not None:
                        line_data_list.append(result)
            
            if not line_data_list:
                print("[WARN] 当前chunk没有有效的行数据")
                continue
            
            print(f"[INFO] 成功提取 {len(line_data_list)} 行特征")
            
            pred_results = []
            num_batches = (len(line_data_list) + batch_size - 1) // batch_size
            
            with torch.no_grad():
                for batch_idx in tqdm(range(num_batches), desc="批量推理"):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(line_data_list))
                    batch_data = line_data_list[start:end]
                    
                    if not batch_data:
                        continue
                    
                    max_width = max(item['resized_w'] for item in batch_data)
                    n_channels = batch_data[0]['features'].shape[1]
                    
                    batch_features = np.zeros((len(batch_data), n_channels, max_width), dtype=np.float32)
                    for i, item in enumerate(batch_data):
                        w = item['resized_w']
                        batch_features[i, :, :w] = item['features'].transpose(1, 0)
                    
                    features_tensor = torch.from_numpy(batch_features).to(device)
                    output = model(features_tensor)
                    pred_probs = torch.sigmoid(output).cpu().numpy()
                    
                    for i, item in enumerate(batch_data):
                        w = item['resized_w']
                        pred_prob = pred_probs[i, 0, :w]
                        
                        intervals = IntervalExtractor.extract(pred_prob, threshold, max_gap)
                        
                        inv_scale = 1.0 / item['scale'] if item['scale'] > 0 else 1.0
                        intervals_orig = [
                            (int(round(s * inv_scale)), int(round(e * inv_scale)))
                            for s, e in intervals
                        ]
                        
                        pred_results.append({
                            'line_id': item['line_id'],
                            'model_intervals': intervals_orig,
                            'pred_prob': pred_prob,
                            'image_width': item['image_width'],
                            'rule_intervals': item['rule_intervals']
                        })
            
            chunk_scores = []
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self._compute_al_score_from_result, pred): pred['line_id']
                          for pred in pred_results}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="AL评分"):
                    result = future.result()
                    chunk_scores.append(result)
            
            global_top_scores.extend(chunk_scores)
            global_top_scores.sort(key=lambda x: x['al_score'], reverse=True)
            global_top_scores = global_top_scores[:top_n * 2]
            
            processed_lines = (chunk_idx + 1) * chunk_size
            processed_lines = min(processed_lines, total_lines)
            
            print(f"\n[INFO] 当前已处理 {processed_lines}/{total_lines} 行")
            print(f"[INFO] 当前Top {min(top_n, len(global_top_scores))} 排名：")
            print("-" * 90)
            print(f"{'排名':<4} {'行ID':<40} {'AL分数':<10} {'分歧':<10}")
            print("-" * 90)
            
            for idx, item in enumerate(global_top_scores[:top_n], 1):
                print(f"{idx:<4} {item['line_id']:<40} {item['al_score']:<10.4f} {item['disagreement']:<10.4f}")
        
        global_top_scores.sort(key=lambda x: x['al_score'], reverse=True)
        
        return global_top_scores[:top_n]


MERGE_MIN_GAP = 3
MERGE_SINGLE_ASPECT_RATIO = 0.7
MERGE_MIN_ASPECT_RATIO = 0.5
MERGE_MAX_ASPECT_RATIO = 1.5


class RuleBasedActiveLearner:
    def __init__(self, data_base_path: Path):
        self.data_base_path = data_base_path
        self.rule_jsons_dir = data_base_path / "rule_jsons"
        self.lines_dir = data_base_path / "lines"
        self.lineage_path = data_base_path / "lineage.json"
        self._lineage_cache = None
    
    def _load_lineage(self) -> dict:
        if self._lineage_cache is None:
            with open(self.lineage_path, 'r', encoding='utf-8') as f:
                self._lineage_cache = json.load(f)
        return self._lineage_cache
    
    def get_line_info(self, line_id: str) -> Optional[dict]:
        lineage = self._load_lineage()
        return lineage.get('lines', {}).get(line_id)
    
    def get_chars_from_lineage(self, line_id: str) -> Optional[List[dict]]:
        line_info = self.get_line_info(line_id)
        if line_info is None:
            return None
        
        char_ids = line_info.get('chars', [])
        if not char_ids:
            return None
        
        lineage = self._load_lineage()
        chars_ref = lineage.get('chars', {})
        
        chars = []
        for cid in char_ids:
            char_data = chars_ref.get(cid)
            if char_data is not None:
                chars.append({
                    'col_start': char_data.get('col_start', 0),
                    'col_end': char_data.get('col_end', 0),
                    'width': char_data.get('width', 0),
                    'height': char_data.get('height', 0),
                    'merged_from': char_data.get('merged_from', None)
                })
        
        return chars
    
    def _load_line_image(self, line_id: str) -> Optional[np.ndarray]:
        line_path = self.lines_dir / f"{line_id}.png"
        if not line_path.exists():
            return None
        return cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
    
    def _compute_vertical_projection(self, line_img: np.ndarray) -> np.ndarray:
        h, w = line_img.shape[:2]
        proj = np.sum(255 - line_img, axis=0) / 255
        return proj
    
    def _find_local_valleys(self, proj: np.ndarray, th_blank: float, min_width: int = 5) -> List[Tuple[int, int, float]]:
        """
        找出投影曲线中的局部谷底
        
        Args:
            proj: 垂直投影曲线
            th_blank: 空白阈值
            min_width: 谷底最小宽度
        
        Returns:
            谷底列表，每个元素为 (start_idx, end_idx, min_val)
        """
        valleys = []
        n = len(proj)
        i = 0
        
        while i < n:
            if proj[i] < th_blank:
                start = i
                min_val = proj[i]
                while i < n and proj[i] < th_blank:
                    if proj[i] < min_val:
                        min_val = proj[i]
                    i += 1
                end = i - 1
                
                if end - start + 1 >= min_width:
                    valleys.append((start, end, min_val))
            else:
                i += 1
        
        return valleys
    
    def _find_high_density_regions(self, proj: np.ndarray, th_text: float, min_width: int = 5) -> List[Tuple[int, int, float]]:
        """
        找出投影曲线中的高密度区域（文字区域）
        
        Args:
            proj: 垂直投影曲线
            th_text: 文字阈值
            min_width: 最小宽度
        
        Returns:
            高密度区域列表，每个元素为 (start_idx, end_idx, avg_val)
        """
        regions = []
        n = len(proj)
        i = 0
        
        while i < n:
            if proj[i] >= th_text:
                start = i
                total_val = proj[i]
                count = 1
                while i < n and proj[i] >= th_text:
                    total_val += proj[i]
                    count += 1
                    i += 1
                end = i - 1
                
                if end - start + 1 >= min_width:
                    avg_val = total_val / count
                    regions.append((start, end, avg_val))
            else:
                i += 1
        
        return regions
    
    def stuck_unsplittable_score(self, line_id: str, chars: List[dict], 
                                  image_width: int, image_height: int) -> float:
        """
        粘连不可分分数：检测宽字符区间内垂直投影无法区分的粘连字符
        
        改进的检测逻辑：
        1. 找出宽字符（宽度 > 平均宽度×1.2 或 宽高比 > 1.2）- 降低阈值提高敏感度
        2. 读取行图像，计算垂直投影曲线
        3. 在宽字符区间内检测：
           - 多个高密度区域（可能是多个字符）
           - 谷底数量和深度（谷底越多越深，说明越可能是粘连）
           - 投影曲线的波动情况（波动大说明可能包含多个字符）
        4. 综合评分：考虑谷底数量、深度、区域数量等因素
        """
        if len(chars) == 0 or image_height <= 0:
            return 0.0
        
        widths = [char.get('width', 0) for char in chars if char.get('width', 0) > 0]
        if not widths:
            return 0.0
        
        avg_width = np.mean(widths)
        median_width = np.median(widths)
        
        wide_chars = []
        
        for char in chars:
            w = char.get('width', 0)
            if w <= 0:
                continue
            aspect_ratio = w / image_height
            if w > avg_width * 1.2 or aspect_ratio > 1.2:
                wide_chars.append(char)
        
        if not wide_chars:
            return 0.0
        
        line_img = self._load_line_image(line_id)
        if line_img is None:
            return 0.0
        
        proj = self._compute_vertical_projection(line_img)
        
        if proj.max() == 0:
            return 0.0
        
        th_text = proj.max() * 0.4
        th_blank = proj.max() * 0.2
        
        total_score = 0.0
        
        for char in wide_chars:
            start = char.get('col_start', 0)
            end = char.get('col_end', 0)
            w = char.get('width', 0)
            
            if w < avg_width * 1.0:
                continue
            
            seg_proj = proj[start:end+1]
            
            if len(seg_proj) < 8:
                continue
            
            regions = self._find_high_density_regions(seg_proj, th_text, min_width=3)
            valleys = self._find_local_valleys(seg_proj, th_blank, min_width=2)
            
            min_val = np.min(seg_proj)
            max_val = np.max(seg_proj)
            mean_val = np.mean(seg_proj)
            std_val = np.std(seg_proj)
            
            char_score = 0.0
            
            if len(regions) >= 2:
                char_score += 0.3 * (len(regions) - 1)
            
            if len(valleys) >= 1:
                valley_depth = (th_blank - min_val) / th_blank if th_blank > 0 else 0
                char_score += 0.3 * len(valleys) * min(1, valley_depth)
            
            width_ratio = w / avg_width
            if width_ratio >= 1.5:
                char_score += 0.2 * (width_ratio - 1.5)
            
            if std_val > 0.1:
                char_score += 0.2 * min(std_val * 2, 1)
            
            total_score += min(char_score, 1.0)
        
        return total_score / len(chars)
    
    def under_merged_score(self, chars: List[dict], image_width: int, image_height: int) -> float:
        """
        合并不足分数：检测本该合并但没有合并的汉字
        
        检测逻辑：
        1. 在合并后的字符列表中，寻找相邻的两个窄片段
        2. 如果两个片段的宽度都远小于平均宽度（< 0.7倍平均宽度），
           且它们之间的间隙很小（< 3px），说明它们本应该是一个汉字的左右部件
        3. 分数 = 合并不足的相邻片段对数 / 总片段对数
        """
        if len(chars) < 2 or image_height <= 0:
            return 0.0
        
        widths = [char.get('width', 0) for char in chars if char.get('width', 0) > 0]
        if len(widths) < 2:
            return 0.0
        
        avg_width = np.mean(widths)
        
        under_merge_count = 0
        total_pairs = 0
        
        for i in range(1, len(chars)):
            prev = chars[i-1]
            curr = chars[i]
            
            gap = curr.get('col_start', 0) - prev.get('col_end', 0)
            
            if gap < 0:
                continue
            
            w1 = prev.get('width', 0)
            w2 = curr.get('width', 0)
            
            if w1 <= 0 or w2 <= 0:
                continue
            
            r1 = w1 / image_height
            r2 = w2 / image_height
            
            total_pairs += 1
            
            if (w1 < avg_width * 0.7 or r1 < 0.5) and \
               (w2 < avg_width * 0.7 or r2 < 0.5) and \
               gap <= 3:
                under_merge_count += 1
        
        if total_pairs == 0:
            return 0.0
        
        return under_merge_count / total_pairs
    
    def compute_al_score(self, line_id: str) -> Optional[Dict]:
        """
        基于后处理合并后的规则切割结果计算主动学习分数
        
        核心指标：
        1. 粘连不可分（stuck_unsplittable）：垂直投影无法区分的粘连字符（权重0.9）
        2. 合并不足（under_merged）：本该合并但没有合并的汉字（权重0.1）
        
        数据来源：lineage.json 中的后处理合并结果
        """
        line_info = self.get_line_info(line_id)
        if line_info is None:
            return None
        
        image_width = line_info.get('width', 0)
        image_height = line_info.get('height', 0)
        
        chars = self.get_chars_from_lineage(line_id)
        if chars is None or len(chars) == 0:
            return None
        
        stuck_score = self.stuck_unsplittable_score(line_id, chars, image_width, image_height)
        under_merge_score = self.under_merged_score(chars, image_width, image_height)
        
        weights = {
            'stuck_unsplittable': 0.9,
            'under_merged': 0.1
        }
        
        al_score = (
            weights['stuck_unsplittable'] * stuck_score +
            weights['under_merged'] * under_merge_score
        )
        
        return {
            'line_id': line_id,
            'al_score': al_score,
            'stuck_unsplittable': stuck_score,
            'under_merged': under_merge_score,
            'total_chars': len(chars),
            'image_width': image_width,
            'image_height': image_height
        }
    
    def rank_lines(self, top_n: int = 100, chunk_size: int = 1000) -> List[Dict]:
        """
        基于后处理合并后的规则切割结果对所有行数据进行主动学习排序
        
        返回：按AL分数从高到低排序的行列表（最可能切割错误的排在前面）
        """
        line_ids = load_all_line_ids(self.data_base_path)
        total_lines = len(line_ids)
        print(f"[INFO] 总共有 {total_lines} 个行数据")
        
        chunks = [line_ids[i:i+chunk_size] for i in range(0, total_lines, chunk_size)]
        num_chunks = len(chunks)
        print(f"[INFO] 分成 {num_chunks} 个chunk，每chunk {chunk_size} 个")
        
        global_top_scores = []
        
        for chunk_idx, chunk_line_ids in enumerate(chunks):
            print(f"\n{'='*60}")
            print(f"[INFO] 处理第 {chunk_idx + 1}/{num_chunks} 个chunk（{len(chunk_line_ids)} 行）")
            print(f"{'='*60}")
            
            chunk_scores = []
            for line_id in tqdm(chunk_line_ids, desc="规则分析"):
                result = self.compute_al_score(line_id)
                if result is not None:
                    chunk_scores.append(result)
            
            global_top_scores.extend(chunk_scores)
            global_top_scores.sort(key=lambda x: x['al_score'], reverse=True)
            global_top_scores = global_top_scores[:top_n * 2]
            
            processed_lines = (chunk_idx + 1) * chunk_size
            processed_lines = min(processed_lines, total_lines)
            
            print(f"\n[INFO] 当前已处理 {processed_lines}/{total_lines} 行")
            print(f"[INFO] 当前Top {min(top_n, len(global_top_scores))} 排名：")
            print("-" * 120)
            print(f"{'排名':<4} {'行ID':<40} {'AL分数':<10} {'粘连不可分':<12} {'合并不足':<10}")
            print("-" * 120)
            
            for idx, item in enumerate(global_top_scores[:top_n], 1):
                print(f"{idx:<4} {item['line_id']:<40} {item['al_score']:<10.4f} "
                      f"{item['stuck_unsplittable']:<12.4f} {item['under_merged']:<10.4f}")
        
        global_top_scores.sort(key=lambda x: x['al_score'], reverse=True)
        
        return global_top_scores[:top_n]


@click.command("active-learn")
@click.argument("top_n", type=int, default=100)
@click.option("--model-path", type=click.Path(exists=True), default=None, help="模型权重路径")
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录（默认: <项目根>/datahome）")
@click.option("--output", type=str, default=None, help="排名结果输出路径")
@click.option("--batch-size", type=int, default=32, show_default=True,
              help="GPU批量推理批大小")
@click.option("--num-workers", type=int, default=4, show_default=True,
              help="并行特征提取和评分的线程数")
@click.option("--chunk-size", type=int, default=1000, show_default=True,
              help="微批量大小，每处理完一个chunk更新一次排名")
def cli(top_n, model_path, data_base_path, output, batch_size, num_workers, chunk_size):
    base_dir = Path(__file__).resolve().parent.parent.parent

    data_path = Path(data_base_path) if data_base_path else base_dir / "datahome"
    model_dir = base_dir / "ai_model" / "models"
    model_file = Path(model_path) if model_path else model_dir / "char_segment_1d_unet_best.pth"

    if not model_file.exists():
        click.echo(f"[ERROR] 模型文件不存在: {model_file}", err=True)
        sys.exit(1)

    click.echo(f"[INFO] 加载模型: {model_file}")
    learner = ActiveLearner(model_file, data_path)

    click.echo(f"[INFO] 开始计算主动学习分数（batch_size={batch_size}, num_workers={num_workers}, chunk_size={chunk_size}）...")
    ranked_lines = learner.rank_lines_batched(top_n, batch_size=batch_size, num_workers=num_workers, chunk_size=chunk_size)

    click.echo(f"\n[INFO] Top {len(ranked_lines)} 需要优先标注的行：")
    click.echo("-" * 120)
    click.echo(f"{'排名':<4} {'行ID':<40} {'AL分数':<10} {'不确定性':<10} {'分歧':<10} {'数量差异':<10}")
    click.echo("-" * 120)

    for idx, item in enumerate(ranked_lines, 1):
        click.echo(f"{idx:<4} {item['line_id']:<40} {item['al_score']:<10.4f} "
                   f"{item['uncertainty']:<10.4f} {item['disagreement']:<10.4f} "
                   f"{item['count_diff']:<10.4f}")

    output_path = Path(output) if output else model_dir / "al_ranking.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ranked_lines, f, indent=2, ensure_ascii=False)

    click.echo(f"\n[INFO] 排名结果已保存到: {output_path}")


@click.command("rule-based-al")
@click.argument("top_n", type=int, default=100)
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录（默认: <项目根>/datahome）")
@click.option("--output", type=str, default=None, help="排名结果输出路径")
@click.option("--chunk-size", type=int, default=1000, show_default=True,
              help="微批量大小，每处理完一个chunk更新一次排名")
def cli_rule_based(top_n, data_base_path, output, chunk_size):
    """
    基于规则的主动学习（后处理合并结果）：识别粘连不可分和合并不足的行
    
    数据来源：lineage.json 中的后处理合并结果
    
    核心指标：
    - 粘连不可分（stuck_unsplittable，权重0.9）：检测宽字符区间内垂直投影曲线
      是否存在未降至阈值的局部谷底，有谷底说明本该断开但投影值不够低
    
    - 合并不足（under_merged，权重0.1）：检测合并后相邻的窄片段+小间隙模式，
      这些片段本应是一个汉字的左右部件，没有被正确合并
    
    返回最可能切割错误的 Top N 行，用于人工审核和标注。
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_path = Path(data_base_path) if data_base_path else base_dir / "datahome"
    model_dir = base_dir / "ai_model" / "models"

    click.echo(f"[INFO] 数据目录: {data_path}")
    learner = RuleBasedActiveLearner(data_path)

    click.echo(f"[INFO] 开始基于规则的主动学习分析（chunk_size={chunk_size}）...")
    ranked_lines = learner.rank_lines(top_n, chunk_size=chunk_size)

    click.echo(f"\n[INFO] Top {len(ranked_lines)} 最可能切割错误的行：")
    click.echo("-" * 130)
    click.echo(f"{'排名':<4} {'行ID':<40} {'AL分数':<10} {'粘连不可分':<12} {'合并不足':<10}")
    click.echo("-" * 130)

    for idx, item in enumerate(ranked_lines, 1):
        click.echo(f"{idx:<4} {item['line_id']:<40} {item['al_score']:<10.4f} "
                   f"{item['stuck_unsplittable']:<12.4f} {item['under_merged']:<10.4f}")

    output_path = Path(output) if output else model_dir / "rule_based_al_ranking.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ranked_lines, f, indent=2, ensure_ascii=False)

    click.echo(f"\n[INFO] 排名结果已保存到: {output_path}")


if __name__ == "__main__":
    cli()
