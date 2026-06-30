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
        """加载规则分割结果"""
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
        """
        不确定性分数：预测概率接近0.5的列越多，不确定性越高
        
        公式：U = mean(min(p, 1-p) * 2)
        即：平均距离0.5的距离（归一化后）
        """
        distances = np.minimum(pred_prob, 1 - pred_prob) * 2
        return float(np.mean(distances))
    
    def entropy_score(self, pred_prob: np.ndarray) -> float:
        """
        熵分数：预测概率分布越均匀，熵越大
        
        公式：E = -mean(p * log(p) + (1-p) * log(1-p))
        """
        eps = 1e-10
        entropy = -np.mean(pred_prob * np.log(pred_prob + eps) + (1 - pred_prob) * np.log(1 - pred_prob + eps))
        return float(entropy)
    
    def disagreement_score(self, rule_intervals: List[Tuple[int, int]], 
                           model_intervals: List[Tuple[int, int]], 
                           image_width: int) -> float:
        """
        分歧分数：规则分割和模型分割的IoU差异
        
        公式：D = 1 - IoU(rule_mask, model_mask)
        """
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
        """
        区间数量差异分数：规则和模型识别的字符数量差异
        
        公式：C = |rule_count - model_count| / max(rule_count, model_count, 1)
        """
        rule_count = len(rule_intervals)
        model_count = len(model_intervals)
        max_count = max(rule_count, model_count, 1)
        return abs(rule_count - model_count) / max_count
    
    def boundary_displacement_score(self, rule_intervals: List[Tuple[int, int]], 
                                     model_intervals: List[Tuple[int, int]]) -> float:
        """
        边界位移分数：规则和模型边界位置的平均差异
        
        公式：B = mean(|rule_boundary - model_boundary|) / avg_char_width
        """
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
        """
        计算主动学习综合分数
        
        综合分数 = w1 * 不确定性 + w2 * 熵 + w3 * 分歧 + w4 * 数量差异 + w5 * 边界位移
        """
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
        
        img = line_path.read_bytes()
        import cv2
        img_arr = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_GRAYSCALE)
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
        """
        对所有行数据进行主动学习排序
        
        返回：按AL分数从高到低排序的行列表
        """
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
        """提取单行的特征（供并行处理）"""
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
        """从预测结果计算AL分数（供并行处理）"""
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
        """
        微批量并行推理版本：对所有行数据进行主动学习排序
        
        加速策略：
        1. 将数据分成多个chunk（默认1000个/批），逐个chunk处理
        2. 每个chunk内部：使用ThreadPoolExecutor并行提取特征（CPU密集）
        3. GPU批量推理（GPU密集，主要加速点）
        4. ThreadPoolExecutor并行计算AL分数（CPU密集）
        5. 每处理完一个chunk立即更新全局Top N排名，快速看到结果
        
        Args:
            top_n: 返回前N个最需要标注的行
            batch_size: GPU推理批大小
            num_workers: 并行特征提取和评分的线程数
            chunk_size: 微批量大小，每处理完一个chunk更新一次排名
        
        Returns:
            按AL分数从高到低排序的行列表
        """
        import torch
        
        line_ids = load_all_line_ids(self.data_base_path)
        total_lines = len(line_ids)
        print(f"[INFO] 总共有 {total_lines} 个行数据")
        
        # 按chunk_size分块
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
            
            # Step 1: 并行提取当前chunk的特征
            print(f"[INFO] 并行提取特征（{num_workers} 线程）...")
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
            
            # Step 2: GPU批量推理
            print(f"[INFO] GPU批量推理（batch_size={batch_size}）...")
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
            
            # Step 3: 并行计算AL分数
            print(f"[INFO] 并行计算AL分数（{num_workers} 线程）...")
            chunk_scores = []
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self._compute_al_score_from_result, pred): pred['line_id']
                          for pred in pred_results}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="AL评分"):
                    result = future.result()
                    chunk_scores.append(result)
            
            # Step 4: 更新全局Top N排名
            global_top_scores.extend(chunk_scores)
            global_top_scores.sort(key=lambda x: x['al_score'], reverse=True)
            global_top_scores = global_top_scores[:top_n * 2]
            
            # 显示当前Top排名
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
MERGE_MAX_HEIGHT_DIFF = 0.2
MERGE_SINGLE_ASPECT_RATIO = 0.7
MERGE_MIN_ASPECT_RATIO = 0.5
MERGE_MAX_ASPECT_RATIO = 1.5


class RuleBasedActiveLearner:
    def __init__(self, data_base_path: Path):
        self.data_base_path = data_base_path
        self.rule_jsons_dir = data_base_path / "rule_jsons"
        self.lines_dir = data_base_path / "lines"
    
    def load_rule_json(self, line_id: str) -> Optional[dict]:
        json_path = self.rule_jsons_dir / f"{line_id}_rule.json"
        if not json_path.exists():
            return None
        
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _merge_chars(self, chars: List[dict], image_height: int) -> List[dict]:
        """
        对原始切割结果执行合并后处理，与训练时的 LabelGenerator._merge_narrow_chars 逻辑一致
        
        合并条件：
        1. 间隙 < min_gap
        2. 两个字符宽高比 < single_ratio（都窄）
        3. 合并后宽高比在 [min_ratio, max_ratio] 范围
        """
        if len(chars) < 2:
            return chars
        
        intervals = [(char['col_start'], char['col_end']) for char in chars]
        
        merged_intervals = []
        for start, end in intervals:
            if not merged_intervals:
                merged_intervals.append((start, end))
            else:
                last_start, last_end = merged_intervals[-1]
                gap = start - last_end - 1
                
                if gap >= 0 and gap < MERGE_MIN_GAP:
                    w1 = last_end - last_start
                    w2 = end - start
                    r1 = w1 / image_height
                    r2 = w2 / image_height
                    
                    if r1 < MERGE_SINGLE_ASPECT_RATIO and r2 < MERGE_SINGLE_ASPECT_RATIO:
                        merged_w = end - last_start
                        merged_r = merged_w / image_height
                        
                        if MERGE_MIN_ASPECT_RATIO < merged_r < MERGE_MAX_ASPECT_RATIO:
                            merged_intervals[-1] = (last_start, end)
                            continue
                
                merged_intervals.append((start, end))
        
        merged_chars = []
        for start, end in merged_intervals:
            merged_chars.append({
                'col_start': start,
                'col_end': end,
                'width': end - start
            })
        
        return merged_chars
    
    def narrow_char_score(self, chars: List[dict], image_height: int) -> float:
        """
        窄字符比例：宽度远小于平均宽度的字符比例
        
        窄字符定义：width < 平均宽度 * 0.5 且 width/height < 0.5
        """
        if len(chars) < 2:
            return 0.0
        
        widths = [char.get('width', 0) for char in chars if char.get('width', 0) > 0]
        if not widths:
            return 0.0
        
        avg_width = np.mean(widths)
        narrow_count = 0
        
        for char in chars:
            w = char.get('width', 0)
            if w > 0 and w < avg_width * 0.5:
                if image_height > 0:
                    aspect_ratio = w / image_height
                    if aspect_ratio < 0.5:
                        narrow_count += 1
                else:
                    narrow_count += 1
        
        return narrow_count / len(chars)
    
    def stuck_char_score(self, chars: List[dict], image_height: int) -> float:
        """
        粘连字符分数：合并后仍存在宽高比在 1.5~3.0 之间的字符比例
        
        这些字符很可能是两个汉字粘在一起没有被正确切割
        
        判定条件：1.5 < width/height <= 3.0
        """
        if len(chars) == 0 or image_height <= 0:
            return 0.0
        
        stuck_count = 0
        for char in chars:
            w = char.get('width', 0)
            if w > 0:
                aspect_ratio = w / image_height
                if 1.5 < aspect_ratio <= 3.0:
                    stuck_count += 1
        
        return stuck_count / len(chars)
    
    def over_merge_score(self, chars_merged: List[dict], image_height: int) -> float:
        """
        合并过度分数：合并后宽高比 > 1.5 的字符比例
        
        合并后宽度仍然过大，说明可能合并了不该合并的字符（合并错误）
        
        判定条件：width/height > 1.5
        """
        if len(chars_merged) == 0 or image_height <= 0:
            return 0.0
        
        over_count = 0
        for char in chars_merged:
            w = char.get('width', 0)
            if w > 0:
                aspect_ratio = w / image_height
                if aspect_ratio > 1.5:
                    over_count += 1
        
        return over_count / len(chars_merged)
    
    def under_merge_score(self, chars_merged: List[dict], image_height: int) -> float:
        """
        合并不足分数：合并后仍存在相邻窄字符+小间隙的情况
        
        合并后仍有相邻字符都窄（width/height < 0.7）且间隙很小（gap < 3px），
        说明这些字符应该被合并但没有被合并
        
        判定条件：相邻两字符宽高比均 < 0.7，且间隙 < 3px
        """
        if len(chars_merged) < 2 or image_height <= 0:
            return 0.0
        
        under_count = 0
        total_pairs = 0
        
        for i in range(1, len(chars_merged)):
            prev = chars_merged[i-1]
            curr = chars_merged[i]
            
            w1 = prev.get('width', 0)
            w2 = curr.get('width', 0)
            gap = curr.get('col_start', 0) - prev.get('col_end', 0)
            
            if w1 > 0 and w2 > 0:
                r1 = w1 / image_height
                r2 = w2 / image_height
                total_pairs += 1
                
                if r1 < 0.7 and r2 < 0.7 and 0 <= gap < 3:
                    under_count += 1
        
        if total_pairs == 0:
            return 0.0
        return under_count / total_pairs
    
    def wide_char_score(self, chars: List[dict], image_height: int) -> float:
        """
        宽字符比例：宽度远大于平均宽度的字符比例
        
        宽字符定义：width > 平均宽度 * 2.0 或 width/height > 2.0
        """
        if len(chars) < 2:
            return 0.0
        
        widths = [char.get('width', 0) for char in chars if char.get('width', 0) > 0]
        if not widths:
            return 0.0
        
        avg_width = np.mean(widths)
        wide_count = 0
        
        for char in chars:
            w = char.get('width', 0)
            if w > 0:
                is_wide_by_avg = w > avg_width * 2.0
                is_wide_by_aspect = False
                if image_height > 0:
                    is_wide_by_aspect = (w / image_height) > 2.0
                if is_wide_by_avg or is_wide_by_aspect:
                    wide_count += 1
        
        return wide_count / len(chars)
    
    def gap_anomaly_score(self, chars: List[dict], image_width: int) -> float:
        """
        间隙异常分数：过小或过大的间隙比例
        
        间隙过小：gap < 2px（可能是分割过细）
        间隙过大：gap > 平均间隙 * 3.0（可能漏分割）
        """
        if len(chars) < 2:
            return 0.0
        
        gaps = []
        for i in range(1, len(chars)):
            prev_end = chars[i-1].get('col_end', 0)
            curr_start = chars[i].get('col_start', 0)
            gap = curr_start - prev_end
            if gap >= 0:
                gaps.append(gap)
        
        if not gaps:
            return 0.0
        
        avg_gap = np.mean(gaps)
        anomaly_count = 0
        
        for gap in gaps:
            if gap < 2 or (avg_gap > 0 and gap > avg_gap * 3.0):
                anomaly_count += 1
        
        return anomaly_count / len(gaps)
    
    def width_variance_score(self, chars: List[dict]) -> float:
        """
        字符宽度变异系数：宽度标准差 / 平均宽度
        
        值越大表示宽度变化越大，可能分割不稳定
        """
        widths = [char.get('width', 0) for char in chars if char.get('width', 0) > 0]
        if len(widths) < 2:
            return 0.0
        
        avg_width = np.mean(widths)
        std_width = np.std(widths)
        
        if avg_width == 0:
            return 0.0
        
        cv = std_width / avg_width
        return min(cv, 2.0) / 2.0
    
    def extreme_aspect_ratio_score(self, chars: List[dict], image_height: int) -> float:
        """
        极端宽高比字符比例：宽高比 < 0.3 或 > 3.0 的字符比例
        
        极端宽高比通常表示分割错误
        """
        if len(chars) == 0 or image_height <= 0:
            return 0.0
        
        extreme_count = 0
        for char in chars:
            w = char.get('width', 0)
            if w > 0:
                aspect_ratio = w / image_height
                if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                    extreme_count += 1
        
        return extreme_count / len(chars)
    
    def empty_region_score(self, chars: List[dict], image_width: int) -> float:
        """
        空白区域比例：最左边字符之前和最右边字符之后的空白区域
        
        空白区域过大可能表示漏识别或图像边缘问题
        """
        if len(chars) == 0 or image_width <= 0:
            return 0.0
        
        first_start = chars[0].get('col_start', image_width)
        last_end = chars[-1].get('col_end', 0)
        
        left_empty = first_start
        right_empty = image_width - last_end - 1
        
        total_empty = left_empty + right_empty
        return min(total_empty / image_width, 1.0)
    
    def line_density_score(self, chars: List[dict], image_width: int) -> float:
        """
        行密度异常分数：字符总宽度占图像宽度的比例
        
        密度过低：可能是稀疏文本或漏识别
        密度过高：可能是粘连严重或分割过粗
        """
        if len(chars) == 0 or image_width <= 0:
            return 0.0
        
        total_char_width = sum(char.get('width', 0) for char in chars)
        density = total_char_width / image_width
        
        if density < 0.1 or density > 0.8:
            return min(abs(density - 0.45) / 0.45, 1.0)
        
        return 0.0
    
    def compute_al_score(self, line_id: str) -> Optional[Dict]:
        """
        基于规则切割结果计算主动学习分数（以合并后指标为主）
        
        重点检测三类问题：
        1. 粘连字符：合并后宽高比 1.5~3.0，可能是两个汉字没切割开
        2. 合并过度：合并后宽高比 > 1.5，可能合并了不该合并的字符
        3. 合并不足：合并后仍有相邻窄字符+小间隙，该合并的没合并
        
        综合分数 = w1 * 粘连字符 + w2 * 合并过度 + w3 * 合并不足 +
                   w4 * 合并后宽字符 + w5 * 合并后间隙异常 + w6 * 合并后宽度变异 +
                   w7 * 合并率 + w8 * 合并后窄字符 + w9 * 合并后极端宽高比
        """
        rule_data = self.load_rule_json(line_id)
        if rule_data is None:
            return None
        
        chars_raw = rule_data.get('chars', [])
        image_width = rule_data.get('image_width', 0)
        image_height = rule_data.get('image_height', 0)
        
        if len(chars_raw) == 0:
            return None
        
        chars_merged = self._merge_chars(chars_raw, image_height)
        
        # === 核心指标（合并后，高权重） ===
        stuck_score = self.stuck_char_score(chars_merged, image_height)
        over_merge = self.over_merge_score(chars_merged, image_height)
        under_merge = self.under_merge_score(chars_merged, image_height)
        
        # === 辅助指标（合并后，中权重） ===
        merged_wide = self.wide_char_score(chars_merged, image_height)
        merged_gap_anomaly = self.gap_anomaly_score(chars_merged, image_width)
        merged_width_variance = self.width_variance_score(chars_merged)
        merged_narrow = self.narrow_char_score(chars_merged, image_height)
        merged_extreme = self.extreme_aspect_ratio_score(chars_merged, image_height)
        
        # === 参考指标（低权重） ===
        merge_ratio = 0.0
        if len(chars_raw) > 0:
            merge_ratio = 1.0 - len(chars_merged) / len(chars_raw)
        
        weights = {
            'stuck_char': 0.25,       # 粘连字符（最重要）
            'over_merge': 0.20,       # 合并过度
            'under_merge': 0.15,      # 合并不足
            'merged_wide': 0.10,      # 合并后宽字符
            'merged_gap_anomaly': 0.08,  # 合并后间隙异常
            'merged_width_variance': 0.07,  # 合并后宽度变异
            'merged_narrow': 0.05,    # 合并后窄字符
            'merged_extreme': 0.05,   # 合并后极端宽高比
            'merge_ratio': 0.05       # 合并率（参考）
        }
        
        al_score = (
            weights['stuck_char'] * stuck_score +
            weights['over_merge'] * over_merge +
            weights['under_merge'] * under_merge +
            weights['merged_wide'] * merged_wide +
            weights['merged_gap_anomaly'] * merged_gap_anomaly +
            weights['merged_width_variance'] * merged_width_variance +
            weights['merged_narrow'] * merged_narrow +
            weights['merged_extreme'] * merged_extreme +
            weights['merge_ratio'] * merge_ratio
        )
        
        return {
            'line_id': line_id,
            'al_score': al_score,
            'stuck_char': stuck_score,
            'over_merge': over_merge,
            'under_merge': under_merge,
            'merged_wide': merged_wide,
            'merged_gap_anomaly': merged_gap_anomaly,
            'merged_width_variance': merged_width_variance,
            'merged_narrow': merged_narrow,
            'merged_extreme': merged_extreme,
            'merge_ratio': merge_ratio,
            'total_chars_raw': len(chars_raw),
            'total_chars_merged': len(chars_merged),
            'image_width': image_width,
            'image_height': image_height
        }
    
    def rank_lines(self, top_n: int = 100, chunk_size: int = 1000) -> List[Dict]:
        """
        基于规则切割结果对所有行数据进行主动学习排序
        
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
            print(f"{'排名':<4} {'行ID':<40} {'AL分数':<10} {'粘连':<8} {'过度合并':<8} {'合并不足':<8} {'合并率':<8}")
            print("-" * 120)
            
            for idx, item in enumerate(global_top_scores[:top_n], 1):
                print(f"{idx:<4} {item['line_id']:<40} {item['al_score']:<10.4f} "
                      f"{item['stuck_char']:<8.4f} {item['over_merge']:<8.4f} "
                      f"{item['under_merge']:<8.4f} {item['merge_ratio']:<8.4f}")
        
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
    """
    主动学习：找出最需要标注的行

    加载训练好的模型对所有行数据进行推理，根据不确定性/分歧度/数量差异
    等指标排序，返回最需要人工标注的 Top N 行。

    使用 GPU 批量推理和多线程并行处理加速计算。
    数据按chunk分块处理，每处理完一个chunk立即显示当前Top排名，快速看到结果。
    """
    import torch
    
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
    基于规则的主动学习：仅使用规则切割结果识别可能切割错误的样本
    
    不需要训练好的模型，直接分析规则JSON中的字符特征：
    - 窄字符比例：宽度远小于平均宽度的字符（可能是分割过细）
    - 宽字符比例：宽度远大于平均宽度的字符（可能是粘连未分开）
    - 间隙异常：过小或过大的间隙（可能分割错误）
    - 宽度变异系数：字符宽度变化大（分割不稳定）
    - 极端宽高比：宽高比异常的字符（分割错误）
    - 空白区域：行首行尾空白过大（漏识别）
    - 行密度异常：字符密度过高或过低（粘连或漏识别）
    
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
    click.echo(f"{'排名':<4} {'行ID':<40} {'AL分数':<10} {'窄字符':<10} {'宽字符':<10} {'间隙异常':<10} {'宽度变异':<10}")
    click.echo("-" * 130)

    for idx, item in enumerate(ranked_lines, 1):
        click.echo(f"{idx:<4} {item['line_id']:<40} {item['al_score']:<10.4f} "
                   f"{item['narrow_char']:<10.4f} {item['wide_char']:<10.4f} "
                   f"{item['gap_anomaly']:<10.4f} {item['width_variance']:<10.4f}")

    output_path = Path(output) if output else model_dir / "rule_based_al_ranking.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ranked_lines, f, indent=2, ensure_ascii=False)

    click.echo(f"\n[INFO] 排名结果已保存到: {output_path}")


if __name__ == "__main__":
    cli()