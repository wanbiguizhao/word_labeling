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
        
        model_intervals, pred_prob, scale = result
        
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


if __name__ == "__main__":
    cli()