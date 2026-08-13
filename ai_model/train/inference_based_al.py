import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from ai_model.inference.inference_storage import (
    read_inference_jsonl, read_detail_json, read_metadata, list_versions,
    get_latest_version, extract_pdf_id
)


class InferenceBasedActiveLearner:
    def __init__(self, data_base_path: Path, inference_version: str = None):
        self.data_base_path = data_base_path
        self.rule_jsons_dir = data_base_path / "rule_jsons"
        self.inference_base = data_base_path / "model_inference"
        
        if inference_version is None:
            inference_version = get_latest_version(self.inference_base)
        
        self.inference_version = inference_version
        self.inference_dir = self.inference_base / inference_version
        self.jsonl_path = self.inference_dir / "inference.jsonl"
        self.detail_dir = self.inference_dir / "detail"
        
        self._inference_cache = None
    
    def _load_inference_results(self) -> Dict[str, List[Tuple[int, int]]]:
        if self._inference_cache is None:
            self._inference_cache = read_inference_jsonl(self.jsonl_path)
        return self._inference_cache
    
    def load_rule_intervals(self, line_id: str) -> Optional[List[Tuple[int, int]]]:
        json_path = self.rule_jsons_dir / f"{line_id}_rule.json"
        if not json_path.exists():
            json_path = self.rule_jsons_dir / f"{line_id.replace('line_page_pdf_', 'line_page_')}_rule.json"
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
    
    def load_model_intervals(self, line_id: str) -> Optional[List[Tuple[int, int]]]:
        results = self._load_inference_results()
        return results.get(line_id)
    
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
    
    def compute_al_score(self, line_id: str) -> Optional[Dict]:
        model_intervals = self.load_model_intervals(line_id)
        if model_intervals is None:
            return None
        
        rule_intervals = self.load_rule_intervals(line_id)
        if rule_intervals is None:
            return None
        
        detail = read_detail_json(self.detail_dir, line_id)
        image_width = detail.get('width', 2000) if detail else 2000
        
        disagreement = self.disagreement_score(rule_intervals, model_intervals, image_width)
        count_diff = self.interval_count_diff_score(rule_intervals, model_intervals)
        boundary_diff = self.boundary_displacement_score(rule_intervals, model_intervals)
        
        weights = {
            'disagreement': 0.5,
            'count_diff': 0.3,
            'boundary_diff': 0.2
        }
        
        al_score = (
            weights['disagreement'] * disagreement +
            weights['count_diff'] * count_diff +
            weights['boundary_diff'] * boundary_diff
        )
        
        return {
            'line_id': line_id,
            'al_score': al_score,
            'disagreement': disagreement,
            'count_diff': count_diff,
            'boundary_diff': boundary_diff,
            'rule_interval_count': len(rule_intervals),
            'model_interval_count': len(model_intervals),
            'image_width': image_width
        }
    
    def rank_lines(self, top_n: int = 100) -> List[Dict]:
        print(f"[INFO] 使用推理版本: {self.inference_version}")
        
        inference_results = self._load_inference_results()
        total_lines = len(inference_results)
        print(f"[INFO] 推理结果包含 {total_lines} 行数据")
        
        scores = []
        for i, line_id in enumerate(inference_results.keys()):
            if (i + 1) % 5000 == 0:
                print(f"[INFO] 已处理 {i+1}/{total_lines} 行")
            
            result = self.compute_al_score(line_id)
            if result is not None:
                scores.append(result)
        
        scores.sort(key=lambda x: x['al_score'], reverse=True)
        
        print(f"\n[INFO] Top {min(top_n, len(scores))} 需要优先标注的行：")
        print("-" * 110)
        print(f"{'排名':<4} {'行ID':<40} {'AL分数':<10} {'分歧':<10} {'数量差异':<10}")
        print("-" * 110)
        
        for idx, item in enumerate(scores[:top_n], 1):
            print(f"{idx:<4} {item['line_id']:<40} {item['al_score']:<10.4f} "
                  f"{item['disagreement']:<10.4f} {item['count_diff']:<10.4f}")
        
        return scores[:top_n]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="基于推理结果的主动学习")
    parser.add_argument("--top-n", type=int, default=100, help="返回Top N行")
    parser.add_argument("--data-base-path", type=str, default=None, help="数据基础目录")
    parser.add_argument("--inference-version", type=str, default=None, help="推理版本")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_path = Path(args.data_base_path) if args.data_base_path else base_dir / "datahome"
    
    learner = InferenceBasedActiveLearner(data_path, args.inference_version)
    
    print(f"[INFO] 数据目录: {data_path}")
    print(f"[INFO] 开始基于推理结果的主动学习分析...")
    
    ranked_lines = learner.rank_lines(args.top_n)
    
    output_path = Path(args.output) if args.output else base_dir / "ai_model" / "models" / "inference_al_ranking.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ranked_lines, f, indent=2, ensure_ascii=False)
    
    print(f"\n[INFO] 排名结果已保存到: {output_path}")


if __name__ == "__main__":
    main()