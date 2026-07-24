import json
import numpy as np
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from ai_model.inference.infer import CharSegmentPredictor, SequenceDecoder
from ai_model.data.dataset import FeatureExtractor


def load_annotations(annotations_path):
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations_list = json.load(f)
    return {item['line_id']: item for item in annotations_list}


def find_line_image(line_id):
    possible_dirs = [
        PROJECT_ROOT / "datahome" / "datasets" / "done3",
        PROJECT_ROOT / "datahome" / "datasets" / "done2",
        PROJECT_ROOT / "datahome" / "datasets" / "done",
        PROJECT_ROOT / "datahome" / "project" / "proj_20260706_230417" / "lines",
    ]
    
    for dir_path in possible_dirs:
        line_path = dir_path / f"{line_id}.png"
        if line_path.exists():
            return line_path
    
    return None


def create_benchmark_dataset():
    model_path = PROJECT_ROOT / "ai_model" / "models" / "char_segment_1d_unet_best.pth"
    annotations_path = PROJECT_ROOT / "datahome" / "datasets" / "merged_annotations_proj_20260706_230417.json"
    benchmark_dir = PROJECT_ROOT / "tests" / "benchmark"
    
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    annotations = load_annotations(annotations_path)
    print(f"[INFO] 加载标注数据: {len(annotations)} 条")
    
    predictor = CharSegmentPredictor(model_path)
    
    benchmark_data = []
    
    for idx, (line_id, annotation) in enumerate(annotations.items()):
        line_path = find_line_image(line_id)
        
        if line_path is None:
            continue
        
        img = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        features, resized_w, scale = FeatureExtractor.extract(img)
        
        width = resized_w
        if width < predictor.max_width:
            pad_width = predictor.max_width - width
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        
        features_tensor = torch.from_numpy(features.transpose(1, 0)).unsqueeze(0).to(predictor.device, dtype=torch.float32)
        
        with torch.no_grad():
            output = predictor.model(features_tensor)
            pred_prob = torch.softmax(output, dim=1).squeeze().cpu().numpy()[:, :width]
            pred_class = np.argmax(pred_prob, axis=0)
        
        gt_chars = annotation['chars']
        gt_intervals = [(c['col_start'], c['col_end']) for c in gt_chars]
        
        pred_intervals = SequenceDecoder.decode(pred_class, pred_prob)
        
        inv_scale = 1.0 / scale if scale > 0 else 1.0
        gt_intervals_scaled = [(int(round(s * inv_scale)), int(round(e * inv_scale))) for s, e in gt_intervals]
        pred_intervals_scaled = [(int(round(s * inv_scale)), int(round(e * inv_scale))) for s, e in pred_intervals]
        
        difficulty = classify_difficulty(gt_intervals_scaled)
        
        benchmark_data.append({
            'line_id': line_id,
            'pred_class': pred_class,
            'pred_prob': pred_prob,
            'gt_intervals': gt_intervals_scaled,
            'pred_intervals': pred_intervals_scaled,
            'difficulty': difficulty,
            'num_chars': len(gt_chars),
            'scale': scale,
            'width': width
        })
        
        if (idx + 1) % 20 == 0:
            print(f"[INFO] 处理进度: {idx + 1}/{len(annotations)}, 已保存: {len(benchmark_data)}")
    
    np.save(str(benchmark_dir / "benchmark_data.npy"), benchmark_data)
    
    difficulty_counts = {}
    for item in benchmark_data:
        d = item['difficulty']
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
    
    print(f"\n[INFO] 基准数据集生成完成")
    print(f"[INFO] 总样本数: {len(benchmark_data)}")
    print(f"[INFO] 难度分布: {difficulty_counts}")
    
    return benchmark_dir


def classify_difficulty(gt_intervals):
    if len(gt_intervals) < 2:
        return 'simple'
    
    gaps = []
    for i in range(len(gt_intervals) - 1):
        gap = gt_intervals[i+1][0] - gt_intervals[i][1]
        gaps.append(gap)
    
    min_gap = min(gaps) if gaps else 0
    avg_gap = sum(gaps) / len(gaps) if gaps else 0
    
    char_widths = [e - s for s, e in gt_intervals]
    avg_width = sum(char_widths) / len(char_widths) if char_widths else 0
    
    shared_boundary_count = sum(1 for g in gaps if g <= 0)
    
    if shared_boundary_count >= 2:
        return 'hard'
    elif min_gap <= 2 or avg_gap < avg_width * 0.2:
        return 'medium'
    else:
        return 'simple'


def create_synthetic_test_cases():
    test_cases = []
    
    test_cases.append({
        'name': 'normal_char',
        'description': '正常字符',
        'difficulty': 'simple',
        'pred_class': np.array([0, 1, 2, 2, 1, 0]),
        'expected': [(1, 4)]
    })
    
    test_cases.append({
        'name': 'normal_char_no_boundary_left',
        'description': '左边界缺失',
        'difficulty': 'simple',
        'pred_class': np.array([0, 0, 2, 2, 1, 0]),
        'expected': [(2, 4)]
    })
    
    test_cases.append({
        'name': 'normal_char_no_boundary_right',
        'description': '右边界缺失',
        'difficulty': 'simple',
        'pred_class': np.array([0, 1, 2, 2, 0, 0]),
        'expected': [(1, 3)]
    })
    
    test_cases.append({
        'name': 'shared_boundary_complete',
        'description': '完全共享边界',
        'difficulty': 'medium',
        'pred_class': np.array([1, 2, 1, 1, 2, 1]),
        'expected': [(0, 2), (2, 5)]
    })
    
    test_cases.append({
        'name': 'shared_boundary_adjacent',
        'description': '相邻共享边界',
        'difficulty': 'medium',
        'pred_class': np.array([1, 2, 1, 0, 1, 2, 1]),
        'expected': [(0, 1), (2, 2), (4, 4), (5, 6)]
    })
    
    test_cases.append({
        'name': 'blank_separation',
        'description': '空白分隔两个字符',
        'difficulty': 'hard',
        'pred_class': np.array([1, 2, 2, 0, 0, 2, 2, 1, 0]),
        'expected': [(0, 2), (5, 7)]
    })
    
    test_cases.append({
        'name': 'blank_separation_narrow',
        'description': '窄空白分隔',
        'difficulty': 'hard',
        'pred_class': np.array([1, 2, 2, 0, 2, 2, 1, 0]),
        'expected': [(0, 2), (4, 6)]
    })
    
    test_cases.append({
        'name': 'no_boundary',
        'description': '无边界点',
        'difficulty': 'simple',
        'pred_class': np.array([0, 2, 2, 2, 0]),
        'expected': [(1, 3)]
    })
    
    test_cases.append({
        'name': 'width_1_char',
        'description': '宽度1字符',
        'difficulty': 'medium',
        'pred_class': np.array([0, 1, 0, 1, 2, 2, 1, 0]),
        'expected': [(1, 1), (3, 6)]
    })
    
    test_cases.append({
        'name': 'multiple_blanks',
        'description': '多个空白分隔',
        'difficulty': 'hard',
        'pred_class': np.array([1, 2, 0, 0, 2, 0, 2, 1]),
        'expected': [(0, 1), (4, 4), (6, 7)]
    })
    
    test_cases.append({
        'name': 'dense_boundary_noise',
        'description': '密集边界噪声',
        'difficulty': 'medium',
        'pred_class': np.array([1, 1, 2, 2, 1, 1, 0]),
        'expected': [(0, 0), (1, 4), (5, 5)]
    })
    
    test_cases.append({
        'name': 'complex_shared_boundary',
        'description': '复杂共享边界',
        'difficulty': 'hard',
        'pred_class': np.array([1, 2, 1, 1, 2, 2, 1, 1, 2, 1]),
        'expected': [(0, 2), (2, 6), (6, 9)]
    })
    
    test_cases.append({
        'name': 'overlapping_internal',
        'description': '重叠内部区域',
        'difficulty': 'hard',
        'pred_class': np.array([1, 2, 2, 2, 1, 1, 2, 2, 1]),
        'expected': [(0, 4), (4, 8)]
    })
    
    test_cases.append({
        'name': 'all_internal',
        'description': '全内部类',
        'difficulty': 'simple',
        'pred_class': np.array([2, 2, 2, 2]),
        'expected': [(0, 3)]
    })
    
    test_cases.append({
        'name': 'edge_case_start',
        'description': '边界点在开头',
        'difficulty': 'simple',
        'pred_class': np.array([1, 2, 2, 1, 0]),
        'expected': [(0, 3)]
    })
    
    test_cases.append({
        'name': 'edge_case_end',
        'description': '边界点在末尾',
        'difficulty': 'simple',
        'pred_class': np.array([0, 1, 2, 2, 1]),
        'expected': [(1, 4)]
    })
    
    benchmark_dir = PROJECT_ROOT / "tests" / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(str(benchmark_dir / "synthetic_test_cases.npy"), test_cases)
    
    difficulty_counts = {}
    for tc in test_cases:
        d = tc['difficulty']
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
    
    print(f"\n[INFO] 合成测试集生成完成")
    print(f"[INFO] 测试用例数: {len(test_cases)}")
    print(f"[INFO] 难度分布: {difficulty_counts}")
    
    return test_cases


if __name__ == "__main__":
    import cv2
    
    print("=" * 60)
    print("创建基准测试数据集")
    print("=" * 60)
    
    create_benchmark_dataset()
    create_synthetic_test_cases()
    
    print("\n" + "=" * 60)
    print("基准测试数据集创建完成！")
    print("=" * 60)