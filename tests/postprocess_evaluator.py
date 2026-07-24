import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from ai_model.inference.infer import SequenceDecoder


DIFFICULTY_WEIGHTS = {
    'simple': 1.0,
    'medium': 1.5,
    'hard': 2.0
}


def evaluate_char_count(pred_intervals, gt_intervals):
    pred_count = len(pred_intervals)
    gt_count = len(gt_intervals)
    return 1.0 if pred_count == gt_count else max(0, 1 - abs(pred_count - gt_count) / max(gt_count, 1))


def evaluate_boundary_mae(pred_intervals, gt_intervals):
    if len(pred_intervals) == 0 and len(gt_intervals) == 0:
        return 1.0
    
    pred_boundaries = []
    for s, e in pred_intervals:
        pred_boundaries.append(s)
        pred_boundaries.append(e)
    
    gt_boundaries = []
    for s, e in gt_intervals:
        gt_boundaries.append(s)
        gt_boundaries.append(e)
    
    pred_boundaries.sort()
    gt_boundaries.sort()
    
    min_len = min(len(pred_boundaries), len(gt_boundaries))
    if min_len == 0:
        return 0.0
    
    mae = 0.0
    for i in range(min_len):
        mae += abs(pred_boundaries[i] - gt_boundaries[i])
    
    mae /= min_len
    
    max_possible_error = max(max(pred_boundaries + [0]), max(gt_boundaries + [0])) if pred_boundaries or gt_boundaries else 1
    return max(0, 1 - mae / max(max_possible_error, 1))


def evaluate_over_segmentation(pred_intervals, gt_intervals):
    pred_count = len(pred_intervals)
    gt_count = len(gt_intervals)
    
    if gt_count == 0:
        return 1.0 if pred_count == 0 else 0.0
    
    over_seg_ratio = max(0, pred_count - gt_count) / gt_count
    return max(0, 1 - over_seg_ratio)


def evaluate_under_segmentation(pred_intervals, gt_intervals):
    pred_count = len(pred_intervals)
    gt_count = len(gt_intervals)
    
    if gt_count == 0:
        return 1.0 if pred_count == 0 else 0.0
    
    under_seg_ratio = max(0, gt_count - pred_count) / gt_count
    return max(0, 1 - under_seg_ratio)


def evaluate_shared_boundary_recognition(pred_intervals, gt_intervals):
    if len(gt_intervals) < 2:
        return 1.0
    
    gt_shared_count = 0
    for i in range(len(gt_intervals) - 1):
        if gt_intervals[i][1] >= gt_intervals[i+1][0]:
            gt_shared_count += 1
    
    pred_shared_count = 0
    for i in range(len(pred_intervals) - 1):
        if pred_intervals[i][1] >= pred_intervals[i+1][0]:
            pred_shared_count += 1
    
    if gt_shared_count == 0:
        return 1.0 if pred_shared_count == 0 else 0.5
    else:
        return 1.0 if pred_shared_count == gt_shared_count else max(0, 1 - abs(pred_shared_count - gt_shared_count) / gt_shared_count)


def evaluate_intersection_iou(pred_intervals, gt_intervals):
    if len(pred_intervals) == 0 or len(gt_intervals) == 0:
        return 0.0 if len(pred_intervals) != len(gt_intervals) else 1.0
    
    pred_union = set()
    for s, e in pred_intervals:
        for x in range(s, e + 1):
            pred_union.add(x)
    
    gt_union = set()
    for s, e in gt_intervals:
        for x in range(s, e + 1):
            gt_union.add(x)
    
    intersection = pred_union & gt_union
    union = pred_union | gt_union
    
    if len(union) == 0:
        return 1.0
    
    return len(intersection) / len(union)


def evaluate_sample(pred_intervals, gt_intervals, difficulty='simple'):
    metrics = {}
    
    metrics['char_count_acc'] = evaluate_char_count(pred_intervals, gt_intervals)
    metrics['boundary_mae'] = evaluate_boundary_mae(pred_intervals, gt_intervals)
    metrics['over_seg'] = evaluate_over_segmentation(pred_intervals, gt_intervals)
    metrics['under_seg'] = evaluate_under_segmentation(pred_intervals, gt_intervals)
    metrics['shared_boundary_rec'] = evaluate_shared_boundary_recognition(pred_intervals, gt_intervals)
    metrics['iou'] = evaluate_intersection_iou(pred_intervals, gt_intervals)
    
    weights = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
    scores = [metrics['char_count_acc'], metrics['boundary_mae'], 
              metrics['over_seg'], metrics['under_seg'],
              metrics['shared_boundary_rec'], metrics['iou']]
    
    weighted_score = sum(w * s for w, s in zip(weights, scores)) * DIFFICULTY_WEIGHTS[difficulty]
    
    metrics['weighted_score'] = weighted_score
    metrics['difficulty_weight'] = DIFFICULTY_WEIGHTS[difficulty]
    
    return metrics


def evaluate_synthetic_test_cases():
    benchmark_dir = PROJECT_ROOT / "tests" / "benchmark"
    test_cases = np.load(str(benchmark_dir / "synthetic_test_cases.npy"), allow_pickle=True)
    
    results = []
    
    for tc in test_cases:
        pred_class = tc['pred_class']
        expected = tc['expected']
        
        pred_intervals = SequenceDecoder.decode(pred_class)
        
        metrics = evaluate_sample(pred_intervals, expected, tc['difficulty'])
        metrics['name'] = tc['name']
        metrics['description'] = tc['description']
        metrics['difficulty'] = tc['difficulty']
        metrics['expected'] = expected
        metrics['predicted'] = pred_intervals
        metrics['passed'] = pred_intervals == expected
        
        results.append(metrics)
    
    return results


def evaluate_real_benchmark():
    benchmark_dir = PROJECT_ROOT / "tests" / "benchmark"
    benchmark_path = benchmark_dir / "benchmark_data.npy"
    
    if not benchmark_path.exists():
        print("[WARN] 基准数据文件不存在")
        return []
    
    benchmark_data = np.load(str(benchmark_path), allow_pickle=True)
    
    results = []
    
    for item in benchmark_data:
        if 'pred_class' not in item or 'gt_intervals' not in item:
            continue
        
        pred_class = item['pred_class']
        pred_prob = item.get('pred_prob', None)
        gt_intervals = item['gt_intervals']
        difficulty = item.get('difficulty', 'simple')
        
        pred_intervals = SequenceDecoder.decode(pred_class, pred_prob)
        
        metrics = evaluate_sample(pred_intervals, gt_intervals, difficulty)
        metrics['line_id'] = item.get('line_id', '')
        metrics['difficulty'] = difficulty
        metrics['num_chars'] = item.get('num_chars', len(gt_intervals))
        
        results.append(metrics)
    
    return results


def aggregate_results(results):
    if len(results) == 0:
        return {}
    
    agg = {}
    
    agg['total_samples'] = len(results)
    
    for metric in ['char_count_acc', 'boundary_mae', 'over_seg', 'under_seg', 'shared_boundary_rec', 'iou', 'weighted_score']:
        values = [r[metric] for r in results]
        agg[f'{metric}_mean'] = np.mean(values)
        agg[f'{metric}_std'] = np.std(values)
        agg[f'{metric}_min'] = np.min(values)
        agg[f'{metric}_max'] = np.max(values)
    
    difficulty_groups = {}
    for r in results:
        d = r['difficulty']
        if d not in difficulty_groups:
            difficulty_groups[d] = []
        difficulty_groups[d].append(r)
    
    agg['difficulty_breakdown'] = {}
    for d, group in difficulty_groups.items():
        agg['difficulty_breakdown'][d] = {
            'count': len(group),
            'weighted_score_mean': np.mean([r['weighted_score'] for r in group])
        }
    
    agg['overall_score'] = np.mean([r['weighted_score'] for r in results])
    
    return agg


def print_results_summary(agg, title="评估结果"):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    print(f"\n总样本数: {agg['total_samples']}")
    print(f"总体加权分数: {agg['overall_score']:.4f}")
    
    print(f"\n各维度指标:")
    metrics = [
        ('char_count_acc', '字符数量准确率'),
        ('boundary_mae', '边界位置准确度'),
        ('over_seg', '过分割抵抗率'),
        ('under_seg', '欠分割抵抗率'),
        ('shared_boundary_rec', '共享边界识别率'),
        ('iou', '区间IoU'),
        ('weighted_score', '加权分数')
    ]
    
    for metric, name in metrics:
        mean = agg[f'{metric}_mean']
        std = agg[f'{metric}_std']
        print(f"  {name}: {mean:.4f} ± {std:.4f}")
    
    print(f"\n难度分布:")
    for d, stats in agg['difficulty_breakdown'].items():
        weight = DIFFICULTY_WEIGHTS[d]
        print(f"  {d} (权重×{weight}): {stats['count']} 样本, 加权分数: {stats['weighted_score_mean']:.4f}")


def print_synthetic_summary(results):
    print(f"\n{'='*60}")
    print("合成测试用例结果")
    print(f"{'='*60}")
    
    difficulty_counts = {'simple': 0, 'medium': 0, 'hard': 0}
    passed_counts = {'simple': 0, 'medium': 0, 'hard': 0}
    
    for r in results:
        d = r['difficulty']
        difficulty_counts[d] += 1
        if r['passed']:
            passed_counts[d] += 1
    
    print(f"\n按难度统计:")
    for d in ['simple', 'medium', 'hard']:
        total = difficulty_counts[d]
        passed = passed_counts[d]
        rate = passed / total if total > 0 else 0
        print(f"  {d}: {passed}/{total} 通过 ({rate:.1%})")
    
    failed = [r for r in results if not r['passed']]
    if failed:
        print(f"\n失败用例:")
        for r in failed:
            print(f"  {r['name']} ({r['description']}):")
            print(f"    期望: {r['expected']}")
            print(f"    实际: {r['predicted']}")


def run_full_evaluation():
    print("=" * 60)
    print("后处理逻辑回归测试")
    print("=" * 60)
    
    print("\n[步骤1] 评估合成测试用例")
    synthetic_results = evaluate_synthetic_test_cases()
    synthetic_agg = aggregate_results(synthetic_results)
    print_synthetic_summary(synthetic_results)
    
    print("\n[步骤2] 评估真实标注数据")
    real_results = evaluate_real_benchmark()
    real_agg = aggregate_results(real_results)
    print_results_summary(real_agg, "真实数据评估结果")
    
    print("\n" + "=" * 60)
    
    all_results = synthetic_results + real_results
    overall_agg = aggregate_results(all_results)
    
    print(f"\n综合评估:")
    print(f"  合成测试: {synthetic_agg['overall_score']:.4f}")
    print(f"  真实数据: {real_agg['overall_score']:.4f}")
    print(f"  综合分数: {overall_agg['overall_score']:.4f}")
    
    passed_all_synthetic = all(r['passed'] for r in synthetic_results)
    
    if passed_all_synthetic:
        print("\n✅ 所有合成测试用例通过！")
    else:
        print("\n❌ 部分合成测试用例失败！")
    
    if overall_agg['overall_score'] >= 0.8:
        print("✅ 综合分数达标（≥0.8）")
    else:
        print("⚠️ 综合分数未达标（<0.8），需要优化后处理逻辑")
    
    return {
        'synthetic': synthetic_agg,
        'real': real_agg,
        'overall': overall_agg,
        'synthetic_passed': passed_all_synthetic
    }


if __name__ == "__main__":
    results = run_full_evaluation()
    
    sys.exit(0 if results['synthetic_passed'] and results['overall']['overall_score'] >= 0.8 else 1)