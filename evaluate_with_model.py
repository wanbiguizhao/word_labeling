#!/usr/bin/env python3
"""
使用指定模型对标注数据进行评估

用法:
    python evaluate_with_model.py --model-path models/char_segment_1d_unet_final_0727.pth
"""
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_model.models.unet1d import UNet1D
from ai_model.data.dataset import FeatureExtractor
from ai_model.inference.infer import CharSegmentPredictor
from evaluate_model import (
    evaluate_sample, 
    aggregate_results, 
    print_results_summary,
    compute_composite_score,
    merge_fragments,
    _compute_iou_distribution,
    _compute_char_count_diff_distribution,
    _compute_problem_distribution
)


def load_annotation_data(annotation_dirs: list) -> dict:
    """
    加载所有标注数据
    
    Returns:
        annotations: {line_id: {'chars': [(start, end), ...], 'image_path': str}}
    """
    annotations = {}
    
    for ann_dir in annotation_dirs:
        if not ann_dir.exists():
            continue
            
        for anno_file in ann_dir.glob("*.json"):
            try:
                with open(anno_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                line_id = data.get('line_id', anno_file.stem)
                chars = [(c['col_start'], c['col_end']) for c in data.get('chars', [])]
                annotations[line_id] = {
                    'chars': chars,
                    'image_path': data.get('image_name', f"{line_id}.png")
                }
            except Exception as e:
                print(f"[WARN] 加载 {anno_file.name} 失败: {e}")
    
    return annotations


def evaluate_and_aggregate(predictor, line_paths_with_chars):
    """
    对每个样本评估，返回 results 列表和 metrics_list

    Args:
        predictor: 已加载的模型
        line_paths_with_chars: [(line_id, image_path, gt_chars), ...]
    """
    results = []
    metrics_list = []
    failed_count = 0
    
    for line_id, line_path, gt_chars in line_paths_with_chars:
        img = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] 无法读取图像: {line_id}")
            failed_count += 1
            continue
        try:
            intervals, _, _, _ = predictor.predict(img)
        except Exception as e:
            print(f"[WARN] 推理失败 {line_id}: {e}")
            failed_count += 1
            continue
        
        pred_chars = [(s, e) for s, e in intervals]
        
        eval_result = evaluate_sample(pred_chars, gt_chars)
        eval_result['line_id'] = line_id
        eval_result['pred_char_count'] = len(pred_chars)
        eval_result['gt_char_count'] = len(gt_chars)
        
        metrics_dict = eval_result['metrics']
        metrics_dict['line_id'] = line_id
        metrics_dict['pred_char_count'] = len(pred_chars)
        metrics_dict['gt_char_count'] = len(gt_chars)
        
        results.append(eval_result)
        metrics_list.append(metrics_dict)
    
    return results, metrics_list, failed_count


def aggregate_with_distributions(results, metrics_list):
    """聚合结果 + 分布统计 + 综合评分"""
    if not results:
        return {}
    aggregated = aggregate_results(results)
    aggregated['composite_score'] = compute_composite_score(aggregated)
    aggregated['iou_distribution'] = _compute_iou_distribution(metrics_list)
    aggregated['count_diff_distribution'] = _compute_char_count_diff_distribution(metrics_list)
    aggregated['samples_with_problems'] = _compute_problem_distribution(metrics_list)
    return aggregated


def print_comparison(agg_baseline, agg_merged):
    """
    打印合并前/合并后的指标对比

    Returns:
        Dict: 对比结果
    """
    print(f"\n{'='*80}")
    print("后处理合并对比：原始 vs 合并碎片")
    print(f"{'='*80}")

    # 需要对比的指标
    metrics = [
        ('avg_precision', 'Precision'),
        ('avg_recall', 'Recall'),
        ('avg_f1', 'F1 Score'),
        ('avg_iou', 'IoU'),
        ('avg_matched_iou', '匹配对IoU'),
        ('char_count_match_rate', '数量匹配率'),
        ('avg_over_seg_ratio', '过分割比例'),
        ('avg_under_seg_ratio', '欠分割比例'),
        ('avg_width_error_ratio', '宽度错误比例'),
        ('avg_boundary_acc', '边界准确率'),
        ('avg_mae_boundary', '边界MAE(px)'),
    ]
    # 计数字段
    count_metrics = [
        ('total_pred_count', '预测字符总数'),
        ('total_correct_pred', '正确预测'),
        ('total_half_char_pred', '半个字符'),
        ('total_dirty_spot_pred', '脏点'),
        ('total_sliver_pred', '碎片'),
        ('total_extra_pred', '多余'),
        ('total_merger_pred', '合并型'),
        ('total_correct_gt', '标注正确'),
        ('total_over_segmented_gt', '被拆分'),
        ('total_merged_gt', '被合并'),
        ('total_missed_gt', '漏检'),
    ]

    print(f"\n{'指标':<22} {'原始':>10} {'合并后':>10} {'变化':>10} {'好坏':>6}")
    print(f"{'-'*62}")
    
    comparison = {'improved': [], 'worsened': [], 'unchanged': []}

    for key, name in metrics:
        v0 = agg_baseline.get(key, 0)
        v1 = agg_merged.get(key, 0)
        delta = v1 - v0
        # 变化方向：这些指标越高越好（MAE越低越好）
        better = '↑' if delta > 1e-6 else ('↓' if delta < -1e-6 else '=')
        # 判断好坏：MAE和错误比例越低越好
        lower_is_better = key in ('avg_over_seg_ratio', 'avg_under_seg_ratio', 'avg_width_error_ratio', 'avg_mae_boundary')
        if abs(delta) < 1e-6:
            mark = '—'
            comparison['unchanged'].append(key)
        else:
            good = (delta < 0) if lower_is_better else (delta > 0)
            mark = '✓好' if good else '✗坏'
            comparison['improved' if good else 'worsened'].append(key)
        print(f"{name:<22} {v0:>10.4f} {v1:>10.4f} {delta:>+10.4f} {mark:>6}")

    print(f"\n{'计数指标':<22} {'原始':>10} {'合并后':>10} {'变化':>10}")
    print(f"{'-'*52}")
    for key, name in count_metrics:
        v0 = agg_baseline.get(key, 0)
        v1 = agg_merged.get(key, 0)
        delta = v1 - v0
        print(f"{name:<22} {v0:>10} {v1:>10} {delta:>+10}")

    # 综合评分对比
    cs0 = agg_baseline.get('composite_score', {}).get('total_score', 0)
    cs1 = agg_merged.get('composite_score', {}).get('total_score', 0)
    print(f"\n综合评分: {cs0:.2f} → {cs1:.2f} ({cs1 - cs0:+.2f})")

    comparison['composite_baseline'] = cs0
    comparison['composite_merged'] = cs1
    
    return comparison


def run_model_on_annotations(
    model_path: Path,
    annotations_dirs: list,
    data_base_path: Path,
    output_path: Path,
    post_process: bool = False,
    pp_width_ratio: float = 0.5,
    pp_strategy: str = 'nearest',
    pp_max_gap: int = 0
):
    """
    使用模型对标注数据进行推理并评估

    Args:
        model_path: 模型权重路径
        annotations_dirs: 标注数据目录列表
        data_base_path: 数据基础目录（包含lines/、rule_jsons/等）
        output_path: 输出结果JSON路径
        post_process: 是否启用后处理合并碎片
        pp_width_ratio: 窄碎片判定阈值（宽度 < 中位宽 × ratio）
        pp_strategy: 合并策略 ('nearest' | 'larger' | 'left' | 'right')
        pp_max_gap: 合并最大允许间隙（像素），0表示不限制
    """
    print("=" * 80)
    print("模型评估：使用指定模型对标注数据进行推理")
    if post_process:
        print(f"后处理：合并碎片 (width_ratio={pp_width_ratio}, strategy={pp_strategy}, max_gap={pp_max_gap})")
    print("=" * 80)
    
    # 1. 加载标注数据
    print("\n[1/4] 加载标注数据...")
    annotations = load_annotation_data(annotations_dirs)
    print(f"  加载了 {len(annotations)} 条标注数据")
    
    # 2. 加载模型
    print("\n[2/4] 加载模型...")
    predictor = CharSegmentPredictor(model_path)
    print(f"  模型加载成功: {model_path.name}")
    print(f"  设备: {predictor.device}")
    
    # 3. 收集所有行图像路径
    print("\n[3/4] 收集行图像路径...")
    lines_dir = data_base_path / "lines"
    line_samples = []  # [(line_id, line_path, gt_chars)]
    missing_count = 0
    
    for line_id, ann_data in annotations.items():
        image_name = ann_data.get('image_path', f"{line_id}.png")
        line_path = lines_dir / image_name
        if not line_path.exists():
            line_path = lines_dir / f"{line_id}.png"
        if not line_path.exists():
            print(f"[WARN] 行图像不存在: {line_id}")
            missing_count += 1
            continue
        line_samples.append((line_id, line_path, ann_data['chars']))
    
    print(f"  收集到 {len(line_samples)} 条可评估样本, 缺少图像 {missing_count} 条")
    if not line_samples:
        print("[ERROR] 没有可评估的样本")
        return
    
    # 4. 推理并评估（原始）
    print("\n[4/5] 原始预测评估...")
    results_base, metrics_base, failed_base = evaluate_and_aggregate(
        predictor, line_samples
    )
    print(f"  推理完成: 成功 {len(results_base)} 条, 失败 {failed_base} 条")
    agg_base = aggregate_with_distributions(results_base, metrics_base)
    
    if post_process:
        # 5. 推理并评估（合并碎片后）
        print("\n[5/5] 后处理合并碎片评估...")
        # 复用推理结果：先对每个样本做推理，再分别评估原始与合并
        results_merged, metrics_merged, failed_merged = evaluate_with_merge(
            predictor, line_samples,
            width_ratio=pp_width_ratio,
            strategy=pp_strategy,
            max_gap=pp_max_gap if pp_max_gap > 0 else None
        )
        print(f"  合并评估完成: 成功 {len(results_merged)} 条, 失败 {failed_merged} 条")
        agg_merged = aggregate_with_distributions(results_merged, metrics_merged)
        
        # 打印对比
        comparison = print_comparison(agg_base, agg_merged)
        
        # 打印详细结果
        print_results_summary(agg_base, f"模型评测结果(原始): {model_path.name}")
        print_results_summary(agg_merged, f"模型评测结果(合并碎片后): {model_path.name}")
        
        # 保存结果
        output = {
            'model_path': str(model_path),
            'post_process': {
                'enabled': True,
                'width_ratio': pp_width_ratio,
                'strategy': pp_strategy,
                'max_gap': pp_max_gap,
            },
            'total_samples': len(results_base),
            'failed_samples': failed_base,
            'comparison': comparison,
            'aggregated_baseline': agg_base,
            'aggregated_merged': agg_merged,
            'detailed': results_base,
            'detailed_merged': results_merged,
        }
    else:
        # 无后处理，只输出原始结果
        print_results_summary(agg_base, f"模型评测结果: {model_path.name}")
        output = {
            'model_path': str(model_path),
            'post_process': {'enabled': False},
            'total_samples': len(results_base),
            'failed_samples': failed_base,
            'aggregated': agg_base,
            'detailed': results_base,
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存: {output_path}")
    print("=" * 80)


def evaluate_with_merge(predictor, line_samples, width_ratio=0.5, strategy='nearest', max_gap=None):
    """
    推理后先合并碎片，再评估

    Args:
        predictor: 已加载的模型
        line_samples: [(line_id, line_path, gt_chars), ...]
        width_ratio: 窄碎片判定阈值
        strategy: 合并策略
        max_gap: 最大允许间隙

    Returns:
        (results, metrics_list, failed_count)
    """
    results = []
    metrics_list = []
    failed_count = 0
    
    for line_id, line_path, gt_chars in line_samples:
        img = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            failed_count += 1
            continue
        try:
            intervals, _, _, _ = predictor.predict(img)
        except Exception as e:
            print(f"[WARN] 推理失败 {line_id}: {e}")
            failed_count += 1
            continue
        
        # 合并碎片后处理
        merged_chars = merge_fragments(
            [(s, e) for s, e in intervals],
            width_ratio=width_ratio,
            strategy=strategy,
            max_gap=max_gap
        )
        
        eval_result = evaluate_sample(merged_chars, gt_chars)
        eval_result['line_id'] = line_id
        eval_result['pred_char_count'] = len(merged_chars)
        eval_result['gt_char_count'] = len(gt_chars)
        
        metrics_dict = eval_result['metrics']
        metrics_dict['line_id'] = line_id
        metrics_dict['pred_char_count'] = len(merged_chars)
        metrics_dict['gt_char_count'] = len(gt_chars)
        
        results.append(eval_result)
        metrics_list.append(metrics_dict)
    
    return results, metrics_list, failed_count


def main():
    parser = argparse.ArgumentParser(description='使用指定模型对标注数据进行评估')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出JSON文件路径')
    parser.add_argument('--data-base', type=str, 
                        default=str(Path(__file__).parent / "datahome"),
                        help='数据基础目录')
    parser.add_argument('--annotations-dir', type=str, action='append',
                        help='标注数据目录（可多次指定）')
    
    # 后处理合并碎片配置
    parser.add_argument('--post-process', action='store_true',
                        help='启用后处理合并碎片，并对比合并前后指标')
    parser.add_argument('--pp-width-ratio', type=float, default=0.5,
                        help='窄碎片判定阈值：宽度 < 行中位宽 × ratio（默认0.5）')
    parser.add_argument('--pp-strategy', type=str, default='nearest',
                        choices=['nearest', 'larger', 'left', 'right'],
                        help='合并策略（默认nearest）')
    parser.add_argument('--pp-max-gap', type=int, default=0,
                        help='合并最大允许间隙（像素），0表示不限制')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {model_path}")
        return
    
    data_base_path = Path(args.data_base)
    
    # 默认标注目录
    if args.annotations_dir:
        annotations_dirs = [Path(d) for d in args.annotations_dir]
    else:
        annotations_dirs = [
            data_base_path / "project/proj_20260725_225531/annotations",
            data_base_path / "project/proj_20260706_230417/annotations",
        ]
    
    # 输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        model_name = model_path.stem
        suffix = "_merge" if args.post_process else ""
        output_path = Path(__file__).parent / f"eval_{model_name}{suffix}.json"
    
    run_model_on_annotations(
        model_path=model_path,
        annotations_dirs=annotations_dirs,
        data_base_path=data_base_path,
        output_path=output_path,
        post_process=args.post_process,
        pp_width_ratio=args.pp_width_ratio,
        pp_strategy=args.pp_strategy,
        pp_max_gap=args.pp_max_gap
    )


if __name__ == "__main__":
    main()
