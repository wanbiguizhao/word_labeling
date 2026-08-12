#!/usr/bin/env python3
"""
评估规则分割结果在标注数据集上的表现，并对比不同后处理链配置的效果

流程：
1. 加载标注数据（GT）和 rule_json（原始规则切割结果）
2. 评估规则基线
3. 对规则结果应用后处理链配置，再次评估
4. 输出对比表

用法:
    # 对比 baseline vs merge_fragments_gap3
    python evaluate_rule_merge.py --config merge_fragments_gap3

    # 指定多个参数
    python evaluate_rule_merge.py --config merge_fragments_gap3 --output eval_result.json

    # 使用绝对路径
    python evaluate_rule_merge.py --config /path/to/config.json
"""
import json
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_model import (
    evaluate_sample,
    aggregate_results,
    print_results_summary,
    compute_composite_score,
    _compute_iou_distribution,
    _compute_char_count_diff_distribution,
    _compute_problem_distribution,
)
from image_tools.postprocess_chain import (
    resolve_config_path,
    load_config as load_pp_config,
    run_chain as run_pp_chain,
    get_default_config as get_default_pp_config,
)

project_root = PROJECT_ROOT
rule_jsons_dir = project_root / "datahome/rule_jsons"
annotation_dirs = [
    project_root / "datahome/project/proj_20260725_225531/annotations",
    project_root / "datahome/project/proj_20260706_230417/annotations",
]


def load_rule_and_annotation():
    """加载所有标注数据及其对应的 rule_json"""
    samples = []  # [(line_id, rule_chars, gt_chars)]
    missing_rule = 0
    for ann_dir in annotation_dirs:
        if not ann_dir.exists():
            continue
        for anno_file in ann_dir.glob("*.json"):
            try:
                with open(anno_file, 'r', encoding='utf-8') as f:
                    anno = json.load(f)
                line_id = anno.get('line_id', anno_file.stem)
                gt_chars = [(c['col_start'], c['col_end']) for c in anno.get('chars', [])]

                rule_file = rule_jsons_dir / f"{line_id}_rule.json"
                if not rule_file.exists():
                    missing_rule += 1
                    continue
                with open(rule_file, 'r', encoding='utf-8') as f:
                    rule = json.load(f)
                rule_chars = [(c['col_start'], c['col_end']) for c in rule.get('chars', [])]

                samples.append((line_id, rule_chars, gt_chars))
            except Exception as e:
                print(f"[WARN] 处理 {anno_file.name} 失败: {e}")
    print(f"  匹配到 {len(samples)} 条标注数据（缺失rule_json: {missing_rule} 条）")
    return samples


def evaluate_samples(samples):
    """对每个样本评估，返回 results 和 metrics_list"""
    results = []
    metrics_list = []
    for line_id, pred_chars, gt_chars in samples:
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
    return results, metrics_list


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


def apply_chain_to_samples(samples, pp_config):
    """对规则结果应用后处理链，返回处理后的 samples"""
    merged_samples = []
    for line_id, rule_chars, gt_chars in samples:
        # 转换为 dict 列表供处理链使用
        char_dicts = [{"col_start": s, "col_end": e, "width": e - s} for s, e in rule_chars]
        context = {"line_id": line_id, "image_width": 0, "image_height": 0}
        processed = run_pp_chain(char_dicts, pp_config, context, verbose=False)
        # 转回元组列表
        processed_tuples = [(c["col_start"], c["col_end"]) for c in processed]
        merged_samples.append((line_id, processed_tuples, gt_chars))
    return merged_samples


def print_comparison(agg_base, agg_merged, config_name):
    """打印规则基线 vs 合并后的对比表"""
    print(f"\n{'='*80}")
    print(f"规则结果对比：基线 vs {config_name}")
    print(f"{'='*80}")

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
    count_metrics = [
        ('total_pred_count', '规则字符总数'),
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

    print(f"\n{'指标':<22} {'规则基线':>10} {'合并后':>10} {'变化':>10} {'好坏':>6}")
    print(f"{'-'*62}")
    for key, name in metrics:
        v0 = agg_base.get(key, 0)
        v1 = agg_merged.get(key, 0)
        delta = v1 - v0
        lower_is_better = key in ('avg_over_seg_ratio', 'avg_under_seg_ratio', 'avg_width_error_ratio', 'avg_mae_boundary')
        if abs(delta) < 1e-6:
            mark = '—'
        else:
            good = (delta < 0) if lower_is_better else (delta > 0)
            mark = '✓好' if good else '✗坏'
        print(f"{name:<22} {v0:>10.4f} {v1:>10.4f} {delta:>+10.4f} {mark:>6}")

    print(f"\n{'计数指标':<22} {'规则基线':>10} {'合并后':>10} {'变化':>10}")
    print(f"{'-'*52}")
    for key, name in count_metrics:
        v0 = agg_base.get(key, 0)
        v1 = agg_merged.get(key, 0)
        delta = v1 - v0
        print(f"{name:<22} {v0:>10} {v1:>10} {delta:>+10}")

    cs0 = agg_base.get('composite_score', {}).get('total_score', 0)
    cs1 = agg_merged.get('composite_score', {}).get('total_score', 0)
    print(f"\n综合评分: {cs0:.2f} → {cs1:.2f} ({cs1 - cs0:+.2f})")

    return {'composite_baseline': cs0, 'composite_merged': cs1}


def main():
    parser = argparse.ArgumentParser(description='评估规则分割结果 + 后处理链对比')
    parser.add_argument('--config', type=str, default='merge_fragments_gap3',
                        help='后处理链配置名（postprocess_configs/目录下）或绝对路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出JSON文件路径')
    args = parser.parse_args()

    # 加载后处理链配置
    config_path = resolve_config_path(args.config)
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        return
    pp_config = load_pp_config(config_path)
    config_version = pp_config.get('version', args.config)

    print("=" * 80)
    print("规则分割结果评估（标注数据集）")
    print(f"后处理配置: {args.config} (版本: {config_version})")
    print(f"配置描述: {pp_config.get('description', '')}")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/4] 加载标注数据与rule_json...")
    samples = load_rule_and_annotation()
    if not samples:
        print("[ERROR] 没有可评估的样本")
        return

    # 2. 评估规则基线
    print("\n[2/4] 评估规则基线...")
    results_base, metrics_base = evaluate_samples(samples)
    agg_base = aggregate_with_distributions(results_base, metrics_base)
    print(f"  基线评估完成: {len(results_base)} 条")

    # 3. 应用后处理链后评估
    print(f"\n[3/4] 应用后处理链 ({config_version}) 后评估...")
    processed_samples = apply_chain_to_samples(samples, pp_config)
    results_merged, metrics_merged = evaluate_samples(processed_samples)
    agg_merged = aggregate_with_distributions(results_merged, metrics_merged)
    print(f"  后处理评估完成: {len(results_merged)} 条")

    # 4. 输出对比
    print("\n[4/4] 输出对比结果...")
    comparison = print_comparison(agg_base, agg_merged, config_version)

    # 打印详细摘要
    print_results_summary(agg_base, f"规则分割结果(基线)")
    print_results_summary(agg_merged, f"规则分割结果({config_version})")

    # 保存
    output_path = Path(args.output) if args.output else (
        Path(__file__).parent / f"eval_rule_{config_version}.json")
    output = {
        'description': '规则分割结果 vs 后处理链对比评估',
        'postprocess_config': pp_config,
        'total_samples': len(samples),
        'comparison': comparison,
        'aggregated_baseline': agg_base,
        'aggregated_merged': agg_merged,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
