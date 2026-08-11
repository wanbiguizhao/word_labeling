#!/usr/bin/env python3
"""
模型评测脚本（v2 - 基于字符档案的评测体系）

核心改进：
1. build_char_archive() 只调1次greedy_match，所有指标从档案派生
2. 字符级状态分类：
   GT:  correct / half_char / over_segmented / merged / missed / borderline
   Pred: correct / half_char / dirty_spot / sliver / extra
3. 标准P/R/F1指标
4. 字符级跨模型对比

使用方法：
    python evaluate_model.py --project-id proj_20260725_225531 --model-version v0724
    python evaluate_model.py --compare eval_v0724.json eval_v0808.json
    python evaluate_model.py --compare-char eval_v0724.json eval_v0808.json
"""

import json
import argparse
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Optional


# ============================================================
# 数据加载
# ============================================================

def load_annotation(annotation_path: Path) -> Tuple[List[Tuple[int, int]], int]:
    """加载标注数据"""
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chars = []
    total_width = 0
    for char in data.get('chars', []):
        col_start = char.get('col_start', 0)
        col_end = char.get('col_end', 0)
        chars.append((col_start, col_end))
        total_width += col_end - col_start
    return chars, total_width


def load_model_prediction_from_model_jsons(model_path: Path) -> Tuple[List[Tuple[int, int]], int]:
    """从项目model_jsons目录加载模型推理结果"""
    with open(model_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chars = []
    total_width = 0
    for char in data.get('chars', []):
        col_start = char.get('col_start', 0)
        col_end = char.get('col_end', 0)
        chars.append((col_start, col_end))
        total_width += col_end - col_start
    return chars, total_width


def load_model_prediction_from_inference(inference_dir: Path, line_id: str) -> Tuple[Optional[List[Tuple[int, int]]], int]:
    """从推理版本目录加载模型推理结果"""
    inference_jsonl = inference_dir / "inference.jsonl"
    with open(inference_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get('line_id') == line_id:
                    chars = []
                    total_width = 0
                    for start, end in data.get('chars', []):
                        chars.append((start, end))
                        total_width += end - start
                    return chars, total_width
            except Exception:
                continue
    return None, 0


# ============================================================
# 区间运算基础
# ============================================================

def interval_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """计算两个区间的重叠长度"""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """计算两个区间的IoU"""
    inter = interval_overlap(a, b)
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(pred_chars: List[Tuple[int, int]], gt_chars: List[Tuple[int, int]], iou_threshold: float = 0.3) -> Tuple[Dict[int, int], Dict[int, int], List[int], List[int]]:
    """贪心一对一匹配（带排序剪枝优化）"""
    pred_to_gt = {}
    gt_to_pred = {}
    unmatched_pred = set(range(len(pred_chars)))
    unmatched_gt = set(range(len(gt_chars)))

    if len(pred_chars) == 0 or len(gt_chars) == 0:
        return pred_to_gt, gt_to_pred, list(unmatched_pred), list(unmatched_gt)

    # 排序后利用区间单调性剪枝
    pred_order = sorted(range(len(pred_chars)), key=lambda k: pred_chars[k][0])
    gt_order = sorted(range(len(gt_chars)), key=lambda k: gt_chars[k][0])

    iou_matrix = np.zeros((len(pred_chars), len(gt_chars)))
    for i in pred_order:
        pred_char = pred_chars[i]
        for j in gt_order:
            gt_char = gt_chars[j]
            if pred_char[1] <= gt_char[0]:
                break
            if pred_char[0] >= gt_char[1]:
                continue
            iou_matrix[i, j] = iou(pred_char, gt_char)

    while True:
        max_iou = np.max(iou_matrix)
        if max_iou < iou_threshold or max_iou == 0:
            break
        pred_idx, gt_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        pred_idx = int(pred_idx)
        gt_idx = int(gt_idx)
        if pred_idx in unmatched_pred and gt_idx in unmatched_gt:
            pred_to_gt[pred_idx] = gt_idx
            gt_to_pred[gt_idx] = pred_idx
            unmatched_pred.discard(pred_idx)
            unmatched_gt.discard(gt_idx)
            iou_matrix[pred_idx, :] = 0
            iou_matrix[:, gt_idx] = 0
        else:
            iou_matrix[pred_idx, gt_idx] = 0

    return pred_to_gt, gt_to_pred, list(unmatched_pred), list(unmatched_gt)


# ============================================================
# 核心：字符档案构建
# ============================================================

# 状态阈值
IOU_CORRECT = 0.7       # IoU >= 0.7 → correct
IOU_HALF_CHAR = 0.3     # 0.3 <= IoU < 0.7 → half_char
IOU_MATCH_THRESHOLD = 0.1  # greedy_match 门限
DIRTY_SPOT_WIDTH_RATIO = 0.2  # 宽度 < 中位数 × 20% → 候选脏点
DIRTY_SPOT_OVERLAP = 0.1     # 重叠率 < 10% → 脏点
SLIVER_OVERLAP = 0.3         # 重叠率 >= 30% → 碎片


def build_char_archive(pred_chars: List[Tuple[int, int]], gt_chars: List[Tuple[int, int]]) -> Dict:
    """
    构建字符级档案：只调1次greedy_match，建立每个GT/Pred的完整状态信息。

    GT状态：
      correct        - 贪心匹配成功 且 IoU >= 0.7
      half_char      - 贪心匹配成功 且 0.3 <= IoU < 0.7 且 overlapping_preds < 2
      over_segmented - 贪心匹配成功 且 0.3 <= IoU < 0.7 且 overlapping_preds >= 2
      borderline     - 贪心匹配成功 且 IoU < 0.3
      merged         - 贪心匹配失败 但 有pred覆盖它（被旁边的pred吞并）
      missed         - 贪心匹配失败 且 无pred重叠（真漏检）

    Pred状态：
      correct    - 贪心匹配成功 且 IoU >= 0.7
      half_char  - 贪心匹配成功 且 IoU < 0.7
      dirty_spot - 未匹配 + 宽度 < 中位数×20% + 重叠率 < 10%
      sliver     - 未匹配 + 与某GT重叠率 >= 30%（从汉字身上拆下的碎片）
      extra      - 未匹配 + 其他
    """
    # 1. 贪心匹配（只调1次）
    pred_to_gt, gt_to_pred, unmatched_pred, unmatched_gt = greedy_match(
        pred_chars, gt_chars, iou_threshold=IOU_MATCH_THRESHOLD
    )

    # 2. 计算overlapping关系（哪些pred覆盖了哪些gt）
    gt_overlapping_preds = {}  # {gt_idx: [pred_idx, ...]}
    pred_overlapping_gts = {}  # {pred_idx: [gt_idx, ...]}
    for gt_idx, gt_char in enumerate(gt_chars):
        for pred_idx, pred_char in enumerate(pred_chars):
            if interval_overlap(pred_char, gt_char) > 0:
                gt_overlapping_preds.setdefault(gt_idx, []).append(pred_idx)
                pred_overlapping_gts.setdefault(pred_idx, []).append(gt_idx)

    # 3. 计算GT宽度中位数（作为参考宽度）
    gt_widths = [e - s for s, e in gt_chars] if gt_chars else [1]
    gt_median_width = max(float(np.median(gt_widths)), 1.0)

    # 4. 构建GT档案
    gt_archive = []
    for gt_idx, (gt_start, gt_end) in enumerate(gt_chars):
        gt_width = gt_end - gt_start
        overlapping_preds = gt_overlapping_preds.get(gt_idx, [])

        entry = {
            'gt_idx': gt_idx,
            'interval': [gt_start, gt_end],
            'width': gt_width,
            'overlapping_preds': overlapping_preds,
            'overlapping_count': len(overlapping_preds),
        }

        if gt_idx in gt_to_pred:
            # 贪心匹配成功
            pred_idx = gt_to_pred[gt_idx]
            iou_val = iou(pred_chars[pred_idx], gt_chars[gt_idx])
            pred_start, pred_end = pred_chars[pred_idx]

            entry['matched_pred_idx'] = pred_idx
            entry['matched_iou'] = round(iou_val, 4)
            entry['start_error'] = abs(pred_start - gt_start)
            entry['end_error'] = abs(pred_end - gt_end)
            entry['width_diff'] = abs((pred_end - pred_start) - gt_width)

            if iou_val >= IOU_CORRECT:
                entry['status'] = 'correct'
            elif iou_val >= IOU_HALF_CHAR:
                if len(overlapping_preds) >= 2:
                    entry['status'] = 'over_segmented'
                else:
                    entry['status'] = 'half_char'
            else:
                entry['status'] = 'borderline'
        else:
            # 贪心匹配失败
            entry['matched_pred_idx'] = None
            entry['matched_iou'] = 0.0
            entry['start_error'] = None
            entry['end_error'] = None
            entry['width_diff'] = None

            if len(overlapping_preds) > 0:
                entry['status'] = 'merged'
            else:
                entry['status'] = 'missed'

        gt_archive.append(entry)

    # 5. 构建Pred档案
    pred_archive = []
    for pred_idx, (pred_start, pred_end) in enumerate(pred_chars):
        pred_width = pred_end - pred_start
        overlapping_gts = pred_overlapping_gts.get(pred_idx, [])

        entry = {
            'pred_idx': pred_idx,
            'interval': [pred_start, pred_end],
            'width': pred_width,
            'overlapping_gts': overlapping_gts,
            'overlapping_count': len(overlapping_gts),
        }

        if pred_idx in pred_to_gt:
            # 贪心匹配成功
            gt_idx = pred_to_gt[pred_idx]
            iou_val = iou(pred_chars[pred_idx], gt_chars[gt_idx])

            entry['matched_gt_idx'] = gt_idx
            entry['matched_iou'] = round(iou_val, 4)

            if iou_val >= IOU_CORRECT:
                entry['status'] = 'correct'
            else:
                entry['status'] = 'half_char'
        else:
            # 未匹配：细分分类
            entry['matched_gt_idx'] = None
            entry['matched_iou'] = 0.0

            # 计算与所有GT的最大重叠率
            max_overlap = 0.0
            for gt_char in gt_chars:
                overlap = interval_overlap(pred_chars[pred_idx], gt_char)
                ratio = overlap / pred_width if pred_width > 0 else 0
                if ratio > max_overlap:
                    max_overlap = ratio

            width_ratio = pred_width / gt_median_width
            entry['width_ratio_to_median'] = round(width_ratio, 3)
            entry['max_overlap_ratio'] = round(max_overlap, 3)

            if width_ratio < DIRTY_SPOT_WIDTH_RATIO and max_overlap < DIRTY_SPOT_OVERLAP:
                entry['status'] = 'dirty_spot'
            elif max_overlap >= SLIVER_OVERLAP:
                entry['status'] = 'sliver'
            else:
                entry['status'] = 'extra'

        pred_archive.append(entry)

    return {
        'gt_archive': gt_archive,
        'pred_archive': pred_archive,
        'gt_median_width': gt_median_width,
        'pred_to_gt': pred_to_gt,
        'gt_to_pred': gt_to_pred,
        'unmatched_pred': unmatched_pred,
        'unmatched_gt': unmatched_gt,
    }


# ============================================================
# 指标计算（全部从archive派生，不重复调greedy_match）
# ============================================================

def _count_states(archive_list: List[Dict]) -> Dict[str, int]:
    """统计各状态数量"""
    states = {}
    for e in archive_list:
        s = e['status']
        states[s] = states.get(s, 0) + 1
    return states


def compute_iou_from_archive(archive: Dict, pred_chars: List, gt_chars: List) -> float:
    """从档案计算整体像素级IoU"""
    if len(pred_chars) == 0 and len(gt_chars) == 0:
        return 1.0
    if len(pred_chars) == 0 or len(gt_chars) == 0:
        return 0.0

    pred_to_gt = archive['pred_to_gt']
    unmatched_pred = archive['unmatched_pred']
    unmatched_gt = archive['unmatched_gt']

    total_intersection = 0
    total_union = 0

    for pred_idx, gt_idx in pred_to_gt.items():
        pred_char = pred_chars[pred_idx]
        gt_char = gt_chars[gt_idx]
        inter = interval_overlap(pred_char, gt_char)
        union = (pred_char[1] - pred_char[0]) + (gt_char[1] - gt_char[0]) - inter
        total_intersection += inter
        total_union += union

    for gt_idx in unmatched_gt:
        total_union += gt_chars[gt_idx][1] - gt_chars[gt_idx][0]

    for pred_idx in unmatched_pred:
        total_union += pred_chars[pred_idx][1] - pred_chars[pred_idx][0]

    return total_intersection / total_union if total_union > 0 else 0.0


def evaluate_sample(pred_chars: List[Tuple[int, int]], gt_chars: List[Tuple[int, int]]) -> Dict:
    """评测单个样本：先建档案，再派生所有指标"""
    archive = build_char_archive(pred_chars, gt_chars)
    gt_arch = archive['gt_archive']
    pred_arch = archive['pred_archive']

    total_gt = len(gt_chars)
    total_pred = len(pred_chars)

    # 状态计数
    gt_states = _count_states(gt_arch)
    pred_states = _count_states(pred_arch)

    correct_gt = gt_states.get('correct', 0)
    correct_pred = pred_states.get('correct', 0)
    missed_gt = gt_states.get('missed', 0)
    merged_gt = gt_states.get('merged', 0)
    over_seg_gt = gt_states.get('over_segmented', 0)
    half_char_gt = gt_states.get('half_char', 0)
    borderline_gt = gt_states.get('borderline', 0)
    dirty_spot_pred = pred_states.get('dirty_spot', 0)
    sliver_pred = pred_states.get('sliver', 0)
    extra_pred = pred_states.get('extra', 0)
    half_char_pred = pred_states.get('half_char', 0)

    # 统计merger pred（覆盖了多个GT的pred）
    merger_pred = sum(1 for e in pred_arch if len(e.get('overlapping_gts', [])) >= 2)

    # P / R / F1
    precision = correct_pred / max(total_pred, 1)
    recall = correct_gt / max(total_gt, 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # IoU
    overall_iou = compute_iou_from_archive(archive, pred_chars, gt_chars)

    # 匹配对的IoU均值
    matched_ious = [e['matched_iou'] for e in gt_arch if e['matched_iou'] > 0]
    avg_matched_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    # 边界MAE
    start_errs = [e['start_error'] for e in gt_arch if e['start_error'] is not None]
    end_errs = [e['end_error'] for e in gt_arch if e['end_error'] is not None]
    mae_start = float(np.mean(start_errs)) if start_errs else 0.0
    mae_end = float(np.mean(end_errs)) if end_errs else 0.0
    mae_boundary = float(np.mean(start_errs + end_errs)) if (start_errs or end_errs) else 0.0

    # 含惩罚的MAE
    matched_pairs = len(pred_to_gt) if (pred_to_gt := archive['pred_to_gt']) else 0
    total_penalty = 0.0
    penalty_count = 0
    for pred_idx in archive['unmatched_pred']:
        w = pred_chars[pred_idx][1] - pred_chars[pred_idx][0]
        total_penalty += w * 2
        penalty_count += 2
    for gt_idx in archive['unmatched_gt']:
        w = gt_chars[gt_idx][1] - gt_chars[gt_idx][0]
        total_penalty += w * 2
        penalty_count += 2
    total_err_sum = sum(start_errs) + sum(end_errs) + total_penalty
    total_points = len(start_errs) + len(end_errs) + penalty_count
    mae_with_penalty = total_err_sum / total_points if total_points > 0 else 0.0
    ref_width = archive['gt_median_width']
    boundary_acc = max(0.0, 1.0 - mae_with_penalty / (ref_width * 1.5)) if ref_width > 0 else 0.0

    # 宽度错误
    width_diffs = [e['width_diff'] for e in gt_arch if e['width_diff'] is not None]
    gt_std_width = float(np.std([e - s for s, e in gt_chars])) if gt_chars else 0.0
    width_threshold = max(gt_std_width * 2, 5)
    width_error_count = sum(1 for wd in width_diffs if wd > width_threshold)
    avg_width_diff = float(np.mean(width_diffs)) if width_diffs else 0.0

    # 字符数量
    char_count_match = total_pred == total_gt
    over_seg_count = max(0, total_pred - total_gt)
    under_seg_count = max(0, total_gt - total_pred)

    return {
        'line_id': '',  # 由调用方填充
        'pred_char_count': total_pred,
        'gt_char_count': total_gt,
        'archive': archive,
        'metrics': {
            # 标准指标
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'iou': round(overall_iou, 4),
            'avg_matched_iou': round(avg_matched_iou, 4),
            'char_count_match': char_count_match,

            # GT状态分布
            'gt_states': gt_states,
            'correct_gt': correct_gt,
            'missed_gt': missed_gt,
            'merged_gt': merged_gt,
            'over_segmented_gt': over_seg_gt,
            'half_char_gt': half_char_gt,
            'borderline_gt': borderline_gt,

            # Pred状态分布
            'pred_states': pred_states,
            'correct_pred': correct_pred,
            'half_char_pred': half_char_pred,
            'dirty_spot_pred': dirty_spot_pred,
            'sliver_pred': sliver_pred,
            'extra_pred': extra_pred,
            'merger_pred': merger_pred,

            # 欠分割
            'under_seg_count': under_seg_count,
            'under_seg_ratio': under_seg_count / max(total_gt, 1),

            # 过分割
            'over_seg_count': over_seg_count,
            'over_seg_ratio': over_seg_count / max(total_gt, 1),
            'extra_total': dirty_spot_pred + sliver_pred + extra_pred,
            'extra_ratio': (dirty_spot_pred + sliver_pred + extra_pred) / max(total_pred, 1),

            # 边界MAE
            'mae_start': round(mae_start, 4),
            'mae_end': round(mae_end, 4),
            'mae_boundary': round(mae_boundary, 4),
            'mae_with_penalty': round(mae_with_penalty, 4),
            'boundary_acc': round(boundary_acc, 4),
            'matched_pairs': matched_pairs,
            'start_errors': start_errs,
            'end_errors': end_errs,

            # 宽度
            'width_error_count': width_error_count,
            'width_error_ratio': width_error_count / max(matched_pairs, 1),
            'avg_width_diff': round(avg_width_diff, 4),
            'gt_std_width': round(gt_std_width, 4),
            'gt_median_width': round(ref_width, 4),
        }
    }


# ============================================================
# 综合评分
# ============================================================

def compute_composite_score(agg: Dict) -> Dict:
    """
    计算加权综合评分（0-100分）

    权重设计：
    - F1 Score (30%)：精确率和召回率的调和平均，核心指标
    - 边界准确率 (20%)：边界MAE归一化分数
    - 过分割抵抗率 (15%)：1 - 过分割比例
    - 欠分割抵抗率 (15%)：1 - 欠分割比例（含merged）
    - 脏点抵抗率 (10%)：1 - 脏点比例，数据清洗效果直接体现
    - 宽度准确率 (10%)：1 - 宽度错误比例
    """
    score_breakdown = {}

    # F1 Score
    f1 = agg.get('avg_f1', 0.0)
    score_breakdown['f1'] = {'value': f1, 'weight': 0.30, 'score': f1 * 0.30}

    # 边界准确率
    boundary_acc = agg.get('avg_boundary_acc', 0.0)
    score_breakdown['boundary_acc'] = {'value': boundary_acc, 'weight': 0.20, 'score': boundary_acc * 0.20}

    # 过分割抵抗率
    over_seg_ratio = agg.get('avg_over_seg_ratio', 0.0)
    over_seg_resist = max(0.0, 1.0 - over_seg_ratio)
    score_breakdown['over_seg_resist'] = {'value': over_seg_resist, 'weight': 0.15, 'score': over_seg_resist * 0.15}

    # 欠分割抵抗率（含merged）
    under_seg_ratio = agg.get('avg_under_seg_ratio', 0.0)
    under_seg_resist = max(0.0, 1.0 - under_seg_ratio)
    score_breakdown['under_seg_resist'] = {'value': under_seg_resist, 'weight': 0.15, 'score': under_seg_resist * 0.15}

    # 脏点抵抗率
    dirty_spot_ratio = agg.get('avg_dirty_spot_ratio', 0.0)
    dirty_spot_resist = max(0.0, 1.0 - dirty_spot_ratio)
    score_breakdown['dirty_spot_resist'] = {'value': dirty_spot_resist, 'weight': 0.10, 'score': dirty_spot_resist * 0.10}

    # 宽度准确率
    width_error_ratio = agg.get('avg_width_error_ratio', 0.0)
    width_acc = max(0.0, 1.0 - width_error_ratio)
    score_breakdown['width_acc'] = {'value': width_acc, 'weight': 0.10, 'score': width_acc * 0.10}

    total_score = sum(s['score'] for s in score_breakdown.values()) * 100.0

    return {
        'total_score': round(total_score, 2),
        'breakdown': {k: {**v, 'score': round(v['score'] * 100, 2)} for k, v in score_breakdown.items()}
    }


# ============================================================
# 聚合
# ============================================================

def aggregate_results(results: List[Dict]) -> Dict:
    """聚合所有评测结果"""
    if not results:
        return {}

    n = len(results)
    metrics_list = [r['metrics'] for r in results]

    agg = {
        'total_samples': n,
        'avg_precision': float(np.mean([m['precision'] for m in metrics_list])),
        'avg_recall': float(np.mean([m['recall'] for m in metrics_list])),
        'avg_f1': float(np.mean([m['f1'] for m in metrics_list])),
        'avg_iou': float(np.mean([m['iou'] for m in metrics_list])),
        'avg_matched_iou': float(np.mean([m['avg_matched_iou'] for m in metrics_list])),
        'char_count_match_rate': sum(1 for m in metrics_list if m['char_count_match']) / n,
        'avg_pred_count': float(np.mean([r['pred_char_count'] for r in results])),
        'avg_gt_count': float(np.mean([r['gt_char_count'] for r in results])),

        # GT状态聚合
        'total_correct_gt': sum(m['correct_gt'] for m in metrics_list),
        'total_missed_gt': sum(m['missed_gt'] for m in metrics_list),
        'total_merged_gt': sum(m['merged_gt'] for m in metrics_list),
        'total_over_segmented_gt': sum(m['over_segmented_gt'] for m in metrics_list),
        'total_half_char_gt': sum(m['half_char_gt'] for m in metrics_list),
        'total_borderline_gt': sum(m['borderline_gt'] for m in metrics_list),

        # Pred状态聚合
        'total_correct_pred': sum(m['correct_pred'] for m in metrics_list),
        'total_half_char_pred': sum(m['half_char_pred'] for m in metrics_list),
        'total_dirty_spot_pred': sum(m['dirty_spot_pred'] for m in metrics_list),
        'total_sliver_pred': sum(m['sliver_pred'] for m in metrics_list),
        'total_extra_pred': sum(m['extra_pred'] for m in metrics_list),
        'total_merger_pred': sum(m['merger_pred'] for m in metrics_list),

        # 比率
        'avg_under_seg_ratio': float(np.mean([m['under_seg_ratio'] for m in metrics_list])),
        'avg_over_seg_ratio': float(np.mean([m['over_seg_ratio'] for m in metrics_list])),
        'avg_extra_ratio': float(np.mean([m['extra_ratio'] for m in metrics_list])),
        'avg_dirty_spot_ratio': float(np.mean([m['dirty_spot_pred'] / max(m['matched_pairs'], 1) for m in metrics_list])) if n else 0.0,

        # 边界MAE
        'avg_mae_start': float(np.mean([m['mae_start'] for m in metrics_list])),
        'avg_mae_end': float(np.mean([m['mae_end'] for m in metrics_list])),
        'avg_mae_boundary': float(np.mean([m['mae_boundary'] for m in metrics_list])),
        'avg_mae_with_penalty': float(np.mean([m['mae_with_penalty'] for m in metrics_list])),
        'avg_boundary_acc': float(np.mean([m['boundary_acc'] for m in metrics_list])),
        'total_matched_pairs': sum(m['matched_pairs'] for m in metrics_list),

        # 宽度
        'avg_width_error_ratio': float(np.mean([m['width_error_ratio'] for m in metrics_list])),
        'total_width_error_count': sum(m['width_error_count'] for m in metrics_list),
        'avg_width_diff': float(np.mean([m['avg_width_diff'] for m in metrics_list])),

        # 样本级统计
        'samples_with_missed': sum(1 for m in metrics_list if m['missed_gt'] > 0),
        'samples_with_merged': sum(1 for m in metrics_list if m['merged_gt'] > 0),
        'samples_with_over_seg': sum(1 for m in metrics_list if m['over_segmented_gt'] > 0),
        'samples_with_dirty_spot': sum(1 for m in metrics_list if m['dirty_spot_pred'] > 0),
        'samples_with_sliver': sum(1 for m in metrics_list if m['sliver_pred'] > 0),
        'samples_with_extra': sum(1 for m in metrics_list if m['extra_pred'] > 0),
    }

    # 全局边界误差（跨样本合并）
    all_start_errs = []
    all_end_errs = []
    for m in metrics_list:
        all_start_errs.extend(m.get('start_errors', []))
        all_end_errs.extend(m.get('end_errors', []))

    agg['global_mae_start'] = float(np.mean(all_start_errs)) if all_start_errs else 0.0
    agg['global_mae_end'] = float(np.mean(all_end_errs)) if all_end_errs else 0.0
    agg['global_mae_boundary'] = float(np.mean(all_start_errs + all_end_errs)) if (all_start_errs or all_end_errs) else 0.0
    agg['global_p90_start'] = float(np.percentile(all_start_errs, 90)) if len(all_start_errs) >= 3 else (max(all_start_errs) if all_start_errs else 0.0)
    agg['global_p90_end'] = float(np.percentile(all_end_errs, 90)) if len(all_end_errs) >= 3 else (max(all_end_errs) if all_end_errs else 0.0)
    agg['global_max_start'] = max(all_start_errs) if all_start_errs else 0.0
    agg['global_max_end'] = max(all_end_errs) if all_end_errs else 0.0
    agg['error_distribution'] = _compute_boundary_error_distribution(all_start_errs + all_end_errs)

    # IoU分布
    agg['iou_distribution'] = _compute_iou_distribution(metrics_list)

    # 字符数量差异分布
    agg['char_count_diff_distribution'] = _compute_char_count_diff_distribution(results)

    # 问题类型分布
    agg['problem_distribution'] = _compute_problem_distribution(metrics_list)

    # 综合评分
    agg['composite_score'] = compute_composite_score(agg)

    return agg


def _compute_boundary_error_distribution(all_errors: List[float]) -> Dict:
    """计算边界误差（像素）的区间分布"""
    if not all_errors:
        return {}
    intervals = [
        (0, 1, '<=1px'), (1, 3, '1-3px'), (3, 5, '3-5px'),
        (5, 10, '5-10px'), (10, 20, '10-20px'), (20, float('inf'), '>20px'),
    ]
    dist = {}
    for lo, hi, label in intervals:
        count = sum(1 for e in all_errors if lo <= e < hi)
        dist[label] = {'count': count, 'ratio': count / max(len(all_errors), 1)}
    return dist


def _compute_iou_distribution(metrics_list: List[Dict]) -> Dict:
    """计算IoU区间分布"""
    iou_values = [m['iou'] for m in metrics_list]
    intervals = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    distribution = {}
    for start, end in intervals:
        count = sum(1 for iou in iou_values if start <= iou < end)
        key = f"{start:.1f}-{end:.1f}"
        distribution[key] = {'count': count, 'ratio': count / max(len(iou_values), 1)}
    return distribution


def _compute_char_count_diff_distribution(results: List[Dict]) -> Dict:
    """计算字符数量差异分布"""
    diffs = [r['pred_char_count'] - r['gt_char_count'] for r in results]
    intervals = [(-999, -5), (-5, -1), (-1, 1), (1, 5), (5, 10), (10, 9999)]
    keys = ['<-5', '-5~-1', '0', '1~5', '5~10', '>10']
    distribution = {}
    for (lo, hi), key in zip(intervals, keys):
        count = sum(1 for d in diffs if lo <= d < hi)
        distribution[key] = {'count': count, 'ratio': count / max(len(diffs), 1)}
    distribution['avg_diff'] = float(np.mean(diffs)) if diffs else 0.0
    distribution['std_diff'] = float(np.std(diffs)) if diffs else 0.0
    return distribution


def _compute_problem_distribution(metrics_list: List[Dict]) -> Dict:
    """计算各类型问题的样本分布"""
    problem_defs = [
        ('missed', lambda m: m['missed_gt'] > 0),
        ('merged', lambda m: m['merged_gt'] > 0),
        ('over_segmented', lambda m: m['over_segmented_gt'] > 0),
        ('half_char', lambda m: m['half_char_gt'] > 0 or m['half_char_pred'] > 0),
        ('dirty_spot', lambda m: m['dirty_spot_pred'] > 0),
        ('sliver', lambda m: m['sliver_pred'] > 0),
        ('extra', lambda m: m['extra_pred'] > 0),
        ('width_error', lambda m: m['width_error_count'] > 0),
    ]
    distribution = {}
    problem_counts = []
    for name, check_fn in problem_defs:
        count = sum(1 for m in metrics_list if check_fn(m))
        distribution[name] = {'count': count, 'ratio': count / max(len(metrics_list), 1)}

    # 多重问题
    for m in metrics_list:
        cnt = sum(1 for _, fn in problem_defs if fn(m))
        problem_counts.append(cnt)
    distribution['multi_problem'] = {}
    for i in range(len(problem_defs) + 1):
        count = sum(1 for c in problem_counts if c == i)
        distribution['multi_problem'][f'{i}_types'] = {'count': count, 'ratio': count / max(len(metrics_list), 1)}

    return distribution


# ============================================================
# 打印
# ============================================================

def print_results_summary(agg: Dict, title: str = "模型评测结果"):
    """打印评测结果摘要"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    # 综合评分
    cs = agg.get('composite_score', {})
    total_score = cs.get('total_score', 0)
    print(f"\n>>> 综合评分: {total_score:.2f} / 100 <<<")

    breakdown = cs.get('breakdown', {})
    if breakdown:
        names = {
            'f1': 'F1 Score', 'boundary_acc': '边界准确率',
            'over_seg_resist': '过分割抵抗', 'under_seg_resist': '欠分割抵抗',
            'dirty_spot_resist': '脏点抵抗', 'width_acc': '宽度准确率',
        }
        print(f"  {'指标':<20} {'原始值':>8} {'权重':>6} {'得分':>8}")
        print(f"  {'-'*44}")
        for key in ['f1', 'boundary_acc', 'over_seg_resist', 'under_seg_resist', 'dirty_spot_resist', 'width_acc']:
            b = breakdown.get(key, {})
            print(f"  {names.get(key, key):<20} {b.get('value', 0):>8.4f} {b.get('weight', 0):>5.0%} {b.get('score', 0):>7.2f}")

    print(f"\n总样本数: {agg['total_samples']}")
    print(f"平均预测字符数: {agg['avg_pred_count']:.1f}  平均标注字符数: {agg['avg_gt_count']:.1f}")
    print(f"字符数量匹配率: {agg['char_count_match_rate']:.2%}")

    # 标准指标
    print(f"\n{'='*60}")
    print("标准指标")
    print(f"{'='*60}")
    print(f"  Precision:  {agg['avg_precision']:.4f}")
    print(f"  Recall:     {agg['avg_recall']:.4f}")
    print(f"  F1 Score:   {agg['avg_f1']:.4f}")
    print(f"  IoU:        {agg['avg_iou']:.4f}")
    print(f"  匹配对IoU:  {agg['avg_matched_iou']:.4f}")

    # GT状态分布
    total_gt = agg['total_correct_gt'] + agg['total_missed_gt'] + agg['total_merged_gt'] + agg['total_over_segmented_gt'] + agg['total_half_char_gt'] + agg['total_borderline_gt']
    print(f"\n{'='*60}")
    print(f"标注字符状态分布（共 {total_gt} 个）")
    print(f"{'='*60}")
    gt_state_info = [
        ('correct', '正确', agg['total_correct_gt']),
        ('half_char', '半个字符', agg['total_half_char_gt']),
        ('over_segmented', '被拆分', agg['total_over_segmented_gt']),
        ('merged', '被合并', agg['total_merged_gt']),
        ('missed', '漏检', agg['total_missed_gt']),
        ('borderline', '勉强匹配', agg['total_borderline_gt']),
    ]
    for key, name, count in gt_state_info:
        ratio = count / max(total_gt, 1)
        samples = agg.get(f'samples_with_{key}', 0) if key in ('missed', 'merged', 'over_seg') else 0
        print(f"  {name:<12}: {count:>5}个 ({ratio:>6.2%})")

    # Pred状态分布
    total_pred = agg['total_correct_pred'] + agg['total_half_char_pred'] + agg['total_dirty_spot_pred'] + agg['total_sliver_pred'] + agg['total_extra_pred']
    print(f"\n{'='*60}")
    print(f"预测字符状态分布（共 {total_pred} 个）")
    print(f"{'='*60}")
    pred_state_info = [
        ('correct', '正确', agg['total_correct_pred'], None),
        ('half_char', '半个字符', agg['total_half_char_pred'], None),
        ('dirty_spot', '脏点', agg['total_dirty_spot_pred'], agg['samples_with_dirty_spot']),
        ('sliver', '碎片', agg['total_sliver_pred'], agg['samples_with_sliver']),
        ('extra', '多余', agg['total_extra_pred'], agg['samples_with_extra']),
        ('merger', '合并型', agg['total_merger_pred'], None),
    ]
    for key, name, count, samples in pred_state_info:
        ratio = count / max(total_pred, 1)
        s = f"  涉及样本: {samples}" if samples is not None else ""
        print(f"  {name:<12}: {count:>5}个 ({ratio:>6.2%}){s}")

    # 欠分割
    print(f"\n{'='*60}")
    print("【欠分割分析】")
    print(f"{'='*60}")
    print(f"  漏检(missed):   {agg['total_missed_gt']}个  涉及样本: {agg['samples_with_missed']}")
    print(f"  被合并(merged):  {agg['total_merged_gt']}个  涉及样本: {agg['samples_with_merged']}")
    print(f"  欠分割总计:      {agg['total_missed_gt'] + agg['total_merged_gt']}个  平均比例: {agg['avg_under_seg_ratio']:.2%}")

    # 过分割
    print(f"\n{'='*60}")
    print("【过分割分析】")
    print(f"{'='*60}")
    print(f"  被拆分(over_segmented): {agg['total_over_segmented_gt']}个  涉及样本: {agg['samples_with_over_seg']}")
    print(f"  脏点(dirty_spot):       {agg['total_dirty_spot_pred']}个  涉及样本: {agg['samples_with_dirty_spot']}")
    print(f"  碎片(sliver):           {agg['total_sliver_pred']}个  涉及样本: {agg['samples_with_sliver']}")
    print(f"  多余(extra):            {agg['total_extra_pred']}个  涉及样本: {agg['samples_with_extra']}")
    print(f"  过分割总计:             {agg['total_dirty_spot_pred'] + agg['total_sliver_pred'] + agg['total_extra_pred']}个  平均比例: {agg['avg_over_seg_ratio']:.2%}")

    # 边界MAE
    bm_keys = ['avg_mae_start', 'avg_mae_end', 'avg_mae_boundary', 'avg_mae_with_penalty', 'avg_boundary_acc']
    print(f"\n{'='*60}")
    print("边界MAE指标")
    print(f"{'='*60}")
    print(f"  总匹配字符对数: {agg['total_matched_pairs']}")
    print(f"  起始边界MAE: {agg['global_mae_start']:.2f} 像素")
    print(f"  结束边界MAE: {agg['global_mae_end']:.2f} 像素")
    print(f"  全部边界MAE: {agg['global_mae_boundary']:.2f} 像素")
    print(f"  90分位起始偏移: {agg['global_p90_start']:.1f} 像素")
    print(f"  90分位结束偏移: {agg['global_p90_end']:.1f} 像素")
    print(f"  最大起始偏移: {agg['global_max_start']:.1f} 像素")
    print(f"  最大结束偏移: {agg['global_max_end']:.1f} 像素")
    print(f"  平均综合MAE: {agg['avg_mae_with_penalty']:.2f} 像素")
    print(f"  平均边界准确率: {agg['avg_boundary_acc']:.4f} ({agg['avg_boundary_acc']:.2%})")

    # 边界误差分布
    print(f"\n【边界误差像素分布】")
    err_dist = agg.get('error_distribution', {})
    cum_ratio = 0.0
    for label in ['<=1px', '1-3px', '3-5px', '5-10px', '10-20px', '>20px']:
        d = err_dist.get(label, {})
        cnt = d.get('count', 0)
        ratio = d.get('ratio', 0)
        cum_ratio += ratio
        print(f"  {label:>8}: {cnt:>6}个 ({ratio:>6.2%})  累计: {cum_ratio:.2%}")

    # IoU分布
    print(f"\n【IoU区间分布】")
    for key in ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']:
        dist = agg['iou_distribution'].get(key, {})
        print(f"  IoU {key}: {dist.get('count', 0)}个样本 ({dist.get('ratio', 0):.2%})")

    # 字符数量差异分布
    print(f"\n【字符数量差异分布】")
    for key in ['<-5', '-5~-1', '0', '1~5', '5~10', '>10']:
        dist = agg['char_count_diff_distribution'].get(key, {})
        print(f"  差异 {key}: {dist.get('count', 0)}个样本 ({dist.get('ratio', 0):.2%})")
    print(f"  平均差异: {agg['char_count_diff_distribution']['avg_diff']:.2f}")

    # 问题类型分布
    print(f"\n【问题类型分布】")
    problem_names = {
        'missed': '漏检', 'merged': '被合并', 'over_segmented': '被拆分',
        'half_char': '半个字符', 'dirty_spot': '脏点', 'sliver': '碎片',
        'extra': '多余字符', 'width_error': '宽度错误',
    }
    for key in ['missed', 'merged', 'over_segmented', 'half_char', 'dirty_spot', 'sliver', 'extra', 'width_error']:
        dist = agg['problem_distribution'].get(key, {})
        print(f"  {problem_names[key]:<8}: {dist.get('count', 0)}个样本 ({dist.get('ratio', 0):.2%})")

    print(f"\n【多重问题分布】")
    for i in range(6):
        key = f'{i}_types'
        dist = agg['problem_distribution']['multi_problem'].get(key, {})
        print(f"  同时存在{i}种问题: {dist.get('count', 0)}个样本 ({dist.get('ratio', 0):.2%})")


# ============================================================
# 指标级对比
# ============================================================

def compare_results(file_a: str, file_b: str):
    """对比两个评测结果（指标级）"""
    with open(file_a, 'r', encoding='utf-8') as f:
        data_a = json.load(f)
    with open(file_b, 'r', encoding='utf-8') as f:
        data_b = json.load(f)

    agg_a = data_a['aggregated']
    agg_b = data_b['aggregated']
    label_a = data_a.get('model_version') or data_a.get('project_id', 'A')
    label_b = data_b.get('model_version') or data_b.get('project_id', 'B')

    print(f"\n{'='*70}")
    print(f"模型对比: {label_a}  vs  {label_b}")
    print(f"{'='*70}")

    # 综合评分
    score_a = agg_a.get('composite_score', {}).get('total_score', 0)
    score_b = agg_b.get('composite_score', {}).get('total_score', 0)
    delta = score_b - score_a
    print(f"\n>>> 综合评分: {score_a:.2f}  →  {score_b:.2f}  ({'+' if delta >= 0 else ''}{delta:.2f}) <<<")

    # 关键指标对比
    metrics = [
        ('F1 Score', ['avg_f1'], '{:.4f}', True),
        ('Precision', ['avg_precision'], '{:.4f}', True),
        ('Recall', ['avg_recall'], '{:.4f}', True),
        ('IoU', ['avg_iou'], '{:.4f}', True),
        ('边界准确率', ['avg_boundary_acc'], '{:.4f}', True),
        ('起始MAE(px)', ['global_mae_start'], '{:.2f}', False),
        ('结束MAE(px)', ['global_mae_end'], '{:.2f}', False),
        ('正确GT数', ['total_correct_gt'], '{:.0f}', True),
        ('正确Pred数', ['total_correct_pred'], '{:.0f}', True),
        ('漏检(missed)', ['total_missed_gt'], '{:.0f}', False),
        ('被合并(merged)', ['total_merged_gt'], '{:.0f}', False),
        ('被拆分(over_seg)', ['total_over_segmented_gt'], '{:.0f}', False),
        ('脏点(dirty_spot)', ['total_dirty_spot_pred'], '{:.0f}', False),
        ('碎片(sliver)', ['total_sliver_pred'], '{:.0f}', False),
        ('多余(extra)', ['total_extra_pred'], '{:.0f}', False),
        ('半个字符(half_char)', ['total_half_char_pred'], '{:.0f}', False),
        ('宽度错误数', ['total_width_error_count'], '{:.0f}', False),
        ('过分割比例', ['avg_over_seg_ratio'], '{:.2%}', False),
        ('欠分割比例', ['avg_under_seg_ratio'], '{:.2%}', False),
    ]

    def get_val(agg, path):
        v = agg
        for k in path:
            v = v.get(k, 0) if isinstance(v, dict) else 0
        return v

    print(f"\n  {'指标':<22} {str(label_a):>12} {str(label_b):>12} {'变化':>10} {'方向':>6}")
    print(f"  {'-'*66}")

    for name, path, fmt, higher_better in metrics:
        va = get_val(agg_a, path)
        vb = get_val(agg_b, path)
        d = vb - va
        arrow = ''
        if d != 0:
            if higher_better:
                arrow = '↑ 改善' if d > 0 else '↓ 退化'
            else:
                arrow = '↓ 改善' if d < 0 else '↑ 退化'
        print(f"  {name:<22} {fmt.format(va):>12} {fmt.format(vb):>12} {('+' if d >= 0 else '')+fmt.format(d):>10} {arrow:>6}")

    print(f"\n  样本数: {agg_a.get('total_samples', 0)} vs {agg_b.get('total_samples', 0)}")


# ============================================================
# 字符级跨模型对比
# ============================================================

def compare_char_level(file_a: str, file_b: str):
    """字符级跨模型对比：找出同一个GT字符在两个模型中状态不同的案例"""
    with open(file_a, 'r', encoding='utf-8') as f:
        data_a = json.load(f)
    with open(file_b, 'r', encoding='utf-8') as f:
        data_b = json.load(f)

    label_a = data_a.get('model_version') or 'A'
    label_b = data_b.get('model_version') or 'B'

    # 建立 line_id → detailed 的映射
    detail_a = {r['line_id']: r for r in data_a.get('detailed', [])}
    detail_b = {r['line_id']: r for r in data_b.get('detailed', [])}

    common_lines = set(detail_a.keys()) & set(detail_b.keys())

    print(f"\n{'='*70}")
    print(f"字符级对比: {label_a}  vs  {label_b}")
    print(f"{'='*70}")
    print(f"共同样本数: {len(common_lines)}")

    improved = []   # A错了 B对了
    degraded = []   # A对了 B错了
    both_wrong_changed = []  # 都错了但错误类型不同
    both_correct = 0
    both_wrong_same = 0

    for line_id in sorted(common_lines):
        arch_a = detail_a[line_id].get('archive', {}).get('gt_archive', [])
        arch_b = detail_b[line_id].get('archive', {}).get('gt_archive', [])

        for gt_idx in range(min(len(arch_a), len(arch_b))):
            ga = arch_a[gt_idx]
            gb = arch_b[gt_idx]
            sa = ga.get('status', 'unknown')
            sb = gb.get('status', 'unknown')

            if sa == sb:
                if sa == 'correct':
                    both_correct += 1
                else:
                    both_wrong_same += 1
            else:
                entry = {
                    'line_id': line_id,
                    'gt_idx': gt_idx,
                    'interval': ga.get('interval'),
                    'status_a': sa,
                    'status_b': sb,
                    'iou_a': ga.get('matched_iou', 0),
                    'iou_b': gb.get('matched_iou', 0),
                }
                if sa == 'correct' and sb != 'correct':
                    degraded.append(entry)
                elif sa != 'correct' and sb == 'correct':
                    improved.append(entry)
                else:
                    both_wrong_changed.append(entry)

    total_gt = both_correct + both_wrong_same + len(improved) + len(degraded) + len(both_wrong_changed)

    print(f"\n总GT字符数: {total_gt}")
    print(f"  两个模型都正确: {both_correct} ({both_correct/max(total_gt,1):.2%})")
    print(f"  两个模型都错误(同类): {both_wrong_same} ({both_wrong_same/max(total_gt,1):.2%})")
    print(f"  改善的字符(A错→B对): {len(improved)} ({len(improved)/max(total_gt,1):.2%})")
    print(f"  退化的字符(A对→B错): {len(degraded)} ({len(degraded)/max(total_gt,1):.2%})")
    print(f"  都错但类型不同: {len(both_wrong_changed)} ({len(both_wrong_changed)/max(total_gt,1):.2%})")

    # 改善详情
    if improved:
        print(f"\n--- 改善的字符（前20个）---")
        print(f"  {'line_id':<40} {'gt_idx':>5} {'区间':<12} {'A状态':<14} {'B状态':<14} {'A_IoU':>6} {'B_IoU':>6}")
        for e in improved[:20]:
            inv = str(e['interval'])
            print(f"  {e['line_id']:<40} {e['gt_idx']:>5} {inv:<12} {e['status_a']:<14} {e['status_b']:<14} {e['iou_a']:>6.2f} {e['iou_b']:>6.2f}")

    # 退化详情
    if degraded:
        print(f"\n--- 退化的字符（前20个）---")
        print(f"  {'line_id':<40} {'gt_idx':>5} {'区间':<12} {'A状态':<14} {'B状态':<14} {'A_IoU':>6} {'B_IoU':>6}")
        for e in degraded[:20]:
            inv = str(e['interval'])
            print(f"  {e['line_id']:<40} {e['gt_idx']:>5} {inv:<12} {e['status_a']:<14} {e['status_b']:<14} {e['iou_a']:>6.2f} {e['iou_b']:>6.2f}")

    # 都错但类型变化
    if both_wrong_changed:
        print(f"\n--- 错误类型变化的字符（前20个）---")
        print(f"  {'line_id':<40} {'gt_idx':>5} {'区间':<12} {'A状态':<14} {'B状态':<14}")
        for e in both_wrong_changed[:20]:
            inv = str(e['interval'])
            print(f"  {e['line_id']:<40} {e['gt_idx']:>5} {inv:<12} {e['status_a']:<14} {e['status_b']:<14}")

    # 保存详细结果
    output_path = Path(file_b).parent / f"char_level_compare_{label_a}_vs_{label_b}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_a': label_a,
            'model_b': label_b,
            'total_gt': total_gt,
            'both_correct': both_correct,
            'both_wrong_same': both_wrong_same,
            'improved': improved,
            'degraded': degraded,
            'both_wrong_changed': both_wrong_changed,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细对比结果已保存到: {output_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="基于标注数据集评测模型（v2 - 字符档案体系）")
    parser.add_argument("--project-id", type=str, help="项目ID")
    parser.add_argument("--model-version", type=str, help="模型推理版本（如 v0724）")
    parser.add_argument("--output", type=str, default="evaluation_result.json", help="输出结果文件路径")
    parser.add_argument("--compare", nargs=2, metavar=('FILE_A', 'FILE_B'), help="指标级对比两个评测结果")
    parser.add_argument("--compare-char", nargs=2, metavar=('FILE_A', 'FILE_B'), help="字符级对比两个评测结果")
    args = parser.parse_args()

    # 字符级对比模式
    if args.compare_char:
        compare_char_level(args.compare_char[0], args.compare_char[1])
        return

    # 指标级对比模式
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    if not args.project_id:
        parser.error("--project-id 是必需的（除非使用 --compare 或 --compare-char 模式）")

    project_root = Path(f"datahome/project/{args.project_id}")
    annotations_dir = project_root / "annotations"
    model_jsons_dir = project_root / "model_jsons"

    use_inference_dir = args.model_version is not None
    inference_dir = None
    if use_inference_dir:
        inference_dir = Path(f"datahome/model_inference/{args.model_version}")
        print(f"\n[INFO] 使用推理版本目录: {inference_dir}")
    else:
        print(f"\n[INFO] 使用项目model_jsons目录: {model_jsons_dir}")

    print(f"[INFO] 评测项目: {args.project_id}")
    if args.model_version:
        print(f"[INFO] 模型版本: {args.model_version}")

    if not annotations_dir.exists():
        print(f"[ERROR] 标注目录不存在: {annotations_dir}")
        return

    if use_inference_dir and not inference_dir.exists():
        print(f"[ERROR] 推理版本目录不存在: {inference_dir}")
        return

    annotation_files = list(annotations_dir.glob("*.json"))
    print(f"[INFO] 找到 {len(annotation_files)} 个标注文件")

    results = []
    failed_count = 0

    for anno_file in annotation_files:
        line_id = anno_file.stem

        if use_inference_dir:
            pred_chars, _ = load_model_prediction_from_inference(inference_dir, line_id)
            if pred_chars is None:
                print(f"[WARN] 推理结果不存在: {line_id}")
                failed_count += 1
                continue
        else:
            model_file = model_jsons_dir / f"{line_id}_model.json"
            if not model_file.exists():
                print(f"[WARN] 模型推理结果不存在: {line_id}")
                failed_count += 1
                continue
            pred_chars, _ = load_model_prediction_from_model_jsons(model_file)

        try:
            gt_chars, _ = load_annotation(anno_file)
            result = evaluate_sample(pred_chars, gt_chars)
            result['line_id'] = line_id
            results.append(result)

            if len(results) % 50 == 0:
                print(f"[INFO] 已处理 {len(results)} 个样本...")

        except Exception as e:
            print(f"[ERROR] 处理 {line_id} 失败: {e}")
            failed_count += 1

    print(f"\n[INFO] 处理完成! 成功: {len(results)}, 失败: {failed_count}")

    agg = aggregate_results(results)
    print_results_summary(agg)

    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'project_id': args.project_id,
            'model_version': args.model_version,
            'total_samples': len(results),
            'failed_samples': failed_count,
            'aggregated': agg,
            'detailed': results,
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[INFO] 详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
