#!/usr/bin/env python3
"""
合并两个项目的标注数据，生成微调用 merged_annotations.json

用法:
    python merge_annotations.py 
        --projects datahome/project/proj_20260725_225531 datahome/project/proj_20260706_230417
        --output datahome/datasets/merged_annotations.json
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent


def load_annotation_files(project_dirs: list) -> list:
    """
    从多个项目目录加载所有 annotation JSON
    
    过滤:
      - is_annotated == True
      - is_postponed == False（排除"暂不标注"的）
      - chars 非空
    """
    records = []
    for proj_dir in project_dirs:
        ann_dir = proj_dir / "annotations"
        if not ann_dir.exists():
            print(f"[WARN] 标注目录不存在: {ann_dir}")
            continue
        
        for ann_file in sorted(ann_dir.glob("*.json")):
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[WARN] 读取失败 {ann_file.name}: {e}")
                continue
            
            # 过滤规则
            if not data.get('is_annotated', False):
                continue
            if data.get('is_postponed', False):
                continue
            chars = data.get('chars', [])
            if not chars:
                continue
            
            # 校验每个 char 字段
            valid_chars = []
            for c in chars:
                if (isinstance(c, dict) 
                    and 'col_start' in c 
                    and 'col_end' in c):
                    valid_chars.append(c)
            if not valid_chars:
                continue
            
            data['chars'] = valid_chars
            data['_project'] = proj_dir.name
            data['_ann_file'] = ann_file.name
            records.append(data)
    
    return records


def resolve_image_paths(records: list, lines_dir: Path) -> list:
    """
    将 annotation 的 image_name 字段解析为相对于 data_base_path 的 image_path，
    并提取 image_height 供训练时使用。
    """
    resolved = []
    missing_images = []
    
    for rec in records:
        line_id = rec.get('line_id') or Path(rec['_ann_file']).stem
        image_name = rec.get('image_name', f"{line_id}.png")
        candidate = lines_dir / image_name
        
        if not candidate.exists():
            candidate_fallback = lines_dir / f"{line_id}.png"
            if candidate_fallback.exists():
                candidate = candidate_fallback
            else:
                missing_images.append((line_id, str(candidate)))
                continue
        
        # 提取 image_height
        try:
            img = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("cv2.imread failed")
            image_height = int(img.shape[0])
            image_width = int(img.shape[1])
        except Exception as e:
            print(f"[WARN] 无法读取图像 {candidate.name}: {e}")
            missing_images.append((line_id, str(candidate)))
            continue
        
        resolved.append({
            'line_id': line_id,
            'image_path': str(candidate.relative_to(PROJECT_ROOT)).replace('\\', '/'),
            'image_height': image_height,
            'image_width': image_width,
            'chars': rec['chars'],
            'char_count': len(rec['chars']),
            'project': rec['_project'],
            'source_file': rec['_ann_file'],
            'updated_at': rec.get('updated_at', '')
        })
    
    if missing_images:
        print(f"\n[WARN] 缺失行图像 {len(missing_images)} 张:")
        for lid, p in missing_images[:5]:
            print(f"  {lid} -> {p}")
        if len(missing_images) > 5:
            print(f"  ...其余 {len(missing_images) - 5} 张省略")
    
    return resolved


def compute_summary(records: list) -> dict:
    """生成数据集统计摘要"""
    total_chars = sum(r['char_count'] for r in records)
    
    widths = []
    for r in records:
        for c in r['chars']:
            w = c.get('width', 0)
            if w > 0:
                widths.append(w)
    
    per_project = {}
    for r in records:
        p = r['project']
        if p not in per_project:
            per_project[p] = {'lines': 0, 'chars': 0}
        per_project[p]['lines'] += 1
        per_project[p]['chars'] += r['char_count']
    
    return {
        'generated_at': datetime.now().isoformat(),
        'total_lines': len(records),
        'total_chars': total_chars,
        'avg_chars_per_line': round(total_chars / len(records), 2) if records else 0,
        'char_width': {
            'mean': round(float(np.mean(widths)), 2) if widths else 0,
            'median': round(float(np.median(widths)), 2) if widths else 0,
            'min': int(np.min(widths)) if widths else 0,
            'max': int(np.max(widths)) if widths else 0,
            'p25': round(float(np.percentile(widths, 25)), 2) if widths else 0,
            'p75': round(float(np.percentile(widths, 75)), 2) if widths else 0
        } if widths else {},
        'per_project': per_project
    }


def main():
    parser = argparse.ArgumentParser(description='合并多项目标注数据，生成微调用 merged_annotations.json')
    parser.add_argument('--projects', type=str, nargs='+',
                        default=[
                            'datahome/project/proj_20260725_225531',
                            'datahome/project/proj_20260706_230417',
                        ],
                        help='项目目录列表')
    parser.add_argument('--output', type=str,
                        default='datahome/datasets/merged_annotations.json',
                        help='输出JSON文件路径')
    parser.add_argument('--data-base', type=str, default='datahome',
                        help='数据根目录（包含 lines/ 子目录）')
    args = parser.parse_args()

    project_dirs = [Path(PROJECT_ROOT / p) for p in args.projects]
    data_base = Path(PROJECT_ROOT / args.data_base)
    lines_dir = data_base / "lines"
    output_path = Path(PROJECT_ROOT / args.output)

    print("=" * 70)
    print("合并微调标注数据")
    print("=" * 70)

    print(f"\n[1/4] 扫描 {len(project_dirs)} 个项目...")
    for d in project_dirs:
        print(f"  {d}")
    print(f"  行图像目录: {lines_dir}")

    print(f"\n[2/4] 读取标注文件...")
    raw_records = load_annotation_files(project_dirs)
    print(f"  读取 {len(raw_records)} 条有效标注 (is_annotated=True, 非推迟, chars非空)")

    # 去重（按 line_id，保留最新 updated_at）
    by_line_id = {}
    for rec in raw_records:
        lid = rec.get('line_id') or Path(rec['_ann_file']).stem
        upd = rec.get('updated_at', '')
        if lid not in by_line_id or upd > by_line_id[lid].get('updated_at', ''):
            by_line_id[lid] = rec
    deduped = list(by_line_id.values())
    dup_count = len(raw_records) - len(deduped)
    print(f"  去重后 {len(deduped)} 条 (移除 {dup_count} 个重复 line_id)")

    print(f"\n[3/4] 解析行图像路径并提取 image_height...")
    resolved = resolve_image_paths(deduped, lines_dir)
    resolved.sort(key=lambda r: r['line_id'])

    print(f"\n[4/4] 写入 {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = compute_summary(resolved)
    out_data = {
        'version': '1.0',
        'summary': summary,
        'annotations': resolved
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out_data['annotations'], f, ensure_ascii=False, indent=2)

    # 同时生成 split 文件，如果配置中没指定 split_file，finetune.py 会调用 generate_dataset_split
    print(f"\n合并完成！")
    print(f"  输出文件: {output_path}")
    print(f"  总行数:   {summary['total_lines']}")
    print(f"  总字符:   {summary['total_chars']}")
    print(f"  每行均值: {summary['avg_chars_per_line']}")
    if summary.get('char_width'):
        cw = summary['char_width']
        print(f"  字宽均值/中位: {cw['mean']} / {cw['median']} px")
        print(f"  字宽范围: {cw['min']} ~ {cw['max']} px")
    print(f"\n  各项目分布:")
    for proj, info in summary['per_project'].items():
        print(f"    {proj}: {info['lines']} 行 / {info['chars']} 字符")


if __name__ == "__main__":
    main()
