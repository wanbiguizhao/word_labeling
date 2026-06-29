import json
import cv2
import numpy as np
import click
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


def load_all_line_ids(data_base_path: Path) -> List[str]:
    rule_jsons_dir = data_base_path / "rule_jsons"
    json_files = sorted(rule_jsons_dir.glob("*_rule.json"))
    
    line_ids = []
    for json_file in json_files:
        line_id = json_file.stem.replace("_rule", "")
        line_path = data_base_path / "lines" / f"{line_id}.png"
        if line_path.exists():
            line_ids.append(line_id)
    
    return line_ids


def get_line_width(data_base_path: Path, line_id: str) -> int:
    line_path = data_base_path / "lines" / f"{line_id}.png"
    img = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
    return img.shape[1] if img is not None else 0


def compute_char_width_stats(data_base_path: Path, line_ids: List[str]) -> Dict:
    """
    计算字符宽度统计信息
    
    Args:
        data_base_path: 数据基础目录
        line_ids: 行ID列表
    
    Returns:
        字符宽度统计信息
    """
    rule_jsons_dir = data_base_path / "rule_jsons"
    all_char_widths = []
    line_char_widths = {}
    
    for line_id in line_ids:
        json_path = rule_jsons_dir / f"{line_id}_rule.json"
        if not json_path.exists():
            continue
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                rule_data = json.load(f)
            
            chars = rule_data.get('chars', [])
            widths = []
            for char in chars:
                w = char.get('width', 0)
                if 3 <= w <= 100:
                    widths.append(w)
            
            if widths:
                line_char_widths[line_id] = float(np.median(widths))
                all_char_widths.extend(widths)
        except Exception:
            pass
    
    stats = {
        'total_chars': len(all_char_widths),
        'global_median_char_width': float(np.median(all_char_widths)) if all_char_widths else 0.0,
        'global_mean_char_width': float(np.mean(all_char_widths)) if all_char_widths else 0.0,
        'global_min_char_width': min(all_char_widths) if all_char_widths else 0,
        'global_max_char_width': max(all_char_widths) if all_char_widths else 0,
        'line_median_char_widths': line_char_widths,
    }
    
    return stats


def generate_dataset_split(
    data_base_path: Path,
    output_path: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    sort_by_width: bool = False
) -> Dict:
    line_ids = load_all_line_ids(data_base_path)
    
    # 计算字符宽度统计
    char_width_stats = compute_char_width_stats(data_base_path, line_ids)
    
    if sort_by_width:
        line_ids_with_width = [(line_id, get_line_width(data_base_path, line_id)) 
                               for line_id in line_ids]
        line_ids_with_width.sort(key=lambda x: x[1])
        line_ids = [x[0] for x in line_ids_with_width]
    
    np.random.seed(seed)
    shuffled_ids = line_ids.copy()
    np.random.shuffle(shuffled_ids)
    
    split_idx = int(len(shuffled_ids) * train_ratio)
    train_ids = shuffled_ids[:split_idx]
    val_ids = shuffled_ids[split_idx:]
    
    train_widths = [get_line_width(data_base_path, line_id) for line_id in train_ids]
    val_widths = [get_line_width(data_base_path, line_id) for line_id in val_ids]
    
    split_info = {
        "version": "1.1",
        "created_at": datetime.now().isoformat(),
        "config": {
            "train_ratio": train_ratio,
            "seed": seed,
            "sort_by_width": sort_by_width,
            "data_base_path": str(data_base_path)
        },
        "train_ids": train_ids,
        "val_ids": val_ids,
        "stats": {
            "total_count": len(line_ids),
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "avg_train_width": int(np.mean(train_widths)) if train_widths else 0,
            "avg_val_width": int(np.mean(val_widths)) if val_widths else 0,
            "max_train_width": max(train_widths) if train_widths else 0,
            "max_val_width": max(val_widths) if val_widths else 0
        },
        "char_width_stats": char_width_stats
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    return split_info


def load_dataset_split(split_path: Path) -> Dict:
    with open(split_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@click.command("split-dataset")
@click.option("--train-ratio", type=float, default=0.8, show_default=True, help="训练集比例")
@click.option("--seed", type=int, default=42, show_default=True, help="随机种子")
@click.option("--sort-by-width", is_flag=True, help="按宽度排序后划分")
@click.option("--data-base-path", type=str, default="datahome", show_default=True,
              help="数据基础目录（相对项目根）")
@click.option("--output", type=str, default="ai_model/data/dataset_split.json",
              show_default=True, help="划分文件输出路径（相对项目根）")
def cli(train_ratio, seed, sort_by_width, data_base_path, output):
    """生成数据集训练/验证划分文件"""
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_path = base_dir / data_base_path
    output_path = base_dir / output

    click.echo(f"[INFO] 数据基础目录: {data_path}")
    click.echo(f"[INFO] 输出路径: {output_path}")
    click.echo(f"[INFO] 训练集比例: {train_ratio}")
    click.echo(f"[INFO] 随机种子: {seed}")
    click.echo(f"[INFO] 按宽度排序: {sort_by_width}")

    split_info = generate_dataset_split(
        data_base_path=data_path,
        output_path=output_path,
        train_ratio=train_ratio,
        seed=seed,
        sort_by_width=sort_by_width
    )

    click.echo(f"\n[INFO] 数据集划分完成!")
    click.echo(f"[INFO] 总样本数: {split_info['stats']['total_count']}")
    click.echo(f"[INFO] 训练集: {split_info['stats']['train_count']}")
    click.echo(f"[INFO] 验证集: {split_info['stats']['val_count']}")
    click.echo(f"[INFO] 训练集平均宽度: {split_info['stats']['avg_train_width']}")
    click.echo(f"[INFO] 验证集平均宽度: {split_info['stats']['avg_val_width']}")
    
    cws = split_info.get('char_width_stats', {})
    click.echo(f"\n[INFO] 字符宽度统计:")
    click.echo(f"[INFO] 总字符数: {cws.get('total_chars', 0)}")
    click.echo(f"[INFO] 全局中位字符宽度: {cws.get('global_median_char_width', 0.0):.2f} px")
    click.echo(f"[INFO] 全局平均字符宽度: {cws.get('global_mean_char_width', 0.0):.2f} px")
    click.echo(f"[INFO] 字符宽度范围: {cws.get('global_min_char_width', 0)} - {cws.get('global_max_char_width', 0)} px")


if __name__ == "__main__":
    cli()