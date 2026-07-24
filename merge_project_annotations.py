import json
from pathlib import Path


def merge_project_annotations(project_id: str):
    base_dir = Path('datahome')
    annotations_dir = base_dir / 'project' / project_id / 'annotations'
    
    if not annotations_dir.exists():
        print(f"[ERROR] 标注目录不存在: {annotations_dir}")
        return
    
    annotation_files = list(annotations_dir.glob('*.json'))
    print(f"[INFO] 找到 {len(annotation_files)} 个标注文件")
    
    all_annotations = []
    
    for json_file in annotation_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            line_id = json_file.stem
            
            chars = []
            if 'chars' in data:
                for char in data['chars']:
                    if 'col_start' in char and 'col_end' in char:
                        width = char.get('width', char['col_end'] - char['col_start'])
                        chars.append({
                            'col_start': char['col_start'],
                            'col_end': char['col_end'],
                            'width': width
                        })
            
            if chars:
                all_annotations.append({
                    'line_id': line_id,
                    'chars': chars,
                    'source': f'project/{project_id}'
                })
        except Exception as e:
            print(f"[WARN] 读取 {json_file} 失败: {e}")
    
    print(f"[INFO] 共合并 {len(all_annotations)} 条有效标注")
    
    sorted_annotations = sorted(all_annotations, key=lambda x: x['line_id'])
    
    output_file = base_dir / 'datasets' / f'merged_annotations_{project_id}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_annotations, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] 合并完成，输出文件: {output_file}")
    
    total_chars = sum(len(a['chars']) for a in sorted_annotations)
    print(f"[INFO] 总字符数: {total_chars}")
    print(f"[INFO] 平均每行字符数: {total_chars / len(sorted_annotations):.1f}")
    
    return output_file


if __name__ == '__main__':
    merge_project_annotations('proj_20260706_230417')
