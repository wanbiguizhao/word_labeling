import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime


def extract_pdf_id(line_id: str) -> str:
    parts = line_id.split('_')
    for i, part in enumerate(parts):
        if part == 'pdf':
            if i + 1 < len(parts):
                return parts[i + 1]
    return 'unknown'


def write_inference_jsonl(output_path: Path, results: Dict[str, List[Tuple[int, int]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line_id, intervals in results.items():
            record = {
                "line_id": line_id,
                "char_count": len(intervals),
                "chars": [[int(start), int(end)] for start, end in intervals]
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def write_detail_json(detail_dir: Path, line_id: str, output_data: Dict) -> None:
    pdf_id = extract_pdf_id(line_id)
    pdf_dir = detail_dir / pdf_id
    pdf_dir.mkdir(parents=True, exist_ok=True)
    output_path = pdf_dir / f"{line_id}_model.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def write_metadata(metadata_path: Path, model_path: str, success_count: int,
                   fail_count: int, params: Dict) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": metadata_path.parent.name,
        "model_path": model_path,
        "inference_time": datetime.now().isoformat(),
        "total_lines": success_count + fail_count,
        "success_count": success_count,
        "fail_count": fail_count,
        "params": params
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def read_inference_jsonl(jsonl_path: Path, filter_line_ids: Optional[List[str]] = None) -> Dict[str, List[Tuple[int, int]]]:
    results = {}
    if not jsonl_path.exists():
        return results
    
    filter_set = set(filter_line_ids) if filter_line_ids else None
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if filter_set is None or record['line_id'] in filter_set:
                results[record['line_id']] = [(int(c[0]), int(c[1])) for c in record['chars']]
    return results


def read_detail_json(detail_dir: Path, line_id: str) -> Optional[Dict]:
    pdf_id = extract_pdf_id(line_id)
    detail_path = detail_dir / pdf_id / f"{line_id}_model.json"
    if detail_path.exists():
        with open(detail_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def read_metadata(metadata_path: Path) -> Optional[Dict]:
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def list_versions(base_dir: Path) -> List[str]:
    if not base_dir.exists():
        return []
    versions = []
    for item in base_dir.iterdir():
        if item.is_dir():
            versions.append(item.name)
    return sorted(versions, reverse=True)


def get_latest_version(base_dir: Path) -> Optional[str]:
    versions = list_versions(base_dir)
    return versions[0] if versions else None