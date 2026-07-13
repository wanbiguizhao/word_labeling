import json
from pathlib import Path
from typing import List, Optional
from PIL import Image
from fastapi import HTTPException


def load_json_file(file_path: Path) -> Optional[dict]:
    """加载 JSON 文件，不存在返回 None"""
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def chars_to_lines(chars: List[dict], image_width: int) -> List[dict]:
    """将字符列表转换为切割线列表
    每个字符的 col_start → red 线，col_end → green 线
    """
    line_dict = {}
    for char in chars:
        col_start = char.get("col_start")
        if col_start is not None and col_start != 0:
            line_dict[col_start] = "red"
    for char in chars:
        col_end = char.get("col_end")
        if col_end is not None and col_end != 0:
            line_dict[col_end] = "green"

    lines = [{"pos": pos, "color": color} for pos, color in line_dict.items()]
    lines.sort(key=lambda x: x["pos"])
    return lines


def lines_to_chars(lines: List[dict], image_width: int) -> List[dict]:
    """将切割线列表转换为字符列表
    排序后两两配对：第0个red+第1个green = 一个字符
    """
    positions = sorted([item["pos"] for item in lines])
    chars = []
    for i in range(0, len(positions), 2):
        if i + 1 >= len(positions):
            break
        left = positions[i]
        right = positions[i + 1]
        chars.append({
            "col_start": left,
            "col_end": right,
            "width": right - left
        })
    return chars


def get_image_width(image_id: str, raw_images_dir: Path) -> int:
    """获取图片宽度（像素）"""
    for ext in [".png", ".jpg", ".jpeg"]:
        img_file = raw_images_dir / f"{image_id}{ext}"
        if img_file.exists():
            with Image.open(img_file) as img:
                return img.width
    raise HTTPException(status_code=404, detail="获取图片宽度失败")
