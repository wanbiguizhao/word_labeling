from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from pathlib import Path

@dataclass
class Line2CharConfig:
    """行转汉字配置类（规则执行模式）"""
    
    # ===== 输出目录配置 =====
    output_dir: str = "output"
    
    raw_pdf_dir: str = "raw/pdf"
    raw_pages_dir: str = "raw/pages"
    
    lines_dir: str = "lines"
    chars_dir: str = "chars"
    
    annotations_dir: str = "annotations"
    datasets_dir: str = "datasets"
    
    rule_json_dir: str = "rule_jsons"
    save_visual: bool = True
    save_char_images: bool = False

    # ===== 规则执行参数 =====
    rule_narrow_blank_threshold: int = 15
    rule_min_char_width: int = 30

    # ===== 可视化配置 =====
    sep_line_color: Tuple[int, int, int] = (128, 0, 128)
    sep_line_width: int = 2
    cut_line_width: int = 1
    start_color: Tuple[int, int, int] = (255, 0, 0)
    end_color: Tuple[int, int, int] = (0, 255, 0)

    # ===== 行高度过滤参数 =====
    min_line_height: int = 30
    max_line_height: int = 55

    # ===== 后处理参数（合并粘连字符）=====
    postprocess_merge_enabled: bool = True
    postprocess_min_gap_width: int = 3
    postprocess_max_height_diff: float = 0.2
    postprocess_min_aspect_ratio: float = 0.5
    postprocess_max_aspect_ratio: float = 1.5
    postprocess_single_aspect_ratio: float = 0.7
    
    # ===== 版本管理 =====
    segmentation_version: str = "rule_v1"

    def validate(self) -> None:
        assert self.rule_narrow_blank_threshold >= 0, "窄空白阈值不能为负数"
        assert self.rule_min_char_width >= 1, "最小字符宽度至少为1"
        assert self.min_line_height >= 0, "最小行高度不能为负数"
        assert self.max_line_height >= self.min_line_height, "最大行高度必须大于最小行高度"
        assert self.sep_line_width >= 1, "分隔线宽度至少为1"
        assert self.cut_line_width >= 1, "切割线宽度至少为1"

    def get_rule_config(self) -> dict:
        return {
            "narrow_blank_threshold": self.rule_narrow_blank_threshold,
            "min_char_width": self.rule_min_char_width
        }
    
    def get_visual_config(self) -> dict:
        return {
            "sep_line_color": self.sep_line_color,
            "sep_line_width": self.sep_line_width,
            "cut_line_width": self.cut_line_width,
            "start_color": self.start_color,
            "end_color": self.end_color
        }
    
    def init_directories(self, base_path: Path) -> None:
        dirs = [
            self.raw_pdf_dir,
            self.raw_pages_dir,
            self.lines_dir,
            self.chars_dir,
            self.annotations_dir,
            self.datasets_dir,
            self.rule_json_dir
        ]
        for d in dirs:
            (base_path / d).mkdir(parents=True, exist_ok=True)