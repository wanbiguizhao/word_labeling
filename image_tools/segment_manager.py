"""
分割流程管理器
统一编排：PDF -> 页面 -> 行 -> 汉字 的完整流程

功能：
    1. PDF转图片（pdf2image）
    2. 图片转行（image2line）
    3. 行转汉字（line2char）- 规则模式
    4. 行高度过滤（filter）
    5. 全链路血缘索引（lineage）

数据目录结构：
    data_base_path/
    ├── raw/
    │   ├── pdf/              # 原始PDF文件
    │   └── pages/            # 页面图像（仅存储一份）
    ├── lines/                # 行图像
    ├── chars/                # 单字符图像
    ├── annotations/          # 标注数据
    ├── datasets/             # 数据集定义
    ├── rule_jsons/           # 规则切割结果
    └── lineage.json          # 全链路血缘索引

血缘索引结构：
    lineage = {
        "metadata": {
            "version": "1.0",
            "created_at": "2024-01-01T00:00:00",
            "segmentation_version": "rule_v1"
        },
        "pdfs": {
            "pdf_id": {
                "pdf_id": "xxx",
                "filename": "xxx.pdf",
                "file_path": "raw/pdf/xxx.pdf",
                "total_pages": 10
            }
        },
        "pages": {
            "page_id": {
                "page_id": "xxx",
                "pdf_id": "xxx",
                "page_num": 1,
                "image_path": "raw/pages/xxx.png",
                "width": 2480,
                "height": 3508,
                "lines": ["line_id_1", "line_id_2", ...]
            }
        },
        "lines": {
            "line_id": {
                "line_id": "xxx",
                "page_id": "xxx",
                "pdf_id": "xxx",
                "page_num": 1,
                "line_idx": 0,
                "image_path": "lines/xxx.png",
                "y_start": 100,
                "y_end": 150,
                "width": 2480,
                "height": 50,
                "chars": ["char_id_1", "char_id_2", ...],
                "annotation_status": "unannotated",
                "confidence": 0.95
            }
        },
        "chars": {
            "char_id": {
                "char_id": "xxx",
                "line_id": "xxx",
                "page_id": "xxx",
                "pdf_id": "xxx",
                "page_num": 1,
                "line_idx": 0,
                "char_idx": 0,
                "image_path": "chars/xxx.png",
                "col_start": 10,
                "col_end": 60,
                "width": 50,
                "height": 50,
                "abs_x": 10,
                "abs_y": 100
            }
        }
    }
"""

import os
import sys
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import fitz

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from image_tools.pdf_config import Pdf2ImageConfig
from image_tools.image_config import Image2LineConfig
from image_tools.segment_config import Line2CharConfig
from image_tools.imageCore import (
    CharSegmentConfig, TextLineDetector, VerticalProjectionSegmenter
)


@dataclass
class ProcessResult:
    """处理结果封装类"""
    success: bool
    message: str
    data: Optional[Dict] = None


class LineHeightFilter:
    """行高度过滤器"""
    
    def __init__(self, min_h: int = 30, max_h: int = 55):
        self.min_h = min_h
        self.max_h = max_h
    
    def is_valid(self, height: int) -> bool:
        return self.min_h <= height <= self.max_h


class SegmentManager:
    """
    分割流程管理器
    
    统一管理 PDF -> 页面 -> 行 -> 汉字 的完整处理流程
    """
    
    def __init__(
        self,
        pdf_cfg: Pdf2ImageConfig,
        img_cfg: Image2LineConfig,
        char_cfg: Line2CharConfig,
        data_base_path: Path
    ):
        """
        初始化分割流程管理器
        
        Args:
            pdf_cfg: PDF转图片配置
            img_cfg: 图片转行配置
            char_cfg: 行转汉字配置
            data_base_path: 数据基础目录
        """
        self.pdf_cfg = pdf_cfg
        self.img_cfg = img_cfg
        self.char_cfg = char_cfg
        self.data_base_path = data_base_path
        
        self._line_filter = LineHeightFilter(
            min_h=char_cfg.min_line_height,
            max_h=char_cfg.max_line_height
        )
        
        self._segment_cfg = CharSegmentConfig()
        self._segmenter = VerticalProjectionSegmenter(self._segment_cfg)
        
        char_cfg.init_directories(data_base_path)
        
        self.lineage = self._load_or_init_lineage()
        self._save_pending = False
    
    def _load_or_init_lineage(self) -> Dict:
        """加载或初始化血缘索引"""
        lineage_path = self.data_base_path / "lineage.json"
        
        if lineage_path.exists():
            try:
                with open(lineage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "segmentation_version": self.char_cfg.segmentation_version,
                "total_pdfs": 0,
                "total_pages": 0,
                "total_lines": 0,
                "total_chars": 0
            },
            "pdfs": {},
            "pages": {},
            "lines": {},
            "chars": {}
        }
    
    def _save_lineage(self, force: bool = False) -> None:
        """保存血缘索引到文件（支持延迟写入）"""
        if not force and not self._save_pending:
            return
        
        lineage_path = self.data_base_path / "lineage.json"
        with open(lineage_path, 'w', encoding='utf-8') as f:
            json.dump(self.lineage, f, ensure_ascii=False, indent=2, default=str)
        
        self._save_pending = False
    
    def _mark_dirty(self) -> None:
        """标记血缘索引需要保存"""
        self._save_pending = True
    
    def _generate_id(self, prefix: str, *args) -> str:
        """生成唯一ID"""
        parts = [prefix] + [str(arg) for arg in args]
        return "_".join(parts)
    
    def pdf_to_images(self, pdf_path: str) -> Tuple[bool, List[str]]:
        """
        PDF转图片
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            (success, page_ids)
        """
        try:
            pdf_filename = Path(pdf_path).name
            pdf_id = self._generate_id("pdf", Path(pdf_path).stem)
            
            raw_pdf_dir = self.data_base_path / self.char_cfg.raw_pdf_dir
            raw_pdf_dir.mkdir(parents=True, exist_ok=True)
            
            dest_pdf_path = raw_pdf_dir / pdf_filename
            if str(pdf_path) != str(dest_pdf_path):
                if not dest_pdf_path.exists():
                    shutil.copy2(pdf_path, dest_pdf_path)
            
            doc = fitz.open(str(dest_pdf_path))
            
            total_pages = doc.page_count
            start_page = 0
            end_page = total_pages - 1
            if self.pdf_cfg.page_range:
                start_page = max(0, self.pdf_cfg.page_range[0] - 1)
                end_page = min(total_pages - 1, self.pdf_cfg.page_range[1] - 1)
            
            raw_pages_dir = self.data_base_path / self.char_cfg.raw_pages_dir
            raw_pages_dir.mkdir(parents=True, exist_ok=True)
            
            zoom = self.pdf_cfg.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            colorspace = fitz.csGRAY if self.pdf_cfg.grayscale else fitz.csRGB
            
            output_format = self.pdf_cfg.output_format
            pages_ref = self.lineage["pages"]
            data_base_path = self.data_base_path
            
            page_ids = [None] * (end_page - start_page + 1)
            for idx, page_idx in enumerate(range(start_page, end_page + 1)):
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=colorspace)
                
                page_num = page_idx + 1
                page_id = self._generate_id("page", pdf_id, page_num)
                img_name = f"{page_id}.{output_format}"
                img_path = raw_pages_dir / img_name
                
                pix.save(str(img_path))
                
                page_info = {
                    "page_id": page_id,
                    "pdf_id": pdf_id,
                    "page_num": page_num,
                    "image_path": str(img_path.relative_to(data_base_path)),
                    "width": pix.width,
                    "height": pix.height,
                    "lines": []
                }
                pages_ref[page_id] = page_info
                page_ids[idx] = page_id
            
            doc.close()
            
            self.lineage["pdfs"][pdf_id] = {
                "pdf_id": pdf_id,
                "filename": pdf_filename,
                "file_path": str(dest_pdf_path.relative_to(data_base_path)),
                "total_pages": total_pages
            }
            
            self._mark_dirty()
            return True, page_ids
            
        except Exception as e:
            return False, [str(e)]
    
    @staticmethod
    def _fast_line_detection(image_path: str, cfg: CharSegmentConfig) -> Tuple[List[Tuple[int, int]], np.ndarray, int, int]:
        """
        快速行检测（优化版预处理管线）
        
        使用简化的预处理流程进行行边界检测：
        - 直接读取灰度图，跳过 BGR->Gray 转换
        - 使用 Otsu 二值化替换慢速的 fastNlMeansDenoising + bilateralFilter + adaptiveThreshold
        - 保留原始图像用于行裁剪
        
        Returns:
            (lines, original_img, h, w)
        """
        # 直接读取灰度图
        original = cv2.imread(image_path)
        if original is None:
            return [], None, 0, 0
        
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        # 快速预处理：Otsu 二值化 + 轻量形态学处理
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 轻量形态学闭合，连接相邻文字
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 行检测
        detector = TextLineDetector(cfg)
        lines = detector.detect(closed, h, w)
        
        return lines, original, h, w

    def image_to_lines(self, page_id: str) -> Tuple[bool, List[str]]:
        """
        图片转行
        
        Args:
            page_id: 页面ID
        
        Returns:
            (success, line_ids)
        """
        try:
            pages_ref = self.lineage["pages"]
            page_info = pages_ref.get(page_id)
            if page_info is None:
                return False, [f"页面ID不存在: {page_id}"]
            
            img_path = self.data_base_path / page_info["image_path"]
            
            lines, original, h, w = self._fast_line_detection(str(img_path), self._segment_cfg)
            if not lines:
                return True, []

            lines_dir = self.data_base_path / self.char_cfg.lines_dir
            lines_dir.mkdir(parents=True, exist_ok=True)

            lines_ref = self.lineage["lines"]
            pdf_id = page_info["pdf_id"]
            page_num = page_info["page_num"]
            line_filter = self._line_filter
            data_base_path = self.data_base_path
            
            line_ids = []
            page_lines_list = pages_ref[page_id]["lines"]
            
            for lid, (y1, y2) in enumerate(lines):
                line_img = original[y1:y2, :]
                line_id = self._generate_id("line", page_id, lid)
                line_save = lines_dir / f"{line_id}.png"
                cv2.imwrite(str(line_save), line_img)
                
                line_height = int(y2 - y1)
                line_info = {
                    "line_id": line_id,
                    "page_id": page_id,
                    "pdf_id": pdf_id,
                    "page_num": page_num,
                    "line_idx": lid,
                    "image_path": str(line_save.relative_to(data_base_path)),
                    "y_start": int(y1),
                    "y_end": int(y2),
                    "width": int(w),
                    "height": line_height,
                    "chars": [],
                    "annotation_status": "unannotated",
                    "confidence": 0.0,
                    "is_valid": line_filter.is_valid(line_height)
                }
                
                lines_ref[line_id] = line_info
                page_lines_list.append(line_id)
                line_ids.append(line_id)

            self._mark_dirty()
            return True, line_ids

        except Exception as e:
            return False, [str(e)]
    
    def filter_lines(self, line_ids: List[str]) -> Tuple[List[str], List[str]]:
        """
        按行高度过滤（从lineage中直接读取高度，无需再次读取图片）
        
        Args:
            line_ids: 行ID列表
        
        Returns:
            (valid_line_ids, invalid_line_ids)
        """
        valid, invalid = [], []
        for line_id in line_ids:
            if line_id not in self.lineage["lines"]:
                continue
            line_info = self.lineage["lines"][line_id]
            if line_info.get("is_valid", self._line_filter.is_valid(line_info["height"])):
                valid.append(line_id)
            else:
                invalid.append(line_id)
        return valid, invalid
    
    def process_lines_by_rule(self, line_ids: List[str]) -> Dict[str, Dict]:
        """
        规则执行：行转汉字（批量处理）
        
        Args:
            line_ids: 行ID列表
        
        Returns:
            处理结果字典 {line_id: rule_data}
        """
        rule_json_dir = self.data_base_path / self.char_cfg.rule_json_dir
        rule_json_dir.mkdir(parents=True, exist_ok=True)
        
        # 预计算时间戳，避免每行重复调用
        created_at = datetime.now().isoformat()
        seg_version = self.char_cfg.segmentation_version
        
        results = {}
        for line_id in line_ids:
            try:
                line_info = self.lineage["lines"].get(line_id)
                if line_info is None:
                    continue
                
                line_path = self.data_base_path / line_info["image_path"]
                
                gray = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    print(f"   [WARN] 无法读取图片: {line_path}")
                    continue
                
                h, w = gray.shape[:2]
                if w == 0 or h == 0:
                    print(f"   [WARN] 空图片: {line_path}")
                    continue
                
                segments = self._segmenter.get_segment_classes(gray)
                
                chars = []
                segments_type_start_end = []
                
                for seg_type, start, end in segments:
                    segments_type_start_end.append([int(seg_type), int(start), int(end)])
                    if seg_type == 1:
                        chars.append({
                            "col_start": int(start),
                            "col_end": int(end),
                            "width": int(end - start)
                        })
                
                rule_result = {
                    "line_id": line_id,
                    "image_path": str(line_path.relative_to(self.data_base_path)),
                    "chars": chars,
                    "segments_type_start_end": segments_type_start_end,
                    "total_chars": len(chars),
                    "image_width": w,
                    "image_height": h,
                    "total_segments": len(segments),
                    "segmentation_version": seg_version,
                    "created_at": created_at
                }
                
                # 使用紧凑格式写入JSON，减少I/O量
                rule_json_path = rule_json_dir / f"{line_id}_rule.json"
                with open(str(rule_json_path), 'w', encoding='utf-8') as f:
                    json.dump(rule_result, f, ensure_ascii=False, separators=(',', ':'))
                
                results[line_id] = rule_result
                line_info["confidence"] = 0.95
                
            except Exception as e:
                print(f"   [ERROR] 处理失败: {line_id} - {str(e)}")
                continue
        
        self._mark_dirty()
        return results
    
    def extract_char_images(self, line_id: str, rule_data: Dict) -> List[str]:
        """
        从行图片中提取单个汉字（仅记录坐标信息，可选保存图片）
        
        Args:
            line_id: 行ID
            rule_data: 规则切割结果数据
        
        Returns:
            字符ID列表
        """
        char_ids = []
        
        line_info = self.lineage["lines"].get(line_id)
        if line_info is None:
            return char_ids
        
        rule_chars = rule_data.get('chars', [])
        if not rule_chars:
            return char_ids
        
        # 延迟初始化：仅当需要保存图片时才读取行图像
        line_img = None
        chars_dir = None
        if self.char_cfg.save_char_images:
            line_path = self.data_base_path / line_info["image_path"]
            line_img = cv2.imread(str(line_path))
            if line_img is None:
                print(f"   [ERROR] 无法读取行图片: {line_path}")
                return char_ids
            chars_dir = self.data_base_path / self.char_cfg.chars_dir
            chars_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用局部变量缓存热点引用，减少 dict 查找开销
        lines_ref = self.lineage["lines"]
        chars_ref = self.lineage["chars"]
        line_chars_list = lines_ref[line_id]["chars"]
        
        for char_idx, char_info in enumerate(rule_chars):
            start = char_info.get('col_start', 0)
            end = char_info.get('col_end', 0)
            
            if end <= start:
                continue
            
            char_id = self._generate_id("char", line_id, char_idx)
            
            image_path = None
            if line_img is not None and chars_dir is not None:
                char_img = line_img[:, start:end]
                if char_img.size == 0:
                    continue
                char_name = f"{char_id}.png"
                char_path = chars_dir / char_name
                cv2.imwrite(str(char_path), char_img)
                image_path = str(char_path.relative_to(self.data_base_path))
            
            abs_x = start
            abs_y = line_info["y_start"]
            
            char_info_full = {
                "char_id": char_id,
                "line_id": line_id,
                "page_id": line_info["page_id"],
                "pdf_id": line_info["pdf_id"],
                "page_num": line_info["page_num"],
                "line_idx": line_info["line_idx"],
                "char_idx": char_idx,
                "image_path": image_path,
                "col_start": start,
                "col_end": end,
                "width": char_info.get('width', 0),
                "height": line_info["height"],
                "abs_x": abs_x,
                "abs_y": abs_y,
                "abs_x_end": abs_x + (end - start),
                "abs_y_end": abs_y + line_info["height"]
            }
            
            chars_ref[char_id] = char_info_full
            line_chars_list.append(char_id)
            char_ids.append(char_id)
        
        return char_ids
    
    def postprocess_merge_chars(self) -> None:
        """
        后处理：合并粘连字符
        
        检测并合并同一行中相邻的、间隙很小的字符段，适用于如"北"字被误切割为两半的情况。
        
        合并条件：
        1. 间隙宽度 < min_gap_width（默认2像素）
        2. 两个字符段的高度差 < max_height_diff（默认20%）
        3. 两个字符段都是窄的（宽高比 < 0.6），说明可能是被分割的半字符
        4. 合并后的字符宽高比在合理范围内（0.6-1.4）
        """
        if not self.char_cfg.postprocess_merge_enabled:
            return
        
        merged_count = 0
        
        # 缓存配置参数到局部变量
        cfg = self.char_cfg
        min_gap = cfg.postprocess_min_gap_width
        max_height_diff = cfg.postprocess_max_height_diff
        min_aspect_ratio = cfg.postprocess_min_aspect_ratio
        max_aspect_ratio = cfg.postprocess_max_aspect_ratio
        single_aspect_ratio = cfg.postprocess_single_aspect_ratio
        
        # 缓存 lineage 热点引用到局部变量
        lines_ref = self.lineage["lines"]
        chars_ref = self.lineage["chars"]
        
        for line_id, line_info in lines_ref.items():
            char_ids = line_info.get("chars", [])
            if len(char_ids) < 2:
                continue
            
            # 批量获取字符对象，过滤掉 None
            chars = [chars_ref.get(cid) for cid in char_ids]
            chars = [c for c in chars if c is not None]
            if len(chars) < 2:
                continue
            
            merged_chars = []
            i = 0
            n = len(chars)
            while i < n:
                current_char = chars[i]
                
                if i + 1 < n:
                    next_char = chars[i + 1]
                    
                    gap_width = next_char["col_start"] - current_char["col_end"]
                    
                    if gap_width >= 0 and gap_width < min_gap:
                        avg_height = (current_char["height"] + next_char["height"]) * 0.5
                        height_diff = abs(current_char["height"] - next_char["height"])
                        
                        if height_diff < avg_height * max_height_diff:
                            current_ratio = current_char["width"] / current_char["height"]
                            next_ratio = next_char["width"] / next_char["height"]
                            
                            if current_ratio < single_aspect_ratio and next_ratio < single_aspect_ratio:
                                merged_width = next_char["col_end"] - current_char["col_start"]
                                merged_height = max(current_char["height"], next_char["height"])
                                merged_ratio = merged_width / merged_height
                                
                                if min_aspect_ratio < merged_ratio < max_aspect_ratio:
                                    merged_char = {
                                        "char_id": current_char["char_id"],
                                        "line_id": line_id,
                                        "page_id": current_char["page_id"],
                                        "pdf_id": current_char["pdf_id"],
                                        "page_num": current_char["page_num"],
                                        "line_idx": current_char["line_idx"],
                                        "char_idx": current_char["char_idx"],
                                        "image_path": None,
                                        "col_start": current_char["col_start"],
                                        "col_end": next_char["col_end"],
                                        "width": merged_width,
                                        "height": merged_height,
                                        "abs_x": current_char["abs_x"],
                                        "abs_y": current_char["abs_y"],
                                        "abs_x_end": next_char["abs_x_end"],
                                        "abs_y_end": max(current_char["abs_y_end"], next_char["abs_y_end"]),
                                        "merged_from": [current_char["char_id"], next_char["char_id"]]
                                    }
                                    
                                    merged_chars.append(merged_char)
                                    del chars_ref[next_char["char_id"]]
                                    merged_count += 1
                                    i += 2
                                    continue
                
                merged_chars.append(current_char)
                i += 1
            
            if len(merged_chars) < len(char_ids):
                line_info["chars"] = [c["char_id"] for c in merged_chars]
                for char in merged_chars:
                    chars_ref[char["char_id"]] = char
        
        if merged_count > 0:
            print(f"   合并 {merged_count} 对粘连字符")
        
        self._mark_dirty()
    
    def process_pdf(self, pdf_path: str, skip_filter: bool = False, save_lineage: bool = True) -> Dict:
        """
        完整流程：PDF -> 页面 -> 行 -> 汉字
        
        Args:
            pdf_path: PDF文件路径
            skip_filter: 是否跳过行高度过滤，默认False
            save_lineage: 是否保存血缘索引到文件（批量模式下设为False，最后统一保存）
        
        Returns:
            血缘索引字典
        """
        import time
        
        start_total = time.time()
        timings = {}
        
        t1 = time.time()
        print(f"[1/5] PDF转图片: {pdf_path}")
        success, page_ids = self.pdf_to_images(pdf_path)
        if not success:
            print(f"Error: {page_ids}")
            return self.lineage
        t2 = time.time()
        timings["pdf_to_images"] = t2 - t1
        print(f"   生成 {len(page_ids)} 张页面图片 | 耗时: {timings['pdf_to_images']:.2f}s")
        
        t1 = time.time()
        all_line_ids = []
        print(f"[2/5] 图片转行...")
        
        for page_id in page_ids:
            success, line_ids = self.image_to_lines(page_id)
            if success:
                all_line_ids.extend(line_ids)
        t2 = time.time()
        timings["image_to_lines"] = t2 - t1
        print(f"   提取 {len(all_line_ids)} 行 | 耗时: {timings['image_to_lines']:.2f}s")
        
        t1 = time.time()
        print(f"[3/5] 行高度过滤...")
        if skip_filter:
            valid_line_ids = all_line_ids
            invalid_line_ids = []
        else:
            valid_line_ids, invalid_line_ids = self.filter_lines(all_line_ids)
        t2 = time.time()
        timings["filter_lines"] = t2 - t1
        print(f"   过滤结果: {len(valid_line_ids)} 有效, {len(invalid_line_ids)} 被拒绝 | 耗时: {timings['filter_lines']:.2f}s")
        
        t1 = time.time()
        print(f"[4/5] 规则切割与汉字提取...")
        rule_results = self.process_lines_by_rule(valid_line_ids)
        t2 = time.time()
        timings["rule_segmentation"] = t2 - t1
        print(f"   生成 {len(rule_results)} 个规则切割JSON | 耗时: {timings['rule_segmentation']:.2f}s")
        
        t1 = time.time()
        total_chars = 0
        for line_id, rule_data in rule_results.items():
            char_ids = self.extract_char_images(line_id, rule_data)
            total_chars += len(char_ids)
        t2 = time.time()
        timings["extract_chars"] = t2 - t1
        print(f"   提取 {total_chars} 个汉字 | 耗时: {timings['extract_chars']:.2f}s")
        
        t1 = time.time()
        print(f"[5/5] 后处理：合并粘连字符...")
        self.postprocess_merge_chars()
        t2 = time.time()
        timings["postprocess_merge"] = t2 - t1
        print(f"   耗时: {timings['postprocess_merge']:.2f}s")
        
        self.lineage["metadata"]["total_pdfs"] = len(self.lineage["pdfs"])
        self.lineage["metadata"]["total_pages"] = len(self.lineage["pages"])
        self.lineage["metadata"]["total_lines"] = len(self.lineage["lines"])
        self.lineage["metadata"]["total_chars"] = len(self.lineage["chars"])
        
        if save_lineage:
            self._save_lineage(force=True)
        
        end_total = time.time()
        total_time = end_total - start_total
        
        print(f"\n=== 性能统计 ===")
        print(f"总耗时: {total_time:.2f}s")
        print("-" * 50)
        for step, duration in timings.items():
            percentage = (duration / total_time) * 100
            print(f"{step}: {duration:.2f}s ({percentage:.1f}%)")
        print("-" * 50)
        
        return self.lineage
    
    def process_pdf_parallel(self, pdf_path: str, skip_filter: bool = False, max_workers: int = 4) -> Dict:
        """
        完整流程（并行版本）：PDF -> 页面 -> 行 -> 汉字
        
        Args:
            pdf_path: PDF文件路径
            skip_filter: 是否跳过行高度过滤，默认False
            max_workers: 最大并行工作线程数
        
        Returns:
            血缘索引字典
        """
        import time
        
        start_total = time.time()
        timings = {}
        
        t1 = time.time()
        print(f"[1/5] PDF转图片: {pdf_path}")
        success, page_ids = self.pdf_to_images(pdf_path)
        if not success:
            print(f"Error: {page_ids}")
            return self.lineage
        t2 = time.time()
        timings["pdf_to_images"] = t2 - t1
        print(f"   生成 {len(page_ids)} 张页面图片 | 耗时: {timings['pdf_to_images']:.2f}s")
        
        t1 = time.time()
        print(f"[2/5] 图片转行（并行处理 {max_workers} 进程）...")
        all_line_ids = []
        
        char_cfg_dict = {
            "lines_dir": self.char_cfg.lines_dir,
            "rule_json_dir": self.char_cfg.rule_json_dir
        }
        line_filter_params = {
            "min_h": self._line_filter.min_h,
            "max_h": self._line_filter.max_h
        }
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {
                executor.submit(
                    self._process_page_to_lines,
                    page_id,
                    self.lineage["pages"][page_id],
                    self.data_base_path,
                    char_cfg_dict,
                    line_filter_params
                ): page_id for page_id in page_ids
            }
            
            for future in as_completed(future_to_page):
                page_id = future_to_page[future]
                try:
                    success, line_ids, lineage_updates = future.result()
                    if success:
                        all_line_ids.extend(line_ids)
                        for line_id, line_info in lineage_updates["lines"].items():
                            self.lineage["lines"][line_id] = line_info
                        self.lineage["pages"][page_id]["lines"].extend(lineage_updates["page_lines"])
                except Exception as e:
                    print(f"   [ERROR] 页面处理失败: {page_id} - {str(e)}")
        t2 = time.time()
        timings["image_to_lines"] = t2 - t1
        print(f"   提取 {len(all_line_ids)} 行 | 耗时: {timings['image_to_lines']:.2f}s")
        
        t1 = time.time()
        print(f"[3/5] 行高度过滤...")
        if skip_filter:
            valid_line_ids = all_line_ids
            invalid_line_ids = []
        else:
            valid_line_ids, invalid_line_ids = self.filter_lines(all_line_ids)
        t2 = time.time()
        timings["filter_lines"] = t2 - t1
        print(f"   过滤结果: {len(valid_line_ids)} 有效, {len(invalid_line_ids)} 被拒绝 | 耗时: {timings['filter_lines']:.2f}s")
        
        t1 = time.time()
        print(f"[4/5] 规则切割与汉字提取（并行处理 {max_workers} 进程）...")
        
        rule_results = {}
        chunk_size = max(1, len(valid_line_ids) // max_workers)
        line_chunks = [valid_line_ids[i:i + chunk_size] for i in range(0, len(valid_line_ids), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(self._process_lines_chunk, chunk): chunk for chunk in line_chunks}
            
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    rule_results.update(chunk_results)
                except Exception as e:
                    print(f"   [ERROR] 行处理失败: {str(e)}")
        
        t2 = time.time()
        timings["rule_segmentation"] = t2 - t1
        print(f"   生成 {len(rule_results)} 个规则切割JSON | 耗时: {timings['rule_segmentation']:.2f}s")
        
        t1 = time.time()
        total_chars = 0
        for line_id, rule_data in rule_results.items():
            char_ids = self.extract_char_images(line_id, rule_data)
            total_chars += len(char_ids)
        t2 = time.time()
        timings["extract_chars"] = t2 - t1
        print(f"   提取 {total_chars} 个汉字 | 耗时: {timings['extract_chars']:.2f}s")
        
        t1 = time.time()
        print(f"[5/5] 后处理：合并粘连字符...")
        self.postprocess_merge_chars()
        t2 = time.time()
        timings["postprocess_merge"] = t2 - t1
        print(f"   耗时: {timings['postprocess_merge']:.2f}s")
        
        self.lineage["metadata"]["total_pdfs"] = len(self.lineage["pdfs"])
        self.lineage["metadata"]["total_pages"] = len(self.lineage["pages"])
        self.lineage["metadata"]["total_lines"] = len(self.lineage["lines"])
        self.lineage["metadata"]["total_chars"] = len(self.lineage["chars"])
        
        self._save_lineage(force=True)
        
        end_total = time.time()
        total_time = end_total - start_total
        
        print(f"\n=== 性能统计（并行模式，{max_workers}进程）===")
        print(f"总耗时: {total_time:.2f}s")
        print("-" * 50)
        for step, duration in timings.items():
            percentage = (duration / total_time) * 100
            print(f"{step}: {duration:.2f}s ({percentage:.1f}%)")
        print("-" * 50)
        
        return self.lineage
    
    def _process_page_to_lines(self, page_id: str, page_info: Dict, data_base_path: Path, char_cfg: Dict, line_filter_params: Dict) -> Tuple[bool, List[str], Dict]:
        """
        页面转行（用于并行处理）
        
        Args:
            page_id: 页面ID
            page_info: 页面信息字典
            data_base_path: 数据基础路径
            char_cfg: 字符分割配置
            line_filter_params: 行过滤参数
        
        Returns:
            (success, line_ids, lineage_updates)
        """
        try:
            img_path = data_base_path / page_info["image_path"]
            
            cfg = CharSegmentConfig()
            lines, original, h, w = SegmentManager._fast_line_detection(str(img_path), cfg)
            if not lines:
                return True, [], {}

            lines_dir = data_base_path / char_cfg["lines_dir"]
            lines_dir.mkdir(parents=True, exist_ok=True)

            pdf_id = page_info["pdf_id"]
            page_num = page_info["page_num"]
            min_h, max_h = line_filter_params["min_h"], line_filter_params["max_h"]
            
            # 预先分配列表容量
            num_lines = len(lines)
            line_ids = [None] * num_lines
            lineage_updates_lines = {}
            page_lines = []
            
            for lid in range(num_lines):
                y1, y2 = lines[lid]
                line_img = original[y1:y2, :]
                line_id = f"line_{page_id}_{lid}"
                line_save = lines_dir / f"{line_id}.png"
                cv2.imwrite(str(line_save), line_img)
                
                line_height = int(y2 - y1)
                is_valid = min_h <= line_height <= max_h
                
                line_info = {
                    "line_id": line_id,
                    "page_id": page_id,
                    "pdf_id": pdf_id,
                    "page_num": page_num,
                    "line_idx": lid,
                    "image_path": str(line_save.relative_to(data_base_path)),
                    "y_start": int(y1),
                    "y_end": int(y2),
                    "width": int(w),
                    "height": line_height,
                    "chars": [],
                    "annotation_status": "unannotated",
                    "confidence": 0.0,
                    "is_valid": is_valid
                }
                
                lineage_updates_lines[line_id] = line_info
                page_lines.append(line_id)
                line_ids[lid] = line_id

            return True, line_ids, {"lines": lineage_updates_lines, "page_lines": page_lines}

        except Exception as e:
            return False, [str(e)], {}
    
    def _process_lines_chunk(self, line_ids: List[str]) -> Dict[str, Dict]:
        """
        处理行数据块（用于并行处理）
        
        Args:
            line_ids: 行ID列表
        
        Returns:
            处理结果字典
        """
        cfg = CharSegmentConfig()
        seg = VerticalProjectionSegmenter(cfg)
        
        rule_json_dir = self.data_base_path / self.char_cfg.rule_json_dir
        rule_json_dir.mkdir(parents=True, exist_ok=True)
        
        # 预计算时间戳
        created_at = datetime.now().isoformat()
        seg_version = self.char_cfg.segmentation_version
        
        # 缓存 lineage 引用
        lines_ref = self.lineage["lines"]
        
        results = {}
        for line_id in line_ids:
            try:
                line_info = lines_ref.get(line_id)
                if line_info is None:
                    continue
                
                line_path = self.data_base_path / line_info["image_path"]
                
                gray = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                
                h, w = gray.shape[:2]
                if w == 0 or h == 0:
                    continue
                
                segments = seg.get_segment_classes(gray)
                
                chars = []
                segments_type_start_end = []
                
                for seg_type, start, end in segments:
                    segments_type_start_end.append([int(seg_type), int(start), int(end)])
                    if seg_type == 1:
                        chars.append({
                            "col_start": int(start),
                            "col_end": int(end),
                            "width": int(end - start)
                        })
                
                rule_result = {
                    "line_id": line_id,
                    "image_path": str(line_path.relative_to(self.data_base_path)),
                    "chars": chars,
                    "segments_type_start_end": segments_type_start_end,
                    "total_chars": len(chars),
                    "image_width": w,
                    "image_height": h,
                    "total_segments": len(segments),
                    "segmentation_version": seg_version,
                    "created_at": created_at
                }
                
                # 紧凑格式写入
                rule_json_path = rule_json_dir / f"{line_id}_rule.json"
                with open(str(rule_json_path), 'w', encoding='utf-8') as f:
                    json.dump(rule_result, f, ensure_ascii=False, separators=(',', ':'))
                
                results[line_id] = rule_result
                
            except Exception:
                continue
        
        return results
    
    def summarize(self) -> None:
        """输出血缘索引统计信息"""
        meta = self.lineage.get("metadata", {})
        print(f"\n=== Process Summary ===")
        print(f"PDFs: {meta.get('total_pdfs', 0)}")
        print(f"Pages: {meta.get('total_pages', 0)}")
        print(f"Lines: {meta.get('total_lines', 0)}")
        print(f"Chars: {meta.get('total_chars', 0)}")
        print(f"Version: {meta.get('segmentation_version', 'N/A')}")
    
    def get_char_by_id(self, char_id: str) -> Optional[Dict]:
        """根据字符ID获取字符信息"""
        return self.lineage["chars"].get(char_id)
    
    def get_line_by_id(self, line_id: str) -> Optional[Dict]:
        """根据行ID获取行信息"""
        return self.lineage["lines"].get(line_id)
    
    def get_page_by_id(self, page_id: str) -> Optional[Dict]:
        """根据页面ID获取页面信息"""
        return self.lineage["pages"].get(page_id)
    
    def get_pdf_by_id(self, pdf_id: str) -> Optional[Dict]:
        """根据PDF ID获取PDF信息"""
        return self.lineage["pdfs"].get(pdf_id)
    
    def get_lines_by_page(self, page_id: str) -> List[Dict]:
        """获取指定页面的所有行"""
        if page_id not in self.lineage["pages"]:
            return []
        line_ids = self.lineage["pages"][page_id]["lines"]
        return [self.lineage["lines"].get(lid, {}) for lid in line_ids]
    
    def get_chars_by_line(self, line_id: str) -> List[Dict]:
        """获取指定行的所有字符"""
        if line_id not in self.lineage["lines"]:
            return []
        char_ids = self.lineage["lines"][line_id]["chars"]
        return [self.lineage["chars"].get(cid, {}) for cid in char_ids]


def run_segment(data_base_path: Path, pdf_path: Path, parallel: bool = False):
    """
    运行单个PDF文本分割流程
    
    Args:
        data_base_path (Path): 数据基础目录
        pdf_path (Path): PDF文件路径
        parallel (bool): 是否使用并行处理
    
    Returns:
        None
    """
    pdf_cfg = Pdf2ImageConfig()
    img_cfg = Image2LineConfig()
    char_cfg = Line2CharConfig()
    char_cfg.max_line_height = 50 
    char_cfg.min_line_height = 40 
    char_cfg.segmentation_version = "rule_v1"
    char_cfg.validate()
    
    print(f"[INFO] 数据基础目录: {data_base_path}")
    print(f"[INFO] PDF文件: {pdf_path}")
    print(f"[INFO] 处理模式: {'并行' if parallel else '串行'}")
    
    segment_manager = SegmentManager(
        pdf_cfg=pdf_cfg,
        img_cfg=img_cfg,
        char_cfg=char_cfg,
        data_base_path=data_base_path
    )
    
    if parallel:
        segment_manager.process_pdf_parallel(str(pdf_path))
    else:
        segment_manager.process_pdf(str(pdf_path))
    
    segment_manager.summarize()


def _process_pdf_standalone(pdf_path_str: str, data_base_path_str: str, config_dict: Dict) -> Dict:
    """
    独立处理单个PDF文件（用于PDF级别并行）
    
    Args:
        pdf_path_str: PDF文件路径（字符串，便于pickle序列化）
        data_base_path_str: 数据基础路径（字符串）
        config_dict: 配置参数字典
    
    Returns:
        该PDF的lineage数据字典
    """
    import sys
    import os
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from image_tools.pdf_config import Pdf2ImageConfig
    from image_tools.image_config import Image2LineConfig
    from image_tools.segment_config import Line2CharConfig
    from image_tools.segment_manager import SegmentManager
    
    pdf_cfg = Pdf2ImageConfig()
    img_cfg = Image2LineConfig()
    
    char_cfg = Line2CharConfig()
    char_cfg.max_line_height = config_dict.get("max_line_height", 50)
    char_cfg.min_line_height = config_dict.get("min_line_height", 40)
    char_cfg.segmentation_version = config_dict.get("segmentation_version", "rule_v1")
    char_cfg.validate()
    
    segment_manager = SegmentManager(
        pdf_cfg=pdf_cfg,
        img_cfg=img_cfg,
        char_cfg=char_cfg,
        data_base_path=Path(data_base_path_str)
    )
    
    # 并行模式下不保存lineage，由主进程统一合并写入
    segment_manager.process_pdf(pdf_path_str, save_lineage=False)
    
    return segment_manager.lineage


def run_segment_batch(
    data_base_path: Path, 
    pdf_pattern: str = "*.pdf",
    start_index: int = 0,
    end_index: int = None,
    parallel: bool = False,
    max_workers: int = 4
):
    """
    批量运行PDF文本分割流程（支持指定范围和进度显示）
    
    Args:
        data_base_path (Path): 数据基础目录
        pdf_pattern (str): PDF文件匹配模式，默认 "*.pdf"
        start_index (int): 开始处理的索引（从0开始），默认0
        end_index (int): 结束处理的索引（不包含），默认None表示处理到最后
        parallel (bool): 是否使用并行处理（PDF级别并行）
        max_workers (int): 并行处理的最大进程数
    
    Returns:
        None
    """
    pdf_cfg = Pdf2ImageConfig()
    img_cfg = Image2LineConfig()
    
    char_cfg = Line2CharConfig()
    char_cfg.max_line_height = 50 
    char_cfg.min_line_height = 40 
    char_cfg.segmentation_version = "rule_v1"
    char_cfg.validate()
    
    config_dict = {
        "max_line_height": char_cfg.max_line_height,
        "min_line_height": char_cfg.min_line_height,
        "segmentation_version": char_cfg.segmentation_version
    }
    
    pdf_dir = data_base_path / "raw" / "pdf"
    if not pdf_dir.exists():
        print(f"[ERROR] PDF目录不存在: {pdf_dir}")
        return
    
    pdf_files = sorted(pdf_dir.glob(pdf_pattern))
    total_pdfs = len(pdf_files)
    
    if end_index is None:
        end_index = total_pdfs
    
    start_index = max(0, min(start_index, total_pdfs))
    end_index = max(start_index, min(end_index, total_pdfs))
    
    selected_files = pdf_files[start_index:end_index]
    selected_count = len(selected_files)
    
    print(f"[INFO] 数据基础目录: {data_base_path}")
    print(f"[INFO] 发现 {total_pdfs} 个PDF文件")
    print(f"[INFO] 处理范围: [{start_index}:{end_index}]，共 {selected_count} 个文件")
    print(f"[INFO] 处理模式: {'PDF级别并行' if parallel else '串行'}")
    print(f"[INFO] 并行进程数: {max_workers}")
    print(f"[INFO] 输出目录已准备就绪")
    
    start_time = time.time()
    
    if parallel and selected_count > 1:
        print(f"\n[BATCH] 启动PDF级别并行处理 ({max_workers} 进程)...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {
                executor.submit(
                    _process_pdf_standalone,
                    str(pdf_path),
                    str(data_base_path),
                    config_dict
                ): pdf_path for pdf_path in selected_files
            }
            
            all_lineages = []
            completed_count = 0
            
            if HAS_TQDM:
                with tqdm(total=selected_count, desc="[BATCH] 并行处理进度", unit="pdf",
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                    for future in as_completed(future_to_pdf):
                        pdf_path = future_to_pdf[future]
                        completed_count += 1
                        
                        try:
                            lineage = future.result()
                            all_lineages.append(lineage)
                            pbar.set_postfix({"完成": pdf_path.name})
                        except Exception as e:
                            print(f"\n[ERROR] 处理失败: {pdf_path.name} - {str(e)}")
                        
                        pbar.update(1)
            else:
                for future in as_completed(future_to_pdf):
                    pdf_path = future_to_pdf[future]
                    completed_count += 1
                    
                    try:
                        lineage = future.result()
                        all_lineages.append(lineage)
                        elapsed = time.time() - start_time
                        progress = _format_progress_bar(completed_count, selected_count)
                        print(f"[BATCH] {progress} [{completed_count}/{selected_count}] 完成: {pdf_path.name} | 已用时: {_format_time(elapsed)}")
                    except Exception as e:
                        print(f"[ERROR] 处理失败: {pdf_path.name} - {str(e)}")
        
        print(f"\n[BATCH] 合并 {len(all_lineages)} 个PDF的血缘索引...")
        segment_manager = SegmentManager(
            pdf_cfg=pdf_cfg,
            img_cfg=img_cfg,
            char_cfg=char_cfg,
            data_base_path=data_base_path
        )
        
        for lineage in all_lineages:
            segment_manager.lineage["pdfs"].update(lineage.get("pdfs", {}))
            segment_manager.lineage["pages"].update(lineage.get("pages", {}))
            segment_manager.lineage["lines"].update(lineage.get("lines", {}))
            segment_manager.lineage["chars"].update(lineage.get("chars", {}))
        
        segment_manager.lineage["metadata"]["total_pdfs"] = len(segment_manager.lineage["pdfs"])
        segment_manager.lineage["metadata"]["total_pages"] = len(segment_manager.lineage["pages"])
        segment_manager.lineage["metadata"]["total_lines"] = len(segment_manager.lineage["lines"])
        segment_manager.lineage["metadata"]["total_chars"] = len(segment_manager.lineage["chars"])
        
        segment_manager._save_lineage(force=True)
        print(f"[BATCH] 血缘索引已保存")
    
    else:
        segment_manager = SegmentManager(
            pdf_cfg=pdf_cfg,
            img_cfg=img_cfg,
            char_cfg=char_cfg,
            data_base_path=data_base_path
        )
        
        if HAS_TQDM:
            with tqdm(total=selected_count, desc="[BATCH] 处理进度", unit="pdf", 
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                for pdf_path in selected_files:
                    pbar.set_postfix({"文件": pdf_path.name})
                    
                    try:
                        # 批量模式下不单独保存，最后统一写入
                        segment_manager.process_pdf(str(pdf_path), save_lineage=False)
                    except Exception as e:
                        print(f"\n[ERROR] 处理失败: {pdf_path.name} - {str(e)}")
                    
                    pbar.update(1)
        else:
            for idx, pdf_path in enumerate(selected_files, 1):
                elapsed_time = time.time() - start_time
                remaining_files = selected_count - idx
                avg_time_per_file = elapsed_time / idx if idx > 0 else 0
                estimated_remaining = avg_time_per_file * remaining_files
                
                progress_bar = _format_progress_bar(idx, selected_count)
                
                print(f"\n{'='*60}")
                print(f"[BATCH] {progress_bar} [{idx}/{selected_count}]: {pdf_path.name}")
                print(f"[BATCH] 已用时: {_format_time(elapsed_time)} | 预计剩余: {_format_time(estimated_remaining)}")
                print(f"{'='*60}")
                
                try:
                    segment_manager.process_pdf(str(pdf_path), save_lineage=False)
                except Exception as e:
                    print(f"[ERROR] 处理失败: {pdf_path.name} - {str(e)}")
    
    # 批量处理结束后统一保存血缘索引
    segment_manager.lineage["metadata"]["total_pdfs"] = len(segment_manager.lineage["pdfs"])
    segment_manager.lineage["metadata"]["total_pages"] = len(segment_manager.lineage["pages"])
    segment_manager.lineage["metadata"]["total_lines"] = len(segment_manager.lineage["lines"])
    segment_manager.lineage["metadata"]["total_chars"] = len(segment_manager.lineage["chars"])
    segment_manager._save_lineage(force=True)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[BATCH] 批量处理完成")
    print(f"[BATCH] 总用时: {_format_time(total_time)}")
    print(f"{'='*60}")
    
    segment_manager.summarize()


def _format_progress_bar(current: int, total: int, bar_length: int = 20) -> str:
    """
    格式化进度条
    
    Args:
        current (int): 当前进度
        total (int): 总数量
        bar_length (int): 进度条长度
    
    Returns:
        格式化的进度条字符串
    """
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    return f"[{bar}] {percent:.1%}"


def _format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds (float): 秒数
    
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}分{secs}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}时{minutes}分{secs}秒"


import click


@click.group()
def cli():
    """PDF 文本分割流程管理"""
    pass


@cli.command()
@click.argument("pdf_name", type=str)
@click.option("--parallel", is_flag=True, help="页面级并行处理")
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录（默认: <项目根>/datahome）")
def single(pdf_name, parallel, data_base_path):
    """处理单个 PDF 文件"""
    base_dir = Path(__file__).resolve().parent.parent
    data_path = Path(data_base_path) if data_base_path else base_dir / "datahome"
    pdf_path = data_path / "raw" / "pdf" / pdf_name
    if not pdf_path.exists():
        click.echo(f"[ERROR] PDF 文件不存在: {pdf_path}", err=True)
        sys.exit(1)
    run_segment(data_path, pdf_path, parallel=parallel)


@cli.command()
@click.option("--start", type=int, default=0, show_default=True, help="起始索引（从0开始）")
@click.option("--end", type=int, default=None, help="结束索引（不包含）")
@click.option("--parallel", is_flag=True, help="PDF 级别并行处理")
@click.option("--max-workers", type=int, default=4, show_default=True, help="并行进程数")
@click.option("--pattern", type=str, default="*.pdf", show_default=True, help="PDF 文件匹配模式")
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录（默认: <项目根>/datahome）")
def batch(start, end, parallel, max_workers, pattern, data_base_path):
    """批量处理 PDF 文件"""
    base_dir = Path(__file__).resolve().parent.parent
    data_path = Path(data_base_path) if data_base_path else base_dir / "datahome"
    run_segment_batch(
        data_path,
        pdf_pattern=pattern,
        start_index=start,
        end_index=end,
        parallel=parallel,
        max_workers=max_workers
    )


if __name__ == "__main__":
    cli()