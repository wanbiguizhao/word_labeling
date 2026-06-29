import cv2
import numpy as np
import os
import traceback
import json
from typing import List, Tuple, Dict, Any, Optional

"""
印刷体字符连通域彩色可视化工具 + 工程化数据集切割流水线
完整功能：
1. 批量行切割 → 输出行图片 + 行JSON（可还原原图）
2. 批量列切割 → 不在生成图片，在行图上画字符分割线
3. 命令行工具：根据JSON在原图上绘制切割框
4. 全流程结构化存储，无胶水代码混乱
"""

# ===================== 配置类 =====================
class CharSegmentConfig:
    PREPROCESS_SAVE_DIR: str = "sticky_char_results"
    CHAR_SAVE_DIR: str = "split_char_images"
    SAVE_PREPROCESS_STEPS: bool = False

    DENOISE_H: int = 10
    BILATERAL_D: int = 5
    BILATERAL_SIGMA_COLOR: int = 75
    BILATERAL_SIGMA_SPACE: int = 75
    ERODE_KERNEL_SIZE: Tuple[int, int] = (1, 1)
    ERODE_ITERATIONS: int = 1
    ADAPTIVE_THRESH_BLOCK_SIZE: int = 15
    ADAPTIVE_THRESH_C: int = 3

    LINE_PROJ_THRESHOLD: float = 0.02
    LINE_MIN_HEIGHT: int = 10
    LINE_EXPAND_PIXEL: int = 2
    TOP_NOISE_ROWS: int = 20
    LINE_EMPTY_RATIO: float = 0.005

    CHAR_MIN_AREA: int = 40

    VERT_PROJ_THRESHOLD: float = 0.01
    VERT_MIN_COL_WIDTH: int = 2
    VERT_GAP_MIN_WIDTH: int = 1
    SEG_LINE_COLOR: Tuple[int, int, int] = (0, 0, 255)
    SEG_LINE_THICKNESS: int = 1

    LINE_HORIZONTAL_CROP_THRESH: float = 0.001
    LINE_CROP_PADDING: int = 1

    @classmethod
    def validate(cls) -> None:
        assert cls.DENOISE_H >= 0 and cls.DENOISE_H <= 15, "去噪强度需在0-15之间"
        assert cls.ADAPTIVE_THRESH_BLOCK_SIZE % 2 == 1, "自适应二值化块大小必须为奇数"
        assert cls.SEG_LINE_THICKNESS >= 1, "分割线宽度至少为1像素"
        assert cls.LINE_CROP_PADDING >= 0, "裁剪填充像素不能为负数"

# ===================== 工具类 =====================
class ImageUtils:
    @staticmethod
    def create_dir(dir_path: str) -> None:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def imwrite_utf8(save_path: str, img: np.ndarray) -> bool:
        try:
            success, img_bytes = cv2.imencode('.png', img)
            if not success:
                return False
            with open(save_path, 'wb') as f:
                f.write(img_bytes)
            return True
        except:
            return False

    @staticmethod
    def is_empty_roi(roi: np.ndarray, threshold_ratio: float = 0.005) -> bool:
        if roi is None or roi.size == 0:
            return True
        total_pixels = roi.shape[0] * roi.shape[1]
        non_zero_pixels = np.count_nonzero(roi)
        non_zero_ratio = non_zero_pixels / total_pixels if total_pixels > 0 else 0.0
        return non_zero_ratio < threshold_ratio

    @staticmethod
    def crop_line_horizontal_blank(line_roi: np.ndarray, crop_thresh: float = 0.001, padding: int = 1):
        if ImageUtils.is_empty_roi(line_roi):
            return line_roi, 0, line_roi.shape[1]
        col_proj = np.sum(line_roi, axis=0) / 255
        col_total = line_roi.shape[0]
        col_non_zero_ratio = col_proj / col_total if col_total > 0 else 0
        valid_cols = np.where(col_non_zero_ratio > crop_thresh)[0]
        if len(valid_cols) == 0:
            return line_roi, 0, line_roi.shape[1]
        left_col = max(0, valid_cols[0] - padding)
        right_col = min(line_roi.shape[1], valid_cols[-1] + 1 + padding)
        cropped_roi = line_roi[:, left_col:right_col]
        return cropped_roi, left_col, right_col

    @staticmethod
    def save_char_roi(char_roi, save_dir, line_idx, char_idx, char_x, char_y, char_w, char_h):
        line_str = f"{line_idx+1:02d}"
        char_str = f"{char_idx+1:02d}"
        filename = f"line_{line_str}_char_{char_str}_x{char_x}y{char_y}w{char_w}h{char_h}.png"
        save_path = os.path.join(save_dir, filename)
        ImageUtils.imwrite_utf8(save_path, char_roi)

# ===================== 预处理 =====================
class Preprocessor:
    def __init__(self, config: CharSegmentConfig):
        self.config = config
        self.save_dir = config.PREPROCESS_SAVE_DIR
        #ImageUtils.create_dir(self.save_dir)

    def process(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像：{image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        denoised = cv2.fastNlMeansDenoising(gray, None, self.config.DENOISE_H, 7, 21)
        bilateral = cv2.bilateralFilter(denoised, self.config.BILATERAL_D, self.config.BILATERAL_SIGMA_COLOR, self.config.BILATERAL_SIGMA_SPACE)
        thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.config.ADAPTIVE_THRESH_BLOCK_SIZE, self.config.ADAPTIVE_THRESH_C)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, self.config.ERODE_KERNEL_SIZE)
        eroded = cv2.erode(thresh, kernel_erode, iterations=self.config.ERODE_ITERATIONS)
        closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))
        processed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1,1)))
        return processed, img, h, w

# ===================== 连通域分析 =====================
class ConnectedComponentAnalyzer:
    def analyze(self, binary_img):
        if len(binary_img.shape) > 2:
            binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
        img = (binary_img == 255).astype(np.int32)
        h, w = img.shape
        label_img = np.zeros_like(img, dtype=np.int32)
        current_label = 1
        equivalence = {}
        for y in range(h):
            for x in range(w):
                if img[y, x] == 0: continue
                left = label_img[y, x-1] if x>0 else 0
                up = label_img[y-1, x] if y>0 else 0
                up_left = label_img[y-1, x-1] if (y>0 and x>0) else 0
                up_right = label_img[y-1, x+1] if (y>0 and x<w-1) else 0
                valid = [v for v in [left, up, up_left, up_right] if v != 0]
                if not valid:
                    label_img[y, x] = current_label
                    current_label +=1
                else:
                    m = min(valid)
                    label_img[y, x] = m
                    for v in valid:
                        if v != m and v not in equivalence:
                            equivalence[v] = m
        def find_root(l):
            if l not in equivalence: return l
            equivalence[l] = find_root(equivalence[l])
            return equivalence[l]
        for l in range(1, current_label): find_root(l)
        for y in range(h):
            for x in range(w):
                l = label_img[y, x]
                if l !=0: label_img[y, x] = find_root(l)
        stats = {}
        for l in np.unique(label_img):
            if l ==0: continue
            ys, xs = np.where(label_img == l)
            stats[l] = {'x': xs.min(), 'y': ys.min(), 'w': xs.max()-xs.min()+1, 'h': ys.max()-ys.min()+1, 'area': len(xs), 'x_coords': xs, 'y_coords': ys}
        return label_img, stats

# ===================== 行检测 =====================
class TextLineDetector:
    def __init__(self, config: CharSegmentConfig):
        self.config = config
    def detect(self, processed_img: np.ndarray, img_h: int, img_w: int) -> List[Tuple[int, int]]:
        filtered = processed_img[self.config.TOP_NOISE_ROWS:]
        proj = np.sum(filtered, axis=1)/255
        if proj.max() ==0: return []
        th = proj.max() * self.config.LINE_PROJ_THRESHOLD
        indices = np.where(proj>th)[0]
        if len(indices)==0: return []
        lines = []
        s = e = indices[0]
        for i in indices[1:]:
            if i == e+1: e =i
            else:
                lines.append((s,e))
                s=e=i
        lines.append((s,e))
        res = []
        for ys, ye in lines:
            yso = ys + self.config.TOP_NOISE_ROWS
            yeo = ye + self.config.TOP_NOISE_ROWS
            ys2 = max(0, yso - self.config.LINE_EXPAND_PIXEL)
            ye2 = min(img_h-1, yeo + self.config.LINE_EXPAND_PIXEL)
            if ye2-ys2 < self.config.LINE_MIN_HEIGHT: continue
            res.append((ys2, ye2))
        return res

# ===================== 竖直投影切割 =====================
class VerticalProjectionSegmenter:
    def __init__(self, config: CharSegmentConfig):
        self.config = config

    def get_segment_classes(self, line_roi: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        输出格式：[(type, start, end), ...]
        type: 0=空白, 1=汉字, 2=其他
        相邻相同类型自动合并
        """
        h, w = line_roi.shape[:2]
        proj = np.sum(255 - line_roi, axis=0) / 255
        if proj.max() == 0:
            return [(0, 0, w - 1)]

        th_text = proj.max() * self.config.VERT_PROJ_THRESHOLD
        th_other = th_text * 0.5

        segments = []
        current_type = None
        segment_start = 0

        for x in range(w):
            val = proj[x]
            if val >= th_text:
                t = 1
            elif val >= th_other:
                t = 2
            else:
                t = 0

            if current_type is None:
                current_type = t
                segment_start = x
            else:
                if t != current_type:
                    segments.append((current_type, segment_start, x - 1))
                    current_type = t
                    segment_start = x

        segments.append((current_type, segment_start, w - 1))
        return segments

# ===================== 可视化 =====================
class ConnectedComponentVisualizer:
    def __init__(self, config: CharSegmentConfig):
        self.config = config
        self.cca = ConnectedComponentAnalyzer()
        self.line_detector = TextLineDetector(config)
        self.vertical_segmenter = VerticalProjectionSegmenter(config)
    def visualize(self, processed_img, original_img, h, w):
        vis = original_img.copy()
        return vis

# -------------------- 第一步：批量行切割 + 输出JSON --------------------
def run_step1_batch_line_cut(input_folder: str, line_out: str, json_out: str):
    cfg = CharSegmentConfig()
    ImageUtils.create_dir(line_out)
    ImageUtils.create_dir(json_out)
    pre = Preprocessor(cfg)
    detector = TextLineDetector(cfg)

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(input_folder, fname)
        try:
            proc_img, ori, h, w = pre.process(img_path)
            lines = detector.detect(proc_img, h, w)
            gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)

            page_json = {
                "image_name": fname,
                "image_path": img_path,
                "image_width": w,
                "image_height": h,
                "total_lines": len(lines),
                "lines": []
            }

            for lid, (y1, y2) in enumerate(lines):
                line_img = gray[y1:y2, :]
                line_save = os.path.join(line_out, f"{fname}_line_{lid}.png")
                cv2.imwrite(line_save, line_img)
                char_json = os.path.join(json_out, f"{fname}_line_{lid}_chars.json")
                page_json["lines"].append({
                    "line_id": lid,
                    "y_start": int(y1),
                    "y_end": int(y2),
                    "x_start": 0,
                    "x_end": int(w),
                    "width": int(w),
                    "height": int(y2 - y1),
                    "save_path": line_save,
                    "char_json_path": char_json
                })

            json_path = os.path.join(json_out, f"{fname}_lines.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(page_json, f, ensure_ascii=False, indent=2)
            print(f"✅ 行切割完成：{json_path}")
        except Exception as e:
            print(f"❌ 失败：{fname} {str(e)}")

# -------------------- 第二步：批量单字切割 + 输出JSON --------------------
# -------------------- 第二步：批量单字切割 + 输出JSON --------------------
def run_step2_batch_char_cut(json_folder: str, char_out: str):
    cfg = CharSegmentConfig()
    seg = VerticalProjectionSegmenter(cfg)
    ImageUtils.create_dir(char_out)

    # 🔥 划线图片保存目录（新目录，不污染原行图片）
    LINE_MARKED_DIR = "output\\lines_marked"
    ImageUtils.create_dir(LINE_MARKED_DIR)

    for jname in os.listdir(json_folder):
        if not jname.endswith("_lines.json"):
            continue
        jpath = os.path.join(json_folder, jname)
        with open(jpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for line in data["lines"]:
            lid = line["line_id"]
            y1 = line["y_start"]
            y2 = line["y_end"]
            w = line["width"]
            line_img_path = line["save_path"]
            char_json_path = line["char_json_path"]

            gray = cv2.imread(line_img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue

            # ==============================================
            # 核心：获取分段结果 (type, start, end)
            # ==============================================
            segments = seg.get_segment_classes(gray)

            # ==============================================
            # 核心：只保留汉字段 type == 1
            # ==============================================
            char_segments = []
            for typ, s, e in segments:
                if typ == 1:
                    char_segments.append((s, e))

            # ==============================================
            # 🔥 在行图上画分割线（带判断：start=end 时上红下绿）
            # ==============================================
            line_color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            h_line = line_color_img.shape[0]
            half_h = h_line // 2  # 行高度一半

            for (start_x, end_x) in char_segments:
                if start_x == end_x:
                    # 情况1：起点终点重合 → 上半红，下半绿
                    x = start_x
                    # 上半部分：红色
                    cv2.line(line_color_img, (x, 0), (x, half_h), (0, 0, 255), 1)
                    # 下半部分：绿色
                    cv2.line(line_color_img, (x, half_h), (x, h_line - 1), (0, 255, 0), 1)
                else:
                    # 情况2：正常区间 → start红，end绿
                    cv2.line(line_color_img, (start_x, 0), (start_x, h_line - 1), (0, 0, 255), 1)
                    cv2.line(line_color_img, (end_x, 0), (end_x, h_line - 1), (0, 255, 0), 1)
            # ==============================================
            # 🔥 新保存路径：自动加后缀 _marked.png
            # ==============================================
            orig_line_name = os.path.basename(line_img_path)
            name_no_ext = os.path.splitext(orig_line_name)[0]
            marked_line_name = f"{name_no_ext}_marked.png"  # 👈 关键后缀
            marked_line_path = os.path.join(LINE_MARKED_DIR, marked_line_name)

            # 保存带划线的图片（新路径，不覆盖原图）
            cv2.imwrite(marked_line_path, line_color_img)

            # ==============================================
            # 构建chars信息
            # ==============================================
            chars = []
            for cid, (s, e) in enumerate(char_segments):
                cw = e - s + 1
                if cw < 2:
                    continue

                chars.append({
                    "char_id": cid,
                    "type": "CHAR",
                    "col_start": int(s),
                    "col_end": int(e),
                    "abs_x_start": int(s),
                    "abs_x_end": int(e),
                    "abs_y_start": int(y1),
                    "abs_y_end": int(y2),
                    "width": int(cw),
                    "height": int(y2 - y1),
                    "marked_line_path": marked_line_path  # 👈 新路径存入JSON
                })

            char_data = {
                "line_id": lid,
                "parent_image": data["image_name"],
                "line_y_start": y1,
                "line_y_end": y2,
                "total_chars": len(chars),
                "segments_type_start_end": segments,
                "marked_line_path": marked_line_path,  # 👈 保存划线路径
                "chars": chars
            }
            with open(char_json_path, 'w', encoding='utf-8') as f:
                json.dump(char_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 字符切割与划线完成：{jname}")
        
# -------------------- 第三步：命令行工具 —— 绘制回原图 --------------------
def run_step3_draw_original(original_img_path: str, line_json_path: str):
    img = cv2.imread(original_img_path)
    if img is None:
        print("图片读取失败")
        return

    with open(line_json_path, 'r', encoding='utf-8') as f:
        line_data = json.load(f)

    for line in line_data["lines"]:
        y1 = line["y_start"]
        y2 = line["y_end"]
        x_start_line = line["x_start"]
        x_end_line = line["x_end"]

        cv2.rectangle(img, (x_start_line, y1), (x_end_line, y2), (0, 255, 0), 2)

        cjson = line["char_json_path"]
        if not os.path.exists(cjson):
            continue

        with open(cjson, 'r', encoding='utf-8') as f:
            char_data = json.load(f)

        for c in char_data["chars"]:
            rel_x1 = c["col_start"]
            rel_x2 = c["col_end"]
            abs_x1 = x_start_line + rel_x1
            abs_x2 = x_start_line + rel_x2
            abs_y1 = c["abs_y_start"]
            abs_y2 = c["abs_y_end"]
            cv2.rectangle(img, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 1)

    cv2.imshow("✅ 原图切割回溯", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------- 主入口：一键全流程 --------------------
if __name__ == "__main__":
    INPUT_FOLDER = "pdf_images"
    LINE_OUTPUT = "output\\lines_output_all"
    CHAR_OUTPUT = "output\\chars_output"
    JSON_OUTPUT = "output\\cut_json"

    print("🚀 开始全流程数据集切割")
    run_step1_batch_line_cut(INPUT_FOLDER, LINE_OUTPUT, JSON_OUTPUT)
    run_step2_batch_char_cut(JSON_OUTPUT, CHAR_OUTPUT)
    print("🎉 全流程完成！")

    # 测试画图
    #run_step3_draw_original("pdf_images\\page_11.png", "output\\cut_json\\page_11.png_lines.json")