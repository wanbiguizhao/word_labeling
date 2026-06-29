from dataclasses import dataclass
from typing import Tuple

@dataclass
class Image2LineConfig:
    """图片转行配置类"""
    
    output_dir: str = "line_images"  # 行图片输出目录
    save_steps: bool = False  # 是否保存处理中间步骤图片

    # ===== 去噪与二值化 =====
    denoise_h: int = 10  # 去噪强度：0-15，值越大去噪越强但可能丢失细节
    bilateral_d: int = 5  # 双边滤波直径：值越大模糊范围越大
    bilateral_sigma_color: int = 75  # 双边滤波颜色sigma：越大表示颜色差异容忍度越高
    bilateral_sigma_space: int = 75  # 双边滤波空间sigma：越大表示空间距离容忍度越高
    erode_kernel: Tuple[int, int] = (1, 1)  # 腐蚀操作核大小：(1,1)表示不腐蚀
    erode_iterations: int = 1  # 腐蚀迭代次数：值越大腐蚀越严重
    adaptive_thresh_block_size: int = 15  # 自适应二值化块大小：必须为奇数，值越大考虑越大范围
    adaptive_thresh_c: int = 3  # 自适应二值化C值：从平均值减去的常数

    # ===== 行切割 =====
    line_proj_threshold: float = 0.02  # 行投影阈值：投影值低于此比例认为是空白行
    line_min_height: int = 10  # 最小行高度：小于此值的行会被过滤
    line_expand_pixel: int = 2  # 行扩展像素：上下扩展的像素数
    top_noise_rows: int = 20  # 顶部噪声行数：忽略顶部多少行
    line_empty_ratio: float = 0.005  # 行空白比例：超过此比例认为是非字符行

    # ===== 字符切割 =====
    char_min_area: int = 40  # 最小字符面积：小于此面积的连通域被忽略

    # ===== 垂直投影 =====
    vert_proj_threshold: float = 0.01  # 垂直投影阈值：确定字符边界
    vert_min_col_width: int = 2  # 最小列宽度：字符最小宽度
    vert_gap_min_width: int = 1  # 间隔最小宽度：字符间隔最小宽度
    seg_line_color: Tuple[int, int, int] = (0, 0, 255)  # 分割线颜色：BGR格式
    seg_line_thickness: int = 1  # 分割线粗细：像素数

    # ===== 行裁剪 =====
    line_horizontal_crop_thresh: float = 0.001  # 行水平裁剪阈值：列投影低于此比例裁剪
    line_crop_padding: int = 1  # 裁剪填充像素：裁剪后左右各加的像素

    def validate(self) -> None:
        """验证配置参数"""
        assert self.denoise_h >= 0 and self.denoise_h <= 15, "去噪强度需在0-15之间"
        assert self.adaptive_thresh_block_size % 2 == 1, "自适应二值化块大小必须为奇数"
        assert self.seg_line_thickness >= 1, "分割线粗细至少为1像素"
        assert self.line_crop_padding >= 0, "裁剪填充像素不能为负数"