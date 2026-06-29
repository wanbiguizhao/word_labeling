from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path

@dataclass
class Pdf2ImageConfig:
    """PDF转图片配置类"""
    
    dpi: int = 300  # 分辨率：DPI值（300适配印刷体，兼顾清晰与速度）
    output_format: str = "png"  # 输出格式：png（无损）/ jpg（压缩）
    output_dir: str = "pdf_images"  # 图片保存目录
    page_range: Optional[Tuple[int, int]] = None  # 转换页码范围：None=全部，如(1, 5)=第1-5页
    grayscale: bool = True  # 是否输出灰度图：True=灰度（适配后续图像处理）

    def validate(self) -> None:
        """验证配置参数"""
        assert self.dpi > 0, "DPI must be positive"
        assert self.output_format in ["png", "jpg", "jpeg"], "Unsupported format"
        assert self.page_range is None or (
            len(self.page_range) == 2 and self.page_range[0] <= self.page_range[1]
        ), "Invalid page_range"

    @property
    def output_path(self) -> Path:
        """获取输出目录路径"""
        return Path(self.output_dir)