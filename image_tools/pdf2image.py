import fitz  # PyMuPDF
import os
from pathlib import Path

def create_dir(dir_path):
    """创建目录（不存在则创建）"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"图片保存目录：{os.path.abspath(dir_path)}")

def pdf_to_images_pymupdf(pdf_path, output_dir="pdf_images", dpi=300, output_format="png", page_range=None, grayscale=True):
    """
    PyMuPDF实现PDF转图片（无系统依赖）
    
    Args:
        pdf_path: PDF文件路径（绝对/相对）
        output_dir: 图片保存目录，默认为"pdf_images"
        dpi: 分辨率，默认300DPI（适配印刷体）
        output_format: 输出格式，png（无损）或jpg（压缩）
        page_range: 转换页码范围，None=全部，如(1, 5)=第1-5页
        grayscale: 是否输出灰度图，默认True（适配后续图像处理）
    
    Returns:
        转换后的图片路径列表
    """
    # 校验PDF文件
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在：{pdf_path}")
    
    # 创建输出目录
    create_dir(output_dir)
    image_paths = []
    
    # 打开PDF文档
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    print(f"[PDF2Image] PDF文件: {os.path.basename(pdf_path)}")
    print(f"[PDF2Image] 总页数: {total_pages}")
    print(f"[PDF2Image] 输出目录: {os.path.abspath(output_dir)}")
    print(f"[PDF2Image] 开始转换...")
    
    # 处理页码范围（转换为0-based索引）
    start_page = 0
    end_page = total_pages - 1
    if page_range:
        start_page = max(0, page_range[0] - 1)
        end_page = min(total_pages - 1, page_range[1] - 1)
    
    # 逐页转换
    for page_idx in range(start_page, end_page + 1):
        # 加载页面
        page = doc.load_page(page_idx)
        # 计算缩放矩阵（PDF默认72DPI，缩放至目标DPI）
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        # 渲染页面为图像对象
        pix = page.get_pixmap(
            matrix=mat,
            alpha=False,        # 去除透明通道
            colorspace=fitz.csGRAY if grayscale else fitz.csRGB,  # 灰度/RGB
        )
        
        # 保存图片
        page_num = page_idx + 1
        img_name = f"page_{page_num}.{output_format}"
        img_path = os.path.join(output_dir, img_name)
        pix.save(img_path)
        
        image_paths.append(img_path)
    
    # 关闭文档
    doc.close()
    print(f"[PDF2Image] 转换完成！共生成 {len(image_paths)} 张图片")
    return image_paths

# ========== 运行示例 ==========
if __name__ == "__main__":
    # 替换为你的PDF文件路径
    PDF_FILE_PATH = Path(__file__).resolve().parent / "gwyb195521.pdf"
    try:
        # 使用默认配置转换
        pdf_to_images_pymupdf(str(PDF_FILE_PATH.absolute()))
    except Exception as e:
        print(f"转换失败：{str(e)}")