"""
字符切割一站式命令行工具

统一入口，使用 click 框架管理所有子命令。

使用方式：
    python cli.py --help
    python cli.py <command> --help
"""

import sys
from pathlib import Path

import click

# 确保项目根目录在 sys.path 中（即使在其它目录执行）
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# 命令组：segment - 数据分割流程
# ============================================================
@click.group()
def segment():
    """PDF 文本分割流程：PDF → 页面 → 行 → 汉字"""


@segment.command("single")
@click.argument("pdf_name", type=str)
@click.option("--parallel", is_flag=True, help="页面级并行处理")
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录（默认: <项目根>/datahome）")
def segment_single(pdf_name, parallel, data_base_path):
    """处理单个 PDF 文件"""
    from image_tools.segment_manager import run_segment
    data_path = Path(data_base_path) if data_base_path else BASE_DIR / "datahome"
    pdf_path = data_path / "raw" / "pdf" / pdf_name
    if not pdf_path.exists():
        click.echo(f"[ERROR] PDF 文件不存在: {pdf_path}", err=True)
        sys.exit(1)
    run_segment(data_path, pdf_path, parallel=parallel)


@segment.command("batch")
@click.option("--start", type=int, default=0, show_default=True, help="起始索引（从0开始）")
@click.option("--end", type=int, default=None, help="结束索引（不包含）")
@click.option("--parallel", is_flag=True, help="PDF 级别并行处理")
@click.option("--max-workers", type=int, default=4, show_default=True, help="并行进程数")
@click.option("--pattern", type=str, default="*.pdf", show_default=True, help="PDF 文件匹配模式")
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录（默认: <项目根>/datahome）")
def segment_batch(start, end, parallel, max_workers, pattern, data_base_path):
    """批量处理 PDF 文件"""
    from image_tools.segment_manager import run_segment_batch
    data_path = Path(data_base_path) if data_base_path else BASE_DIR / "datahome"
    run_segment_batch(
        data_path,
        pdf_pattern=pattern,
        start_index=start,
        end_index=end,
        parallel=parallel,
        max_workers=max_workers
    )


# ============================================================
# 命令组：train - 模型训练相关
# ============================================================
@click.group()
def train():
    """训练深度学习分割模型"""


@train.command("start")
@click.option("--batch-size", type=int, default=8, show_default=True, help="批大小")
@click.option("--lr", "learning_rate", type=float, default=1e-4, show_default=True, help="学习率")
@click.option("--epochs", "num_epochs", type=int, default=50, show_default=True, help="训练轮数")
@click.option("--train-ratio", type=float, default=0.8, show_default=True, help="训练集比例")
@click.option("--device", type=str, default="auto", show_default=True,
              help='训练设备: "auto", "cuda", "cpu"')
@click.option("--num-workers", type=int, default=4, show_default=True, help="DataLoader 并行数")
@click.option("--use-amp/--no-amp", default=True, help="是否启用混合精度训练")
@click.option("--checkpoint-dir", type=str, default="models", show_default=True, help="模型保存目录")
@click.option("--data-base-path", type=str, default="datahome", show_default=True,
              help="数据基础目录（相对项目根）")
@click.option("--split-file", type=str, default="ai_model/data/dataset_split.json",
              show_default=True, help="数据集划分文件路径（相对项目根）")
@click.option("--seed", type=int, default=42, show_default=True, help="随机种子")
def train_start(batch_size, learning_rate, num_epochs, train_ratio,
                device, num_workers, use_amp, checkpoint_dir,
                data_base_path, split_file, seed):
    """训练分割模型"""
    from ai_model.train.train_config import TrainConfig
    from ai_model.train.train import main as train_main

    cfg = TrainConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        train_ratio=train_ratio,
        device=device,
        num_workers=num_workers,
        use_amp=use_amp,
        checkpoint_dir=checkpoint_dir,
        data_base_path=data_base_path,
        split_file=split_file,
        seed=seed
    )
    train_main(cfg)


@train.command("split-dataset")
@click.option("--train-ratio", type=float, default=0.8, show_default=True, help="训练集比例")
@click.option("--seed", type=int, default=42, show_default=True, help="随机种子")
@click.option("--sort-by-width", is_flag=True, help="按宽度排序后划分")
@click.option("--data-base-path", type=str, default="datahome", show_default=True,
              help="数据基础目录（相对项目根）")
@click.option("--output", type=str, default="ai_model/data/dataset_split.json",
              show_default=True, help="划分文件输出路径（相对项目根）")
def train_split_dataset(train_ratio, seed, sort_by_width, data_base_path, output):
    """生成数据集划分文件"""
    from ai_model.data.generate_dataset_split import generate_dataset_split

    data_path = BASE_DIR / data_base_path
    output_path = BASE_DIR / output

    split_info = generate_dataset_split(
        data_base_path=data_path,
        output_path=output_path,
        train_ratio=train_ratio,
        seed=seed,
        sort_by_width=sort_by_width
    )

    click.echo(f"[INFO] 数据集划分完成!")
    click.echo(f"[INFO] 总样本数: {split_info['stats']['total_count']}")
    click.echo(f"[INFO] 训练集: {split_info['stats']['train_count']}")
    click.echo(f"[INFO] 验证集: {split_info['stats']['val_count']}")
    click.echo(f"[INFO] 输出文件: {output_path}")


@train.command("active-learn")
@click.argument("top_n", type=int, default=100)
@click.option("--model-path", type=click.Path(exists=True), default=None,
              help="模型权重路径（默认: models/char_segment_1d_unet_best.pth）")
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录（默认: <项目根>/datahome）")
@click.option("--batch-size", type=int, default=32, show_default=True,
              help="GPU批量推理批大小")
@click.option("--num-workers", type=int, default=4, show_default=True,
              help="并行特征提取和评分的线程数")
def train_active_learn(top_n, model_path, data_base_path, batch_size, num_workers):
    """主动学习：找出最需要标注的行（使用GPU批量并行推理加速）"""
    from ai_model.train.active_learning import ActiveLearner

    data_path = Path(data_base_path) if data_base_path else BASE_DIR / "datahome"
    model_dir = BASE_DIR / "ai_model" / "models"

    if model_path:
        model_file = Path(model_path)
    else:
        model_file = model_dir / "char_segment_1d_unet_best.pth"

    if not model_file.exists():
        click.echo(f"[ERROR] 模型文件不存在: {model_file}", err=True)
        sys.exit(1)

    click.echo(f"[INFO] 加载模型: {model_file}")
    learner = ActiveLearner(model_file, data_path)
    click.echo(f"[INFO] 开始计算主动学习分数（batch_size={batch_size}, num_workers={num_workers}）...")
    ranked_lines = learner.rank_lines_batched(top_n, batch_size=batch_size, num_workers=num_workers)

    click.echo(f"\n[INFO] Top {len(ranked_lines)} 需要优先标注的行：")
    click.echo("-" * 120)
    click.echo(f"{'排名':<4} {'行ID':<40} {'AL分数':<10} {'不确定性':<10} {'分歧':<10} {'数量差异':<10}")
    click.echo("-" * 120)

    for idx, item in enumerate(ranked_lines, 1):
        click.echo(f"{idx:<4} {item['line_id']:<40} {item['al_score']:<10.4f} "
                   f"{item['uncertainty']:<10.4f} {item['disagreement']:<10.4f} "
                   f"{item['count_diff']:<10.4f}")

    output_path = model_dir / "al_ranking.json"
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ranked_lines, f, indent=2, ensure_ascii=False)
    click.echo(f"\n[INFO] 排名结果已保存到: {output_path}")


# ============================================================
# 命令组：predict - 模型推理
# ============================================================
@click.group()
def predict():
    """使用训练好的模型进行推理"""


@predict.command("line")
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--model-path", type=click.Path(exists=True), default=None,
              help="模型权重路径")
@click.option("--output", type=str, default=None,
              help="可视化结果保存路径（默认不保存）")
@click.option("--threshold", type=float, default=0.5, show_default=True,
              help="字符概率阈值")
@click.option("--max-gap", type=int, default=2, show_default=True,
              help="合并间隙（特征像素，设为 -1 禁用合并）")
def predict_line(image_path, model_path, output, threshold, max_gap):
    """对单行图像进行字符分割推理"""
    import cv2
    import numpy as np
    from ai_model.inference.infer import CharSegmentPredictor

    base_dir = BASE_DIR / "ai_model" / "models"
    if model_path:
        model_file = Path(model_path)
    else:
        model_file = base_dir / "char_segment_1d_unet_best.pth"

    if not model_file.exists():
        click.echo(f"[ERROR] 模型文件不存在: {model_file}", err=True)
        sys.exit(1)

    click.echo(f"[INFO] 加载模型: {model_file}")
    predictor = CharSegmentPredictor(model_file)
    predictor.threshold = threshold
    predictor.max_gap = max_gap

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        click.echo(f"[ERROR] 无法读取图像: {image_path}", err=True)
        sys.exit(1)

    intervals, pred_prob, scale = predictor.predict(img)

    if output:
        output_path = Path(output)
        predictor.draw_boundaries_with_prob(img, intervals, pred_prob, scale, output_path)
        click.echo(f"[INFO] 可视化结果已保存: {output_path}")

    click.echo(f"[INFO] 识别到 {len(intervals)} 个字符")
    if len(pred_prob) > 0:
        click.echo(f"[INFO] 概率统计:")
        click.echo(f"  平均概率: {np.mean(pred_prob):.4f}")
        click.echo(f"  最大概率: {np.max(pred_prob):.4f}")


@predict.command("compare")
@click.argument("line_id", type=str)
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录")
@click.option("--image-path", type=click.Path(exists=True), default=None,
              help="直接指定行图像路径（替代 data_base_path + line_id）")
@click.option("--rule-json-path", type=click.Path(exists=True), default=None,
              help="直接指定规则 JSON 路径")
@click.option("--model-path", type=click.Path(exists=True), default=None,
              help="模型权重路径")
@click.option("--save-dir", type=str, default=None,
              help="可视化结果保存目录")
@click.option("--max-gap", type=int, default=2, show_default=True,
              help="模型合并间隙（特征像素，设为 -1 禁用合并）")
@click.option("--merge-min-gap", type=int, default=3, show_default=True,
              help="规则后处理合并：最小间隙")
@click.option("--merge-single-ratio", type=float, default=0.7, show_default=True,
              help="规则后处理合并：单字符宽高比上限")
@click.option("--merge-min-ratio", type=float, default=0.5, show_default=True,
              help="规则后处理合并：合并后宽高比下限")
@click.option("--merge-max-ratio", type=float, default=1.5, show_default=True,
              help="规则后处理合并：合并后宽高比上限")
@click.option("--threshold", type=float, default=0.3, show_default=True,
              help="模型预测概率阈值")
def predict_compare(line_id, data_base_path, image_path, rule_json_path,
                    model_path, save_dir, max_gap, merge_min_gap,
                    merge_single_ratio, merge_min_ratio, merge_max_ratio,
                    threshold):
    """对比规则与模型的切割结果（生成五行可视化图）"""
    # 直接委托给 visualize_comparison 的 CLI
    from ai_model.inference.visualize_comparison import cli as viz_cli
    import sys

    # 构造 argv 参数
    argv = ["compare", line_id]
    if data_base_path:
        argv += ["--data-base-path", str(data_base_path)]
    if image_path:
        argv += ["--image-path", str(image_path)]
    if rule_json_path:
        argv += ["--rule-json-path", str(rule_json_path)]
    if model_path:
        argv += ["--model-path", str(model_path)]
    if save_dir:
        argv += ["--save-dir", str(save_dir)]
    argv += ["--max-gap", str(max_gap)]
    if merge_min_gap != 3:
        argv += ["--merge-min-gap", str(merge_min_gap)]
    if abs(merge_single_ratio - 0.7) > 1e-6:
        argv += ["--merge-single-ratio", str(merge_single_ratio)]
    if abs(merge_min_ratio - 0.5) > 1e-6:
        argv += ["--merge-min-ratio", str(merge_min_ratio)]
    if abs(merge_max_ratio - 1.5) > 1e-6:
        argv += ["--merge-max-ratio", str(merge_max_ratio)]
    if abs(threshold - 0.3) > 1e-6:
        argv += ["--threshold", str(threshold)]

    sys.argv = argv
    viz_cli()


# ============================================================
# 顶层命令组
# ============================================================
class AliasedGroup(click.Group):
    """支持短命令别名"""


@click.group(cls=AliasedGroup)
def cli():
    """字符切割全流程工具

    支持 PDF 分割、模型训练、推理对比等完整工作流。
    使用子命令查看各模块帮助：python cli.py <模块> --help
    """


# 注册子命令组
cli.add_command(segment)
cli.add_command(train)
cli.add_command(predict)


if __name__ == "__main__":
    cli()
