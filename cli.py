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


@segment.command("reprocess")
@click.option("--config", type=str, default=None,
              help="后处理链配置名（如 merge_fragments_gap3），留空使用默认配置")
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录（默认: <项目根>/datahome）")
def segment_reprocess(config, data_base_path):
    """对已有 rule_jsons 重新应用后处理链（不重新切割图像）"""
    from image_tools.segment_manager import SegmentManager
    from image_tools.pdf_config import Pdf2ImageConfig
    from image_tools.image_config import Image2LineConfig
    from image_tools.segment_config import Line2CharConfig

    data_path = Path(data_base_path) if data_base_path else BASE_DIR / "datahome"
    char_cfg = Line2CharConfig()
    mgr = SegmentManager(
        pdf_cfg=Pdf2ImageConfig(),
        img_cfg=Image2LineConfig(),
        char_cfg=char_cfg,
        data_base_path=data_path
    )
    stats = mgr.reprocess_rule_jsons(config_name=config)
    click.echo(f"\n处理: {stats['processed']}/{stats['total']}, "
               f"变更: {stats['changed']}, 失败: {stats['failed']}")


# ============================================================
# 命令组：train - 模型训练相关
# ============================================================
@click.group()
def train():
    """训练深度学习分割模型"""


@train.command("pretrain")
@click.option("--batch-size", type=int, default=8, show_default=True, help="批大小")
@click.option("--lr", "learning_rate", type=float, default=1e-4, show_default=True, help="学习率")
@click.option("--epochs", "num_epochs", type=int, default=50, show_default=True, help="训练轮数")
@click.option("--train-ratio", type=float, default=0.8, show_default=True, help="训练集比例")
@click.option("--device", type=str, default="auto", show_default=True,
              help='训练设备: "auto", "cuda", "cpu"')
@click.option("--num-workers", type=int, default=4, show_default=True, help="DataLoader 并行数")
@click.option("--use-amp/--no-amp", default=True, help="是否启用混合精度训练")
@click.option("--checkpoint-dir", type=str, default="models", show_default=True, help="模型保存目录")
@click.option("--model-name", type=str, default=None,
              help="模型名称前缀（默认: char_segment_1d_unet）。产物为 {name}_best.pth / {name}_final.pth")
@click.option("--data-base-path", type=str, default="datahome", show_default=True,
              help="数据基础目录（相对项目根）")
@click.option("--split-file", type=str, default="ai_model/data/dataset_split.json",
              show_default=True, help="数据集划分文件路径（相对项目根）")
@click.option("--seed", type=int, default=42, show_default=True, help="随机种子")
def train_pretrain(batch_size, learning_rate, num_epochs, train_ratio,
                   device, num_workers, use_amp, checkpoint_dir, model_name,
                   data_base_path, split_file, seed):
    """预训练分割模型（使用 rule_jsons 数据）"""
    from ai_model.train.train_config import TrainConfig
    from ai_model.train.pretrain import main as pretrain_main

    cfg = TrainConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        train_ratio=train_ratio,
        device=device,
        num_workers=num_workers,
        use_amp=use_amp,
        checkpoint_dir=checkpoint_dir,
        **({"model_name": model_name} if model_name else {}),
        data_base_path=data_base_path,
        split_file=split_file,
        seed=seed
    )
    click.echo(f"[INFO] 输出模型名: {cfg.model_name} → "
               f"{cfg.model_name}_best.pth / {cfg.model_name}_final.pth")
    pretrain_main(cfg)


@train.command("finetune")
@click.option("--batch-size", type=int, default=8, show_default=True, help="批大小")
@click.option("--epochs", "num_epochs", type=int, default=30, show_default=True, help="训练轮数")
@click.option("--train-ratio", type=float, default=0.8, show_default=True, help="训练集比例")
@click.option("--device", type=str, default="auto", show_default=True,
              help='训练设备: "auto", "cuda", "cpu"')
@click.option("--num-workers", type=int, default=4, show_default=True, help="DataLoader 并行数")
@click.option("--use-amp/--no-amp", default=True, help="是否启用混合精度训练")
@click.option("--checkpoint-dir", type=str, default="models", show_default=True, help="模型保存目录")
@click.option("--model-name", type=str, default=None,
              help="模型名称前缀（默认: char_segment_1d_unet_finetune）。产物为 {name}_best.pth / {name}_final.pth")
@click.option("--data-base-path", type=str, default="datahome", show_default=True,
              help="数据基础目录（相对项目根）")
@click.option("--dataset", type=str, default=None,
              help="合并标注文件路径（如 datahome/datasets/merged_annotations.json）")
@click.option("--split-file", type=str, default="ai_model/data/dataset_split.json",
              show_default=True, help="数据集划分文件路径（相对项目根）")
@click.option("--seed", type=int, default=42, show_default=True, help="随机种子")
@click.option("--pretrained-model", type=str, default=None,
              help="预训练模型路径，用于微调（如 models/char_segment_1d_unet_best.pth）")
@click.option("--freeze-layers/--no-freeze-layers", default=False,
              help="是否冻结编码器层，只训练解码器（微调时使用）")
@click.option("--fine-tune-lr", type=float, default=1e-5, show_default=True,
              help="微调时使用的学习率")
@click.option("--no-validation/--with-validation", default=False,
              help="是否使用全部数据训练（无验证集）")
def train_finetune(batch_size, num_epochs, train_ratio,
                   device, num_workers, use_amp, checkpoint_dir, model_name,
                   data_base_path, dataset, split_file, seed,
                   pretrained_model, freeze_layers, fine_tune_lr, no_validation):
    """微调分割模型（使用合并标注数据）"""
    from ai_model.train.train_config import FineTuneConfig
    from ai_model.train.finetune import main as finetune_main

    cfg = FineTuneConfig(
        batch_size=batch_size,
        num_epochs=num_epochs,
        train_ratio=train_ratio,
        device=device,
        num_workers=num_workers,
        use_amp=use_amp,
        checkpoint_dir=checkpoint_dir,
        **({"model_name": model_name} if model_name else {}),
        data_base_path=data_base_path,
        annotations_file=dataset,
        split_file=split_file,
        seed=seed,
        pretrained_model_path=pretrained_model,
        freeze_layers=freeze_layers,
        fine_tune_lr=fine_tune_lr,
        no_validation=no_validation
    )
    click.echo(f"[INFO] 输出模型名: {cfg.model_name} → "
               f"{cfg.model_name}_best.pth / {cfg.model_name}_final.pth")
    finetune_main(cfg)


@train.command("split-dataset")
@click.option("--train-ratio", type=float, default=0.8, show_default=True, help="训练集比例")
@click.option("--seed", type=int, default=42, show_default=True, help="随机种子")
@click.option("--sort-by-width", is_flag=True, help="按宽度排序后划分")
@click.option("--data-base-path", type=str, default="datahome", show_default=True,
              help="数据基础目录（相对项目根）")
@click.option("--dataset", type=str, default=None,
              help="合并标注文件路径（如 datahome/datasets/merged_annotations.json）")
@click.option("--output", type=str, default="ai_model/data/dataset_split.json",
              show_default=True, help="划分文件输出路径（相对项目根）")
def train_split_dataset(train_ratio, seed, sort_by_width, data_base_path, dataset, output):
    """生成数据集划分文件"""
    from ai_model.data.generate_dataset_split import generate_dataset_split
    import json

    data_path = BASE_DIR / data_base_path
    output_path = BASE_DIR / output

    annotations = None
    if dataset:
        annotations_path = BASE_DIR / dataset
        click.echo(f"[INFO] 加载合并标注文件: {annotations_path}")
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations_list = json.load(f)
        annotations = {item['line_id']: item for item in annotations_list}
        click.echo(f"[INFO] 合并标注文件包含 {len(annotations)} 条记录")

    split_info = generate_dataset_split(
        data_base_path=data_path,
        output_path=output_path,
        train_ratio=train_ratio,
        seed=seed,
        sort_by_width=sort_by_width,
        annotations=annotations
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
@click.option("--chunk-size", type=int, default=1000, show_default=True,
              help="微批量大小，每处理完一个chunk更新一次排名")
def train_active_learn(top_n, model_path, data_base_path, batch_size, num_workers, chunk_size):
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
    click.echo(f"[INFO] 开始计算主动学习分数（batch_size={batch_size}, num_workers={num_workers}, chunk_size={chunk_size}）...")
    ranked_lines = learner.rank_lines_batched(top_n, batch_size=batch_size, num_workers=num_workers, chunk_size=chunk_size)

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


@train.command("rule-based-al")
@click.argument("top_n", type=int, default=100)
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录（默认: <项目根>/datahome）")
@click.option("--chunk-size", type=int, default=1000, show_default=True,
              help="微批量大小，每处理完一个chunk更新一次排名")
def train_rule_based_al(top_n, data_base_path, chunk_size):
    """基于规则的主动学习：仅使用规则切割结果识别可能切割错误的样本"""
    from ai_model.train.active_learning import RuleBasedActiveLearner

    data_path = Path(data_base_path) if data_base_path else BASE_DIR / "datahome"
    model_dir = BASE_DIR / "ai_model" / "models"

    click.echo(f"[INFO] 数据目录: {data_path}")
    learner = RuleBasedActiveLearner(data_path)
    click.echo(f"[INFO] 开始基于规则的主动学习分析（chunk_size={chunk_size}）...")
    ranked_lines = learner.rank_lines(top_n, chunk_size=chunk_size)

    click.echo(f"\n[INFO] Top {len(ranked_lines)} 最可能切割错误的行：")
    click.echo("-" * 130)
    click.echo(f"{'排名':<4} {'行ID':<40} {'AL分数':<10} {'粘连不可分':<12} {'合并不足':<10}")
    click.echo("-" * 130)

    for idx, item in enumerate(ranked_lines, 1):
        click.echo(f"{idx:<4} {item['line_id']:<40} {item['al_score']:<10.4f} "
                   f"{item['stuck_unsplittable']:<12.4f} {item['under_merged']:<10.4f}")

    output_path = model_dir / "rule_based_al_ranking.json"
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
@click.option("--threshold", type=float, default=0.3, show_default=True,
              help="模型预测概率阈值")
def predict_compare(line_id, data_base_path, image_path, rule_json_path,
                    model_path, save_dir, max_gap, threshold):
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
    if abs(threshold - 0.3) > 1e-6:
        argv += ["--threshold", str(threshold)]

    sys.argv = argv
    viz_cli()


# ============================================================
# 命令组：project - 项目管理相关
# ============================================================
@click.group()
def project():
    """项目管理：批量推理、缓存生成等"""


@project.command("infer")
@click.argument("project_id", type=str)
@click.option("--model-path", type=click.Path(exists=True), default=None,
              help="模型权重路径（默认: models/char_segment_1d_unet_best.pth）")
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录（默认: <项目根>/datahome）")
@click.option("--batch-size", type=int, default=32, show_default=True,
              help="GPU批量推理批大小")
@click.option("--threshold", type=float, default=0.5, show_default=True,
              help="模型预测概率阈值")
def project_infer(project_id, model_path, data_base_path, batch_size, threshold):
    """批量使用模型对项目中的图片进行推理
    
    从项目的 line_id_list.json 获取 line_id，使用指定模型进行推理，
    结果保存到项目目录的 model_jsons 文件夹中。
    """
    import json
    import cv2
    import numpy as np
    from ai_model.inference.infer import CharSegmentPredictor
    
    data_path = Path(data_base_path) if data_base_path else BASE_DIR / "datahome"
    project_root = data_path / "project" / project_id
    
    line_id_list_path = project_root / "line_id_list.json"
    if not line_id_list_path.exists():
        click.echo(f"[ERROR] line_id_list.json 不存在: {line_id_list_path}", err=True)
        sys.exit(1)
    
    with open(line_id_list_path, 'r', encoding='utf-8') as f:
        line_id_data = json.load(f)
    
    line_ids = line_id_data.get('line_ids', [])
    click.echo(f"[INFO] 项目 {project_id} 包含 {len(line_ids)} 条数据")
    
    model_dir = BASE_DIR / "ai_model" / "models"
    if model_path:
        model_file = Path(model_path)
    else:
        model_file = model_dir / "char_segment_1d_unet_best.pth"
    
    if not model_file.exists():
        click.echo(f"[ERROR] 模型文件不存在: {model_file}", err=True)
        sys.exit(1)
    
    global_char_width = 0.0
    annotations_dir = project_root / "annotations"
    if annotations_dir.exists():
        all_widths = []
        for f in annotations_dir.glob('*.json'):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    if 'chars' in data:
                        for char in data['chars']:
                            w = char.get('width', 0)
                            if w >= 3 and w <= 100:
                                all_widths.append(w)
            except Exception:
                pass
        
        if len(all_widths) > 0:
            global_char_width = float(np.median(all_widths))
            click.echo(f"[INFO] 从标注数据计算全局中位字符宽度: {global_char_width:.2f} px")
        else:
            click.echo(f"[WARN] 标注目录为空，无法计算全局字符宽度")
    
    if global_char_width <= 0:
        rule_jsons_dir = data_path / "rule_jsons"
        if rule_jsons_dir.exists():
            all_widths = []
            for line_id in line_ids[:100]:
                f = rule_jsons_dir / f"{line_id}_rule.json"
                if f.exists():
                    try:
                        with open(f, 'r', encoding='utf-8') as fp:
                            data = json.load(fp)
                            if 'chars' in data:
                                for char in data['chars']:
                                    w = char.get('width', 0)
                                    if w >= 3 and w <= 100:
                                        all_widths.append(w)
                    except Exception:
                        pass
            
            if len(all_widths) > 0:
                global_char_width = float(np.median(all_widths))
                click.echo(f"[INFO] 从规则切割数据计算全局中位字符宽度: {global_char_width:.2f} px")
    
    click.echo(f"[INFO] 加载模型: {model_file}")
    predictor = CharSegmentPredictor(model_file, global_char_width=global_char_width)
    
    model_jsons_dir = project_root / "model_jsons"
    model_jsons_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    click.echo(f"[INFO] 开始批量推理（batch_size={batch_size}）...")
    
    for i, line_id in enumerate(line_ids):
        line_img_path = data_path / "lines" / f"{line_id}.png"
        
        if not line_img_path.exists():
            click.echo(f"[WARN] 图像不存在: {line_img_path}")
            fail_count += 1
            continue
        
        img = cv2.imread(str(line_img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            click.echo(f"[WARN] 无法读取图像: {line_img_path}")
            fail_count += 1
            continue
        
        result = predictor.predict(img)
        if result is None:
            fail_count += 1
            continue
        
        intervals, pred_prob, pred_logits, scale = result
        
        chars = []
        for idx, (start, end) in enumerate(intervals):
            chars.append({
                "char_id": f"{line_id}_char_{idx}",
                "line_id": line_id,
                "char_idx": idx,
                "col_start": start,
                "col_end": end,
                "width": end - start
            })
        
        output_data = {
            "line_id": line_id,
            "chars": chars,
            "char_count": len(chars),
            "width": img.shape[1],
            "height": img.shape[0],
            "threshold": threshold,
            "model_path": str(model_file.name)
        }
        
        output_path = model_jsons_dir / f"{line_id}_model.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        success_count += 1
        
        if (i + 1) % batch_size == 0:
            click.echo(f"[INFO] 已处理 {i + 1}/{len(line_ids)} ({success_count}成功, {fail_count}失败)")
    
    click.echo(f"\n[INFO] 批量推理完成!")
    click.echo(f"[INFO] 成功: {success_count}")
    click.echo(f"[INFO] 失败: {fail_count}")
    click.echo(f"[INFO] 输出目录: {model_jsons_dir}")


@project.command("infer-all")
@click.option("--model-path", type=click.Path(exists=True), default=None,
              help="模型权重路径（默认: models/char_segment_1d_unet_best.pth）")
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录（默认: <项目根>/datahome）")
@click.option("--output-version", type=str, default=None,
              help="输出版本目录名（默认: 自动生成 v{timestamp}）")
@click.option("--threshold", type=float, default=0.5, show_default=True,
              help="模型预测概率阈值")
@click.option("--save-detail/--no-save-detail", default=False, show_default=True,
              help="是否保存详细结果（完整JSON，包含所有字段）")
@click.option("--report-interval", type=int, default=500, show_default=True,
              help="进度报告间隔")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True,
              help="是否覆盖已有结果（否则跳过已完成的line_id）")
@click.option("--limit", type=int, default=None,
              help="限制处理数量（用于测试，默认处理全部）")
def project_infer_all(model_path, data_base_path, output_version, threshold, 
                      save_detail, report_interval, overwrite, limit):
    """全量推理：对所有规则分割数据进行模型推理，使用分层存储
    
    存储结构：
        datahome/model_inference/{version}/
            metadata.json      - 推理元数据
            inference.jsonl    - 轻量结果（字符区间）
            detail/            - 详细结果（按PDF分区）
    
    轻量JSONL格式：
        {"line_id":"xxx","char_count":N,"chars":[[start,end],...]}
    
    详细结果格式与 project infer 命令相同。
    
    支持断点续传：如果jsonl文件已存在且不启用overwrite，会跳过已完成的line_id。
    """
    import json
    import cv2
    import numpy as np
    from datetime import datetime
    from ai_model.inference.infer import CharSegmentPredictor
    from ai_model.inference.inference_storage import (
        write_inference_jsonl, write_detail_json, write_metadata,
        read_inference_jsonl, extract_pdf_id
    )
    
    data_path = Path(data_base_path) if data_base_path else BASE_DIR / "datahome"
    model_dir = BASE_DIR / "ai_model" / "models"
    
    if model_path:
        model_file = Path(model_path)
    else:
        model_file = model_dir / "char_segment_1d_unet_best.pth"
    
    if not model_file.exists():
        click.echo(f"[ERROR] 模型文件不存在: {model_file}", err=True)
        sys.exit(1)
    
    if output_version is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_version = f"v{timestamp}"
    
    inference_base = data_path / "model_inference" / output_version
    jsonl_path = inference_base / "inference.jsonl"
    detail_dir = inference_base / "detail"
    metadata_path = inference_base / "metadata.json"
    
    inference_base.mkdir(parents=True, exist_ok=True)
    
    existing_line_ids = set()
    if jsonl_path.exists() and not overwrite:
        existing_results = read_inference_jsonl(jsonl_path)
        existing_line_ids = set(existing_results.keys())
        click.echo(f"[INFO] 检测到已有结果，将跳过 {len(existing_line_ids)} 个已完成的line_id")
    
    click.echo(f"[INFO] 推理版本: {output_version}")
    click.echo(f"[INFO] 输出目录: {inference_base}")
    click.echo(f"[INFO] 轻量结果: {jsonl_path}")
    if save_detail:
        click.echo(f"[INFO] 详细结果: {detail_dir}")
    
    rule_jsons_dir = data_path / "rule_jsons"
    if not rule_jsons_dir.exists():
        click.echo(f"[ERROR] 规则分割目录不存在: {rule_jsons_dir}", err=True)
        sys.exit(1)
    
    rule_files = list(rule_jsons_dir.glob('*_rule.json'))
    if len(rule_files) == 0:
        click.echo(f"[ERROR] 规则分割目录为空", err=True)
        sys.exit(1)
    
    line_ids = []
    for f in rule_files:
        stem = f.stem.replace('_rule', '')
        if 'line_page_' in stem and '_pdf_' not in stem:
            stem = stem.replace('line_page_', 'line_page_pdf_')
        line_ids.append(stem)
    if limit is not None:
        line_ids = line_ids[:limit]
        click.echo(f"[INFO] 测试模式：限制处理 {limit} 条数据")
    
    click.echo(f"[INFO] 总数据量: {len(line_ids)}")
    
    global_char_width = 0.0
    all_widths = []
    for f in rule_files[:100]:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                if 'chars' in data:
                    for char in data['chars']:
                        w = char.get('width', 0)
                        if w >= 3 and w <= 100:
                            all_widths.append(w)
        except Exception:
            pass
    
    if len(all_widths) > 0:
        global_char_width = float(np.median(all_widths))
        click.echo(f"[INFO] 从规则切割数据计算全局中位字符宽度: {global_char_width:.2f} px")
    
    click.echo(f"[INFO] 加载模型: {model_file}")
    predictor = CharSegmentPredictor(model_file, global_char_width=global_char_width)
    
    success_count = len(existing_line_ids)
    fail_count = 0
    
    jsonl_buffer = []
    detail_buffer = []
    
    click.echo(f"[INFO] 开始推理...")
    
    for i, line_id in enumerate(line_ids):
        if line_id in existing_line_ids:
            continue
        
        line_img_path = data_path / "lines" / f"{line_id}.png"
        
        if not line_img_path.exists():
            click.echo(f"[WARN] 图像不存在: {line_img_path}")
            fail_count += 1
            continue
        
        img = cv2.imread(str(line_img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            click.echo(f"[WARN] 无法读取图像: {line_img_path}")
            fail_count += 1
            continue
        
        result = predictor.predict(img)
        if result is None:
            fail_count += 1
            continue
        
        intervals, pred_prob, pred_logits, scale = result
        
        jsonl_buffer.append({
            "line_id": line_id,
            "char_count": len(intervals),
            "chars": [[int(start), int(end)] for start, end in intervals]
        })
        
        if save_detail:
            chars = []
            for idx, (start, end) in enumerate(intervals):
                chars.append({
                    "char_id": f"{line_id}_char_{idx}",
                    "line_id": line_id,
                    "char_idx": idx,
                    "col_start": start,
                    "col_end": end,
                    "width": end - start
                })
            
            output_data = {
                "line_id": line_id,
                "chars": chars,
                "char_count": len(chars),
                "width": img.shape[1],
                "height": img.shape[0],
                "threshold": threshold,
                "model_path": str(model_file.name)
            }
            detail_buffer.append((line_id, output_data))
        
        success_count += 1
        
        if (i + 1) % report_interval == 0:
            with open(jsonl_path, 'a', encoding='utf-8') as f:
                for record in jsonl_buffer:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            jsonl_buffer = []
            
            if save_detail:
                for line_id_detail, output_data_detail in detail_buffer:
                    write_detail_json(detail_dir, line_id_detail, output_data_detail)
                detail_buffer = []
            
            click.echo(f"[INFO] 已处理 {i + 1}/{len(line_ids)} ({success_count}成功, {fail_count}失败)")
    
    if jsonl_buffer:
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            for record in jsonl_buffer:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    if save_detail and detail_buffer:
        for line_id_detail, output_data_detail in detail_buffer:
            write_detail_json(detail_dir, line_id_detail, output_data_detail)
    
    params = {
        "threshold": threshold,
        "global_char_width": global_char_width,
        "save_detail": save_detail,
        "overwrite": overwrite,
        "limit": limit
    }
    write_metadata(metadata_path, str(model_file.name), success_count, fail_count, params)
    
    click.echo(f"\n[INFO] 推理完成!")
    click.echo(f"[INFO] 版本: {output_version}")
    click.echo(f"[INFO] 成功: {success_count}")
    click.echo(f"[INFO] 失败: {fail_count}")
    click.echo(f"[INFO] 元数据: {metadata_path}")
    click.echo(f"[INFO] 轻量结果: {jsonl_path}")
    if save_detail:
        click.echo(f"[INFO] 详细结果: {detail_dir}")


@project.command("cache-lineage")
@click.argument("project_id", type=str)
@click.option("--data-base-path", type=click.Path(exists=True), default=None,
              help="数据基础目录（默认: <项目根>/datahome）")
def project_cache_lineage(project_id, data_base_path):
    """生成项目的 lineage_cache.json 缓存
    
    基于项目的 line_id_list.json，从全局 lineage.json 中提取相关数据，
    生成项目级的 lineage_cache.json 文件。
    """
    import json
    
    data_path = Path(data_base_path) if data_base_path else BASE_DIR / "datahome"
    project_root = data_path / "project" / project_id
    
    line_id_list_path = project_root / "line_id_list.json"
    if not line_id_list_path.exists():
        click.echo(f"[ERROR] line_id_list.json 不存在: {line_id_list_path}", err=True)
        sys.exit(1)
    
    with open(line_id_list_path, 'r', encoding='utf-8') as f:
        line_id_data = json.load(f)
    
    line_ids = line_id_data.get('line_ids', [])
    click.echo(f"[INFO] 项目 {project_id} 包含 {len(line_ids)} 条数据")
    
    lineage_path = data_path / "lineage.json"
    if not lineage_path.exists():
        click.echo(f"[ERROR] 全局 lineage.json 不存在: {lineage_path}", err=True)
        sys.exit(1)
    
    click.echo(f"[INFO] 加载全局 lineage.json（可能需要数秒）...")
    with open(lineage_path, 'r', encoding='utf-8') as f:
        global_lineage = json.load(f)
    
    global_lines = global_lineage.get('lines', {})
    global_chars = global_lineage.get('chars', {})
    
    click.echo(f"[INFO] 全局 lineage.json 包含 {len(global_lines)} 行数据")
    
    project_lines = {}
    project_chars = {}
    
    for line_id in line_ids:
        if line_id in global_lines:
            project_lines[line_id] = global_lines[line_id]
            
            char_ids = global_lines[line_id].get('chars', [])
            for char_id in char_ids:
                if char_id in global_chars:
                    project_chars[char_id] = global_chars[char_id]
    
    cache_data = {
        "lines": project_lines,
        "chars": project_chars
    }
    
    cache_path = project_root / "lineage_cache.json"
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False)
    
    click.echo(f"[INFO] lineage_cache.json 生成完成!")
    click.echo(f"[INFO] 行数: {len(project_lines)}")
    click.echo(f"[INFO] 字符数: {len(project_chars)}")
    click.echo(f"[INFO] 输出文件: {cache_path}")


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
cli.add_command(project)


if __name__ == "__main__":
    cli()
