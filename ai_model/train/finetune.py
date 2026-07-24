import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from ai_model.models.unet1d import UNet1D
from ai_model.data.dataset import CharSegmentDataset, collate_fn, load_all_line_ids
from ai_model.train.train_config import FineTuneConfig
from ai_model.train.train_common import train_model, evaluate_model, save_model_and_history, setup_scheduler, FocalLoss


def load_annotations(config: FineTuneConfig):
    if not config.annotations_file:
        return None
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    annotations_path = base_dir / config.annotations_file
    
    print(f"[INFO] 加载合并标注文件: {annotations_path}")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations_list = json.load(f)
    annotations = {item['line_id']: item for item in annotations_list}
    print(f"[INFO] 合并标注文件包含 {len(annotations)} 条记录")
    
    return annotations


def load_dataset_split(config: FineTuneConfig):
    base_dir = Path(__file__).resolve().parent.parent.parent
    split_path = base_dir / config.split_file
    
    if split_path.exists():
        from ai_model.data.generate_dataset_split import load_dataset_split as load_split
        split_info = load_split(split_path)
        return split_info['train_ids'], split_info['val_ids'], split_info.get('char_width_stats', None)
    return None, None, None


def load_pretrained_model(model: nn.Module, config: FineTuneConfig, device: torch.device):
    if not config.pretrained_model_path:
        return
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    model_path = base_dir / config.pretrained_model_path
    
    if model_path.exists():
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
        
        model_state_dict = model.state_dict()
        checkpoint_keys = set(checkpoint.keys())
        model_keys = set(model_state_dict.keys())
        
        matching_keys = checkpoint_keys & model_keys
        mismatched_keys = checkpoint_keys - model_keys
        missing_keys = model_keys - checkpoint_keys
        
        if mismatched_keys:
            print(f"[WARNING] 预训练模型中有但当前模型中没有的参数（可能是输出层差异）: {mismatched_keys}")
        if missing_keys:
            print(f"[WARNING] 当前模型中有但预训练模型中没有的参数: {missing_keys}")
        
        filtered_checkpoint = {}
        for key in checkpoint:
            if key in model_state_dict and checkpoint[key].shape == model_state_dict[key].shape:
                filtered_checkpoint[key] = checkpoint[key]
        
        model.load_state_dict(filtered_checkpoint, strict=False)
        
        print(f"[INFO] 加载预训练模型: {model_path}")
        print(f"[INFO] 成功加载 {len(filtered_checkpoint)} 个参数")
        if len(filtered_checkpoint) < len(checkpoint):
            print(f"[INFO] 跳过 {len(checkpoint) - len(filtered_checkpoint)} 个不匹配的参数（输出层）")
    else:
        print(f"[WARNING] 预训练模型路径不存在: {model_path}")
        print("[INFO] 将从头开始训练")


def freeze_encoder(model: nn.Module):
    print("[INFO] 冻结编码器层，只训练解码器和输出层...")
    for name, param in model.named_parameters():
        if 'decoder' not in name.lower() and 'outc' not in name.lower():
            param.requires_grad = False


def main(config: FineTuneConfig = None):
    if config is None:
        config = FineTuneConfig()
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_base_path = base_dir / config.data_base_path
    split_path = base_dir / config.split_file
    
    annotations = load_annotations(config)
    
    print(f"[INFO] === 微调模式 ===")
    print(f"[INFO] 数据基础目录: {data_base_path}")
    print(f"[INFO] 训练配置:")
    print(f"  - 批大小: {config.batch_size}")
    print(f"  - 微调学习率: {config.fine_tune_lr}")
    print(f"  - 训练轮数: {config.num_epochs}")
    print(f"  - 训练/验证比例: {config.train_ratio}/{1-config.train_ratio}")
    print(f"  - 设备: {config.device}")
    print(f"  - AMP: {config.use_amp}")
    print(f"  - DataLoader workers: {config.num_workers}")
    print(f"  - 数据集划分文件: {split_path}")
    print(f"  - 预训练模型: {config.pretrained_model_path}")
    print(f"  - 冻结编码器: {config.freeze_layers}")
    
    train_ids, val_ids, char_width_stats = load_dataset_split(config)
    
    if train_ids is None:
        print("[INFO] 加载行ID列表...")
        if annotations is not None:
            line_ids = list(annotations.keys())
        else:
            line_ids = load_all_line_ids(data_base_path)
        print(f"[INFO] 找到 {len(line_ids)} 个行图像")
        
        if len(line_ids) == 0:
            print("[ERROR] 未找到训练数据")
            return
        
        np.random.seed(config.seed)
        np.random.shuffle(line_ids)
        
        split_idx = int(len(line_ids) * config.train_ratio)
        train_ids = line_ids[:split_idx]
        val_ids = line_ids[split_idx:]
        
        print(f"[WARNING] 划分文件不存在，动态生成划分")
        print(f"[INFO] 使用种子: {config.seed}")
    else:
        print(f"[INFO] 从划分文件加载预计算的字符宽度统计")
    
    print(f"[INFO] 训练集: {len(train_ids)} 样本")
    print(f"[INFO] 验证集: {len(val_ids)} 样本")
    
    train_dataset = CharSegmentDataset(data_base_path, train_ids, char_width_stats=char_width_stats, annotations=annotations)
    val_dataset = CharSegmentDataset(data_base_path, val_ids, char_width_stats=char_width_stats, annotations=annotations)
    
    global_char_width = train_dataset.global_char_width
    print(f"[INFO] 全局中位字符宽度: {global_char_width:.2f} px")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0
    )
    
    device = torch.device(config.device)
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    print(f"[INFO] 使用设备: {device}")
    
    model = UNet1D(n_channels=6, n_classes=3).to(device)
    
    load_pretrained_model(model, config, device)
    
    if config.freeze_layers:
        freeze_encoder(model)
    
    criterion = FocalLoss(gamma=2.0)
    
    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_train, lr=config.fine_tune_lr)
    scheduler = setup_scheduler(
        optimizer,
        config.lr_scheduler_type,
        config.lr_scheduler_min_lr,
        config.lr_scheduler_factor,
        config.lr_scheduler_patience
    )
    
    print(f"[INFO] 学习率调度器: {config.lr_scheduler_type}")
    print(f"[INFO] 参与训练的参数数量: {sum(p.numel() for p in params_to_train)}")
    
    print("[INFO] 开始微调...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        device_type=device_type,
        num_epochs=config.num_epochs,
        warmup_epochs=config.warmup_epochs,
        base_lr=config.fine_tune_lr,
        use_amp=config.use_amp,
        log_interval=config.log_interval,
        checkpoint_dir=config.checkpoint_dir,
        model_name=config.model_name,
        lr_scheduler_type=config.lr_scheduler_type,
        global_char_width=global_char_width,
        mode="微调"
    )
    
    print("[INFO] 微调完成")
    
    save_model_and_history(model, history, config.checkpoint_dir, config.model_name, "finetune_history.json")
    
    print("\n[INFO] 验证集最终评估:")
    model.load_state_dict(torch.load(str(Path(config.checkpoint_dir) / f"{config.model_name}_best.pth"), weights_only=True))
    results = evaluate_model(model, val_loader, criterion, device, device_type, config.use_amp, global_char_width)
    
    print(f"  损失: {results['loss']:.4f}")
    print(f"  列级准确率: {results['col_acc']:.4f}")
    print(f"  区间IoU: {results['iou']:.4f}")
    print(f"  ROI-IoU: {results['roi_iou']:.4f}")
    print(f"  Char-IoU: {results['char_iou']:.4f}")
    print(f"  Gap-IoU: {results['gap_iou']:.4f}")


if __name__ == "__main__":
    import click
    
    @click.command("finetune")
    @click.option("--batch-size", type=int, default=8, show_default=True, help="批大小")
    @click.option("--epochs", "num_epochs", type=int, default=30, show_default=True, help="训练轮数")
    @click.option("--train-ratio", type=float, default=0.8, show_default=True, help="训练集比例")
    @click.option("--device", type=str, default="auto", show_default=True,
                  help='训练设备: "auto", "cuda", "cpu"')
    @click.option("--num-workers", type=int, default=4, show_default=True, help="DataLoader 并行数")
    @click.option("--use-amp/--no-amp", default=True, help="是否启用混合精度训练")
    @click.option("--checkpoint-dir", type=str, default="models", show_default=True, help="模型保存目录")
    @click.option("--data-base-path", type=str, default="datahome", show_default=True,
                  help="数据基础目录（相对项目根）")
    @click.option("--dataset", "annotations_file", type=str, default=None,
                  help="合并标注文件路径（如 datahome/datasets/merged_annotations.json）")
    @click.option("--split-file", type=str, default="ai_model/data/dataset_split.json",
                  show_default=True, help="数据集划分文件路径（相对项目根）")
    @click.option("--seed", type=int, default=42, show_default=True, help="随机种子")
    @click.option("--pretrained-model", "pretrained_model_path", type=str, default=None,
                  help="预训练模型路径，用于微调（如 models/char_segment_1d_unet_best.pth）")
    @click.option("--freeze-layers/--no-freeze-layers", default=False,
                  help="是否冻结编码器层，只训练解码器（微调时使用）")
    @click.option("--fine-tune-lr", type=float, default=1e-5, show_default=True,
                  help="微调时使用的学习率")
    def cli(batch_size, num_epochs, train_ratio,
            device, num_workers, use_amp, checkpoint_dir,
            data_base_path, annotations_file, split_file, seed,
            pretrained_model_path, freeze_layers, fine_tune_lr):
        cfg = FineTuneConfig(
            batch_size=batch_size,
            num_epochs=num_epochs,
            train_ratio=train_ratio,
            device=device,
            num_workers=num_workers,
            use_amp=use_amp,
            checkpoint_dir=checkpoint_dir,
            data_base_path=data_base_path,
            annotations_file=annotations_file,
            split_file=split_file,
            seed=seed,
            pretrained_model_path=pretrained_model_path,
            freeze_layers=freeze_layers,
            fine_tune_lr=fine_tune_lr
        )
        main(cfg)
    
    cli()
