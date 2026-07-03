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

from ai_model.models.unet1d import UNet1D, DiceBCELoss
from ai_model.data.dataset import CharSegmentDataset, collate_fn, load_all_line_ids
from ai_model.train.train_config import TrainConfig
from ai_model.train.train_common import train_model, evaluate_model, save_model_and_history, setup_scheduler


def load_dataset_split(config: TrainConfig):
    base_dir = Path(__file__).resolve().parent.parent.parent
    split_path = base_dir / config.split_file
    
    if split_path.exists():
        from ai_model.data.generate_dataset_split import load_dataset_split as load_split
        split_info = load_split(split_path)
        return split_info['train_ids'], split_info['val_ids'], split_info.get('char_width_stats', None)
    return None, None, None


def main(config: TrainConfig = None):
    if config is None:
        config = TrainConfig()
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_base_path = base_dir / config.data_base_path
    split_path = base_dir / config.split_file
    
    print(f"[INFO] === 预训练模式 ===")
    print(f"[INFO] 数据基础目录: {data_base_path}")
    print(f"[INFO] 训练配置:")
    print(f"  - 批大小: {config.batch_size}")
    print(f"  - 学习率: {config.learning_rate}")
    print(f"  - 训练轮数: {config.num_epochs}")
    print(f"  - 训练/验证比例: {config.train_ratio}/{1-config.train_ratio}")
    print(f"  - 设备: {config.device}")
    print(f"  - AMP: {config.use_amp}")
    print(f"  - DataLoader workers: {config.num_workers}")
    print(f"  - 数据集划分文件: {split_path}")
    
    train_ids, val_ids, char_width_stats = load_dataset_split(config)
    
    if train_ids is None:
        print("[INFO] 加载行ID列表...")
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
    
    train_dataset = CharSegmentDataset(data_base_path, train_ids, char_width_stats=char_width_stats)
    val_dataset = CharSegmentDataset(data_base_path, val_ids, char_width_stats=char_width_stats)
    
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
    
    model = UNet1D(n_channels=6, n_classes=1).to(device)
    
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = setup_scheduler(
        optimizer,
        config.lr_scheduler_type,
        config.lr_scheduler_min_lr,
        config.lr_scheduler_factor,
        config.lr_scheduler_patience
    )
    
    print(f"[INFO] 学习率调度器: {config.lr_scheduler_type}")
    
    print("[INFO] 开始预训练...")
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
        base_lr=config.learning_rate,
        use_amp=config.use_amp,
        log_interval=config.log_interval,
        checkpoint_dir=config.checkpoint_dir,
        model_name=config.model_name,
        lr_scheduler_type=config.lr_scheduler_type,
        global_char_width=global_char_width,
        mode="预训练"
    )
    
    print("[INFO] 预训练完成")
    
    save_model_and_history(model, history, config.checkpoint_dir, config.model_name)
    
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
    
    @click.command("pretrain")
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
    def cli(batch_size, learning_rate, num_epochs, train_ratio,
            device, num_workers, use_amp, checkpoint_dir,
            data_base_path, split_file, seed):
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
        main(cfg)
    
    cli()
