import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import click
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from ai_model.models.unet1d import UNet1D, DiceBCELoss, column_accuracy, interval_iou, roi_interval_iou, split_interval_iou
from ai_model.data.dataset import CharSegmentDataset, collate_fn, load_all_line_ids
from ai_model.train.train_config import TrainConfig

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config, global_char_width=16.0):
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_col_acc': [],
        'val_col_acc': [],
        'train_iou': [],
        'val_iou': [],
        'train_roi_iou': [],
        'val_roi_iou': [],
        'train_char_iou': [],
        'val_char_iou': [],
        'train_gap_iou': [],
        'val_gap_iou': [],
        'learning_rate': []
    }
    
    model_dir = Path(config.checkpoint_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # AMP GradScaler
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=config.use_amp)
    
    epoch_range = tqdm(range(config.num_epochs), desc="训练进度", unit="epoch") if HAS_TQDM else range(config.num_epochs)
    
    for epoch in epoch_range:
        model.train()
        train_loss = 0.0
        train_col_acc = 0.0
        train_iou = 0.0
        train_count = 0
        
        if epoch < config.warmup_epochs:
            warmup_factor = (epoch + 1) / config.warmup_epochs
            current_lr = config.learning_rate * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        batch_range = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", unit="batch", leave=False) if HAS_TQDM else train_loader
        
        for batch in batch_range:
            features = torch.from_numpy(batch['features']).to(device, non_blocking=True)
            labels = torch.from_numpy(batch['labels']).to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type, enabled=config.use_amp):
                outputs = model(features)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * features.size(0)
            train_col_acc += column_accuracy(outputs.detach(), labels.detach()).item() * features.size(0)
            train_iou += interval_iou(outputs.detach(), labels.detach()).item() * features.size(0)
            train_count += features.size(0)
            
            if HAS_TQDM:
                batch_range.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{current_lr:.2e}"})
        
        train_loss /= train_count
        train_col_acc /= train_count
        train_iou /= train_count
        
        history['learning_rate'].append(current_lr)
        
        model.eval()
        val_loss = 0.0
        val_col_acc = 0.0
        val_iou = 0.0
        val_roi_iou = 0.0
        val_char_iou = 0.0
        val_gap_iou = 0.0
        val_count = 0
        
        val_range = tqdm(val_loader, desc=f"验证 Epoch {epoch+1}", unit="batch", leave=False) if HAS_TQDM else val_loader
        
        with torch.no_grad():
            for batch in val_range:
                features = torch.from_numpy(batch['features']).to(device, non_blocking=True)
                labels = torch.from_numpy(batch['labels']).to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type, enabled=config.use_amp):
                    outputs = model(features)
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                val_col_acc += column_accuracy(outputs, labels).item() * features.size(0)
                val_iou += interval_iou(outputs, labels).item() * features.size(0)
                
                val_roi_iou += roi_interval_iou(outputs, labels, char_width=global_char_width).item() * features.size(0)
                split_iou = split_interval_iou(outputs, labels, char_width=global_char_width)
                val_char_iou += split_iou['char_iou'].item() * features.size(0)
                val_gap_iou += split_iou['gap_iou'].item() * features.size(0)
                
                val_count += features.size(0)
        
        val_loss /= val_count
        val_col_acc /= val_count
        val_iou /= val_count
        val_roi_iou /= val_count
        val_char_iou /= val_count
        val_gap_iou /= val_count
        
        if config.lr_scheduler_type == "cosine":
            scheduler.step(epoch)
        else:
            scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_col_acc'].append(train_col_acc)
        history['val_col_acc'].append(val_col_acc)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['val_roi_iou'].append(val_roi_iou)
        history['val_char_iou'].append(val_char_iou)
        history['val_gap_iou'].append(val_gap_iou)
        
        if (epoch + 1) % config.log_interval == 0:
            epoch_str = f"Epoch [{epoch+1}/{config.num_epochs}]"
            lr_str = f"LR={current_lr:.2e}"
            train_str = f"Train: Loss={train_loss:.4f}, ColAcc={train_col_acc:.4f}, IoU={train_iou:.4f}"
            val_str = f"Val:   Loss={val_loss:.4f}, ColAcc={val_col_acc:.4f}, IoU={val_iou:.4f}"
            val_roi_str = f"ROI-IoU={val_roi_iou:.4f}, Char-IoU={val_char_iou:.4f}, Gap-IoU={val_gap_iou:.4f}"
            
            if HAS_TQDM:
                epoch_range.write(f"\n{epoch_str} | {lr_str}")
                epoch_range.write(f"  {train_str}")
                epoch_range.write(f"  {val_str}")
                epoch_range.write(f"  {val_roi_str}")
            else:
                print(f"{epoch_str} | {lr_str}")
                print(f"  {train_str}")
                print(f"  {val_str}")
                print(f"  {val_roi_str}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = model_dir / f"{config.model_name}_best.pth"
            torch.save(model.state_dict(), str(best_path))
            
            if HAS_TQDM:
                epoch_range.write(f"  [INFO] 保存最佳模型: {best_path}")
            else:
                print(f"  [INFO] 保存最佳模型: {best_path}")
        
        if HAS_TQDM:
            epoch_range.set_postfix({
                "Val Loss": f"{val_loss:.4f}",
                "Val IoU": f"{val_iou:.4f}"
            })
    
    return history


def main(config: TrainConfig = None):
    if config is None:
        config = TrainConfig()
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_base_path = base_dir / config.data_base_path
    split_path = base_dir / config.split_file
    
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
    
    if split_path.exists():
        print("[INFO] 从划分文件加载数据集...")
        from ai_model.data.generate_dataset_split import load_dataset_split
        
        split_info = load_dataset_split(split_path)
        train_ids = split_info['train_ids']
        val_ids = split_info['val_ids']
        
        char_width_stats = split_info.get('char_width_stats', None)
        if char_width_stats:
            print(f"[INFO] 从划分文件加载预计算的字符宽度统计")
        else:
            print(f"[INFO] 划分文件无字符宽度统计，将在 Dataset 中计算")
        
        print(f"[INFO] 数据集版本: {split_info['version']}")
        print(f"[INFO] 创建时间: {split_info['created_at']}")
        print(f"[INFO] 划分种子: {split_info['config']['seed']}")
    else:
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
        
        char_width_stats = None
        
        print(f"[WARNING] 划分文件不存在，动态生成划分")
        print(f"[INFO] 使用种子: {config.seed}")
    
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
    
    if config.lr_scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=2,
            eta_min=config.lr_scheduler_min_lr
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', 
            factor=config.lr_scheduler_factor, 
            patience=config.lr_scheduler_patience, 
            min_lr=config.lr_scheduler_min_lr
        )
    
    print(f"[INFO] 学习率调度器: {config.lr_scheduler_type}")
    
    print("[INFO] 开始训练...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config, global_char_width)
    
    print("[INFO] 训练完成")
    
    model_dir = Path(config.checkpoint_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    final_path = model_dir / f"{config.model_name}_final.pth"
    torch.save(model.state_dict(), str(final_path))
    
    history_path = model_dir / "training_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    
    print(f"[INFO] 模型保存到: {model_dir}")
    
    print("\n[INFO] 验证集最终评估:")
    model.load_state_dict(torch.load(str(model_dir / f"{config.model_name}_best.pth")))
    model.eval()
    
    val_loss = 0.0
    val_col_acc = 0.0
    val_iou = 0.0
    val_roi_iou = 0.0
    val_char_iou = 0.0
    val_gap_iou = 0.0
    val_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            features = torch.from_numpy(batch['features']).to(device, non_blocking=True)
            labels = torch.from_numpy(batch['labels']).to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type, enabled=config.use_amp):
                outputs = model(features)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item() * features.size(0)
            val_col_acc += column_accuracy(outputs, labels).item() * features.size(0)
            val_iou += interval_iou(outputs, labels).item() * features.size(0)
            
            val_roi_iou += roi_interval_iou(outputs, labels, char_width=global_char_width).item() * features.size(0)
            split_iou = split_interval_iou(outputs, labels, char_width=global_char_width)
            val_char_iou += split_iou['char_iou'].item() * features.size(0)
            val_gap_iou += split_iou['gap_iou'].item() * features.size(0)
            
            val_count += features.size(0)
    
    val_loss /= val_count
    val_col_acc /= val_count
    val_iou /= val_count
    val_roi_iou /= val_count
    val_char_iou /= val_count
    val_gap_iou /= val_count
    
    print(f"  损失: {val_loss:.4f}")
    print(f"  列级准确率: {val_col_acc:.4f}")
    print(f"  区间IoU: {val_iou:.4f}")
    print(f"  ROI-IoU: {val_roi_iou:.4f}")
    print(f"  Char-IoU: {val_char_iou:.4f}")
    print(f"  Gap-IoU: {val_gap_iou:.4f}")


@click.command("train")
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
    """
    训练字符分割 1D UNet 模型

    使用规则切割结果作为训练标签，训练一个列级分割模型。
    """
    from train_config import TrainConfig

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


if __name__ == "__main__":
    cli()
