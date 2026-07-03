import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    device_type: str,
    use_amp: bool,
    warmup_epochs: int,
    current_epoch: int,
    base_lr: float
) -> Tuple[float, float, float]:
    model.train()
    train_loss = 0.0
    train_col_acc = 0.0
    train_iou = 0.0
    train_count = 0
    
    if current_epoch < warmup_epochs:
        warmup_factor = (current_epoch + 1) / warmup_epochs
        current_lr = base_lr * warmup_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    else:
        current_lr = optimizer.param_groups[0]['lr']
    
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)
    
    batch_range = tqdm(train_loader, desc=f"Epoch {current_epoch+1}", unit="batch", leave=False) if HAS_TQDM else train_loader
    
    for batch in batch_range:
        features = torch.from_numpy(batch['features']).to(device, non_blocking=True)
        labels = torch.from_numpy(batch['labels']).to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type, enabled=use_amp):
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
    
    return train_loss / train_count, train_col_acc / train_count, train_iou / train_count, current_lr


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    device_type: str,
    use_amp: bool,
    global_char_width: float
) -> Tuple[float, float, float, float, float, float]:
    model.eval()
    val_loss = 0.0
    val_col_acc = 0.0
    val_iou = 0.0
    val_roi_iou = 0.0
    val_char_iou = 0.0
    val_gap_iou = 0.0
    val_count = 0
    
    val_range = tqdm(val_loader, desc="验证", unit="batch", leave=False) if HAS_TQDM else val_loader
    
    with torch.no_grad():
        for batch in val_range:
            features = torch.from_numpy(batch['features']).to(device, non_blocking=True)
            labels = torch.from_numpy(batch['labels']).to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type, enabled=use_amp):
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
    
    return (
        val_loss / val_count,
        val_col_acc / val_count,
        val_iou / val_count,
        val_roi_iou / val_count,
        val_char_iou / val_count,
        val_gap_iou / val_count
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    device_type: str,
    num_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    use_amp: bool,
    log_interval: int,
    checkpoint_dir: str,
    model_name: str,
    lr_scheduler_type: str,
    global_char_width: float = 16.0,
    mode: str = "train"
) -> Dict:
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_col_acc': [],
        'val_col_acc': [],
        'train_iou': [],
        'val_iou': [],
        'val_roi_iou': [],
        'val_char_iou': [],
        'val_gap_iou': [],
        'learning_rate': []
    }
    
    model_dir = Path(checkpoint_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    epoch_range = tqdm(range(num_epochs), desc=f"{mode}进度", unit="epoch") if HAS_TQDM else range(num_epochs)
    
    for epoch in epoch_range:
        train_loss, train_col_acc, train_iou, current_lr = train_epoch(
            model, train_loader, criterion, optimizer, device,
            device_type, use_amp, warmup_epochs, epoch, base_lr
        )
        
        history['learning_rate'].append(current_lr)
        
        val_loss, val_col_acc, val_iou, val_roi_iou, val_char_iou, val_gap_iou = validate_epoch(
            model, val_loader, criterion, device, device_type, use_amp, global_char_width
        )
        
        if lr_scheduler_type == "cosine":
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
        
        if (epoch + 1) % log_interval == 0:
            epoch_str = f"Epoch [{epoch+1}/{num_epochs}]"
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
            best_path = model_dir / f"{model_name}_best.pth"
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


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    device_type: str,
    use_amp: bool,
    global_char_width: float
) -> Dict:
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
            
            with torch.amp.autocast(device_type, enabled=use_amp):
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
    
    return {
        'loss': val_loss / val_count,
        'col_acc': val_col_acc / val_count,
        'iou': val_iou / val_count,
        'roi_iou': val_roi_iou / val_count,
        'char_iou': val_char_iou / val_count,
        'gap_iou': val_gap_iou / val_count
    }


def save_model_and_history(
    model: nn.Module,
    history: Dict,
    checkpoint_dir: str,
    model_name: str,
    history_file: str = "training_history.json"
) -> None:
    model_dir = Path(checkpoint_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    final_path = model_dir / f"{model_name}_final.pth"
    torch.save(model.state_dict(), str(final_path))
    
    history_path = model_dir / history_file
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    
    print(f"[INFO] 模型保存到: {model_dir}")


def setup_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    min_lr: float = 1e-6,
    factor: float = 0.5,
    patience: int = 5
):
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=min_lr
        )
    else:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min',
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )


from ai_model.models.unet1d import column_accuracy, interval_iou, roi_interval_iou, split_interval_iou
