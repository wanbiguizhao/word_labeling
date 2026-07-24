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
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item() * features.size(0)
        train_col_acc += column_accuracy_multiclass(outputs.detach(), labels.detach()).item() * features.size(0)
        train_iou += interval_iou_multiclass(outputs.detach(), labels.detach()).item() * features.size(0)
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
                loss = criterion(outputs, labels)
            
            val_loss += loss.item() * features.size(0)
            val_col_acc += column_accuracy_multiclass(outputs, labels).item() * features.size(0)
            val_iou += interval_iou_multiclass(outputs, labels).item() * features.size(0)
            
            val_roi_iou += roi_interval_iou_multiclass(outputs, labels, char_width=global_char_width).item() * features.size(0)
            split_iou = split_interval_iou_multiclass(outputs, labels, char_width=global_char_width)
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
    mode: str = "train",
    no_validation: bool = False
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
        
        if no_validation:
            val_loss, val_col_acc, val_iou, val_roi_iou, val_char_iou, val_gap_iou = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            if lr_scheduler_type == "cosine":
                scheduler.step(epoch)
            else:
                scheduler.step(train_loss)
        else:
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
            
            if HAS_TQDM:
                epoch_range.write(f"\n{epoch_str} | {lr_str}")
                epoch_range.write(f"  {train_str}")
                if not no_validation:
                    val_str = f"Val:   Loss={val_loss:.4f}, ColAcc={val_col_acc:.4f}, IoU={val_iou:.4f}"
                    val_roi_str = f"ROI-IoU={val_roi_iou:.4f}, Char-IoU={val_char_iou:.4f}, Gap-IoU={val_gap_iou:.4f}"
                    epoch_range.write(f"  {val_str}")
                    epoch_range.write(f"  {val_roi_str}")
            else:
                print(f"{epoch_str} | {lr_str}")
                print(f"  {train_str}")
                if not no_validation:
                    val_str = f"Val:   Loss={val_loss:.4f}, ColAcc={val_col_acc:.4f}, IoU={val_iou:.4f}"
                    val_roi_str = f"ROI-IoU={val_roi_iou:.4f}, Char-IoU={val_char_iou:.4f}, Gap-IoU={val_gap_iou:.4f}"
                    print(f"  {val_str}")
                    print(f"  {val_roi_str}")
        
        if not no_validation and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = model_dir / f"{model_name}_best.pth"
            torch.save(model.state_dict(), str(best_path))
            
            if HAS_TQDM:
                epoch_range.write(f"  [INFO] 保存最佳模型: {best_path}")
            else:
                print(f"  [INFO] 保存最佳模型: {best_path}")
        
        if HAS_TQDM:
            if no_validation:
                epoch_range.set_postfix({
                    "Train Loss": f"{train_loss:.4f}",
                    "Train IoU": f"{train_iou:.4f}"
                })
            else:
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
                loss = criterion(outputs, labels)
            
            val_loss += loss.item() * features.size(0)
            val_col_acc += column_accuracy_multiclass(outputs, labels).item() * features.size(0)
            val_iou += interval_iou_multiclass(outputs, labels).item() * features.size(0)
            
            val_roi_iou += roi_interval_iou_multiclass(outputs, labels, char_width=global_char_width).item() * features.size(0)
            split_iou = split_interval_iou_multiclass(outputs, labels, char_width=global_char_width)
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=alpha, ignore_index=ignore_index, reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def column_accuracy_multiclass(pred, target):
    pred_class = torch.argmax(pred, dim=1)
    correct = (pred_class == target).float().sum()
    total = (target != -100).float().sum()
    return correct / max(total, 1)


def interval_iou_multiclass(pred, target, char_width: float = 16.0):
    pred_class = torch.argmax(pred, dim=1)
    pred_bin = (pred_class > 0).float()
    target_bin = (target > 0).float()
    
    intersection = (pred_bin * target_bin).sum(dim=-1)
    union = pred_bin.sum(dim=-1) + target_bin.sum(dim=-1) - intersection
    
    smooth = 1e-6
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def roi_interval_iou_multiclass(pred, target, char_width: float = 16.0):
    pred_class = torch.argmax(pred, dim=1)
    pred_bin = (pred_class > 0).float()
    
    target_int = target.int()
    if target_int.dim() == 2:
        batch_size = target_int.size(0)
        width = target_int.size(1)
    else:
        batch_size = 1
        width = target_int.size(0)
        target_int = target_int.unsqueeze(0)
        pred_bin = pred_bin.unsqueeze(0)
    
    total_iou = 0.0
    count = 0
    
    for i in range(batch_size):
        line_target = target_int[i]
        diff = line_target[1:] - line_target[:-1]
        starts = (diff > 0).nonzero(as_tuple=True)[0] + 1
        ends = (diff < 0).nonzero(as_tuple=True)[0]
        
        if line_target[0] > 0:
            starts = torch.cat([torch.tensor([0], device=line_target.device), starts])
        if line_target[-1] > 0:
            ends = torch.cat([ends, torch.tensor([width - 1], device=line_target.device)])
        
        if len(starts) == 0 or len(ends) == 0:
            continue
        
        roi_start = max(0, int(starts[0].item() - char_width))
        roi_end = min(width - 1, int(ends[-1].item() + char_width))
        
        if roi_end <= roi_start:
            continue
        
        roi_pred = pred_bin[i, roi_start:roi_end + 1]
        roi_target = (line_target[roi_start:roi_end + 1] > 0).float()
        
        intersection = (roi_pred * roi_target).sum()
        union = roi_pred.sum() + roi_target.sum() - intersection
        
        if union > 0:
            total_iou += (intersection + 1e-6) / (union + 1e-6)
            count += 1
    
    return total_iou / max(count, 1)


def split_interval_iou_multiclass(pred, target, char_width: float = 16.0):
    pred_class = torch.argmax(pred, dim=1)
    pred_bin = (pred_class > 0).float()
    
    target_int = target.int()
    if target_int.dim() == 2:
        batch_size = target_int.size(0)
        width = target_int.size(1)
    else:
        batch_size = 1
        width = target_int.size(0)
        target_int = target_int.unsqueeze(0)
        pred_bin = pred_bin.unsqueeze(0)
    
    total_char_iou = 0.0
    total_gap_iou = 0.0
    char_count = 0
    gap_count = 0
    
    for i in range(batch_size):
        line_target = target_int[i]
        
        diff = line_target[1:] - line_target[:-1]
        starts = (diff > 0).nonzero(as_tuple=True)[0] + 1
        ends = (diff < 0).nonzero(as_tuple=True)[0]
        
        if line_target[0] > 0:
            starts = torch.cat([torch.tensor([0], device=line_target.device), starts])
        if line_target[-1] > 0:
            ends = torch.cat([ends, torch.tensor([width - 1], device=line_target.device)])
        
        if len(starts) == 0 or len(ends) == 0:
            continue
        
        char_mask = (line_target > 0).float()
        char_pred = pred_bin[i]
        char_intersection = (char_pred * char_mask).sum()
        char_union = char_pred.sum() + char_mask.sum() - char_intersection
        
        if char_mask.sum() > 0:
            total_char_iou += (char_intersection + 1e-6) / (char_union + 1e-6) if char_union > 0 else 0
            char_count += 1
        
        gap_intersection = 0.0
        gap_union = 0.0
        
        for j in range(len(ends)):
            if j < len(starts) - 1:
                gap_start = ends[j] + 1
                gap_end = starts[j + 1] - 1
                gap_width = gap_end - gap_start + 1
                
                if gap_width > 0 and gap_width <= char_width:
                    gap_pred = pred_bin[i, gap_start:gap_end + 1]
                    gap_tgt = (line_target[gap_start:gap_end + 1] > 0).float()
                    
                    gap_pred_inv = 1.0 - gap_pred
                    gap_tgt_inv = 1.0 - gap_tgt
                    
                    gap_intersection += (gap_pred_inv * gap_tgt_inv).sum()
                    gap_union += gap_pred_inv.sum() + gap_tgt_inv.sum() - (gap_pred_inv * gap_tgt_inv).sum()
        
        if gap_union > 0:
            total_gap_iou += (gap_intersection + 1e-6) / (gap_union + 1e-6)
            gap_count += 1
    
    return {
        'char_iou': total_char_iou / max(char_count, 1),
        'gap_iou': total_gap_iou / max(gap_count, 1)
    }
