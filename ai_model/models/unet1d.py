import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv1D -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)  # 输出 logits，sigmoid 在 loss 中处理


class UNet1D(nn.Module):
    def __init__(self, n_channels=6, n_classes=1):
        super(UNet1D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
    
    def forward(self, inputs, targets, smooth=1e-6):
        # inputs = logits, targets = {0,1}
        
        # BCE 部分：使用 with_logits 版本，AMP 安全
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # Dice 部分：需要概率值 [0,1]
        probs = torch.sigmoid(inputs).view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs.sum() + targets_flat.sum() + smooth)
        
        return bce + dice_loss


def column_accuracy(pred, target, threshold=0.5):
    pred_bin = (pred >= threshold).float()
    correct = (pred_bin == target).float().sum()
    total = target.numel()
    return correct / total


def interval_iou(pred, target, threshold=0.5, char_width: float = 16.0):
    pred_bin = (pred >= threshold).float()
    
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    
    smooth = 1e-6
    return (intersection + smooth) / (union + smooth)


def roi_interval_iou(pred, target, threshold=0.5, char_width: float = 16.0):
    """
    ROI-IOU：只在有效区域内计算IOU，排除首尾空白区域
    
    Args:
        pred: 预测概率图 (batch, width) 或 (width,)
        target: 目标标签 (batch, width) 或 (width,)
        threshold: 二值化阈值
        char_width: 字符宽度，用于确定有效区域范围
    
    Returns:
        ROI内的IOU值
    """
    pred_bin = (pred >= threshold).float()
    
    # 提取目标中的字符区间
    target_int = target.int()
    if target_int.dim() == 2:
        batch_size = target_int.size(0)
        width = target_int.size(1)
    else:
        batch_size = 1
        width = target_int.size(0)
        target_int = target_int.unsqueeze(0)
    
    total_iou = 0.0
    count = 0
    
    for i in range(batch_size):
        # 找到所有字符区域的起始和结束位置
        line_target = target_int[i]
        
        # 计算梯度找到边界
        diff = line_target[1:] - line_target[:-1]
        starts = (diff == 1).nonzero(as_tuple=True)[0] + 1
        ends = (diff == -1).nonzero(as_tuple=True)[0]
        
        # 处理开头和结尾
        if line_target[0] == 1:
            starts = torch.cat([torch.tensor([0], device=line_target.device), starts])
        if line_target[-1] == 1:
            ends = torch.cat([ends, torch.tensor([width - 1], device=line_target.device)])
        
        if len(starts) == 0 or len(ends) == 0:
            continue
        
        # 计算有效区域：字符区间并集 + 一个字符宽度扩展
        roi_start = max(0, int(starts[0].item() - char_width))
        roi_end = min(width - 1, int(ends[-1].item() + char_width))
        
        if roi_end <= roi_start:
            continue
        
        # 在ROI内计算IOU
        roi_pred = pred_bin[i, roi_start:roi_end + 1] if pred_bin.dim() == 2 else pred_bin[roi_start:roi_end + 1]
        roi_target = line_target[roi_start:roi_end + 1].float()
        
        intersection = (roi_pred * roi_target).sum()
        union = roi_pred.sum() + roi_target.sum() - intersection
        
        if union > 0:
            total_iou += (intersection + 1e-6) / (union + 1e-6)
            count += 1
    
    return total_iou / max(count, 1)


def split_interval_iou(pred, target, threshold=0.5, char_width: float = 16.0):
    """
    拆分IOU：分别计算字符区IOU和间隙区IOU
    
    Args:
        pred: 预测概率图 (batch, width) 或 (width,)
        target: 目标标签 (batch, width) 或 (width,)
        threshold: 二值化阈值
        char_width: 字符宽度，用于确定间隙区域
    
    Returns:
        dict: {'char_iou': 字符区IOU, 'gap_iou': 间隙区IOU}
    """
    pred_bin = (pred >= threshold).float()
    
    target_int = target.int()
    if target_int.dim() == 2:
        batch_size = target_int.size(0)
        width = target_int.size(1)
    else:
        batch_size = 1
        width = target_int.size(0)
        target_int = target_int.unsqueeze(0)
    
    total_char_iou = 0.0
    total_gap_iou = 0.0
    char_count = 0
    gap_count = 0
    
    for i in range(batch_size):
        line_target = target_int[i]
        
        # 找到所有字符区域的起始和结束位置
        diff = line_target[1:] - line_target[:-1]
        starts = (diff == 1).nonzero(as_tuple=True)[0] + 1
        ends = (diff == -1).nonzero(as_tuple=True)[0]
        
        if line_target[0] == 1:
            starts = torch.cat([torch.tensor([0], device=line_target.device), starts])
        if line_target[-1] == 1:
            ends = torch.cat([ends, torch.tensor([width - 1], device=line_target.device)])
        
        if len(starts) == 0 or len(ends) == 0:
            continue
        
        # 计算字符区IOU
        char_mask = line_target.float()
        char_pred = pred_bin[i] if pred_bin.dim() == 2 else pred_bin
        char_intersection = (char_pred * char_mask).sum()
        char_union = char_pred.sum() + char_mask.sum() - char_intersection
        
        if char_mask.sum() > 0:
            total_char_iou += (char_intersection + 1e-6) / (char_union + 1e-6) if char_union > 0 else 0
            char_count += 1
        
        # 计算间隙区IOU（只考虑小于一个字符宽度的间隙）
        gap_intersection = 0.0
        gap_union = 0.0
        
        for j in range(len(ends)):
            if j < len(starts) - 1:
                gap_start = ends[j] + 1
                gap_end = starts[j + 1] - 1
                gap_width = gap_end - gap_start + 1
                
                if gap_width > 0 and gap_width <= char_width:
                    gap_pred = pred_bin[i, gap_start:gap_end + 1] if pred_bin.dim() == 2 else pred_bin[gap_start:gap_end + 1]
                    gap_tgt = line_target[gap_start:gap_end + 1].float()
                    
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