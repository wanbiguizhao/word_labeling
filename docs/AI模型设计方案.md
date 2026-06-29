# 1D U-Net 字符分割模型设计方案

## 版本记录

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| v1.0 | 2026-06-26 | System | 初始版本，1D U-Net模型设计 |
| v1.1 | 2026-06-28 | System | 文档修正：标签生成已使用后处理合并逻辑，引用实际代码实现 |
| v1.2 | 2026-06-29 | System | 新增ROI-IOU和拆分IOU指标；字符宽度自动计算；数据集预计算优化 |

---

## 一、问题定义

### 1.1 核心目标

将一行包含多类字符（汉字、数字、标点等）的印刷文本图像，拆分为逐个独立的完整字符单元，满足工业级的极高准确率要求。

### 1.2 问题转化

将字符分割问题转化为**1D语义分割任务**：
- 输入：行图像的1D特征序列（宽度方向）
- 输出：每个列位置属于"字符"或"非字符"的概率
- 通过后处理提取字符区间边界

### 1.3 优势

| 维度 | 传统规则方法 | 1D深度学习方法 |
|------|-------------|---------------|
| 精度 | 中等，依赖手工调参 | 高，自动学习特征 |
| 边界处理 | 粗糙，依赖阈值 | 精确，平滑过渡 |
| 泛化能力 | 差，针对特定场景 | 强，适应多样字体 |
| 类间区分 | 困难 | 自动学习差异 |

---

## 二、模型架构设计

### 2.1 1D U-Net 架构

```
输入 (W × 6)
    ↓
┌─────────────┐
│  DoubleConv │ → (W × 64)
└─────────────┘
    ↓ MaxPool (W/2)
┌─────────────┐
│  DoubleConv │ → (W/2 × 128)
└─────────────┘
    ↓ MaxPool (W/4)
┌─────────────┐
│  DoubleConv │ → (W/4 × 256)
└─────────────┘
    ↓ MaxPool (W/8)
┌─────────────┐
│  DoubleConv │ → (W/8 × 512) ← 瓶颈层
└─────────────┘
    ↓ UpConv (W/4)
┌─────────────────────────┐
│  Concatenate + DoubleConv│ → (W/4 × 256)
│   (Skip Connection)     │
└─────────────────────────┘
    ↓ UpConv (W/2)
┌─────────────────────────┐
│  Concatenate + DoubleConv│ → (W/2 × 128)
│   (Skip Connection)     │
└─────────────────────────┘
    ↓ UpConv (W)
┌─────────────────────────┐
│  Concatenate + DoubleConv│ → (W × 64)
│   (Skip Connection)     │
└─────────────────────────┘
    ↓ Conv1x1 + Sigmoid
输出 (W × 1) - 字符概率
```

### 2.2 核心组件

#### 2.2.1 DoubleConv

```python
class DoubleConv(nn.Module):
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
```

#### 2.2.2 Down（下采样）

```python
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)
```

#### 2.2.3 Up（上采样 + Skip Connection）

```python
class Up(nn.Module):
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
```

#### 2.2.4 OutConv（输出层）

```python
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return torch.sigmoid(self.conv(x))
```

### 2.3 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入特征通道 | 6 | 6种1D特征 |
| 输出通道 | 1 | 字符概率(0-1) |
| 编码器通道 | 64→128→256→512 | 逐步增加 |
| 解码器通道 | 512→256→128→64 | 逐步减少 |
| 卷积核大小 | 3 | 1D卷积 |
| 下采样方式 | MaxPool1d(2) | 步长为2 |
| 上采样方式 | ConvTranspose1d(2,2) | 转置卷积 |

---

## 三、特征提取设计

### 3.1 特征列表

| 特征名称 | 计算方式 | 维度 | 作用 |
|----------|----------|------|------|
| 垂直投影均值 | mean(img, axis=0) | W | 整体灰度分布 |
| 垂直投影方差 | var(img, axis=0) | W | 行内灰度变化 |
| 非零像素比 | count_nonzero > 0.1 / H | W | 字符密度 |
| 3窗口局部投影 | mean(img[:,i-1:i+2]) | W | 局部平滑特征 |
| 5窗口局部投影 | mean(img[:,i-2:i+3]) | W | 更大范围平滑 |
| 梯度投影 | mean(\|Sobel_x\|, axis=0) | W | 边缘/边界检测 |

### 3.2 特征提取流程

```python
def extract(line_img):
    # 1. 归一化
    line_img_norm = line_img / 255.0
    
    # 2. 计算6种特征
    proj_mean = np.mean(line_img_norm, axis=0)
    proj_var = np.var(line_img_norm, axis=0)
    proj_nonzero = np.count_nonzero(line_img_norm > 0.1, axis=0) / h
    
    # 3. 局部窗口特征
    pad_img = np.pad(line_img_norm, ((0,0),(1,1)), mode='constant')
    local_proj = np.array([np.mean(pad_img[:,i:i+3]) for i in range(w)])
    
    # 4. 梯度特征
    grad_x = cv2.Sobel(line_img_norm, cv2.CV_64F, 1, 0, ksize=3)
    grad_proj = np.mean(np.abs(grad_x), axis=0)
    
    # 5. 堆叠特征
    features = np.stack([proj_mean, proj_var, proj_nonzero, 
                        local_proj, grad_proj, local_proj5], axis=-1)
    
    return features  # shape: (W, 6)
```

---

## 四、数据预处理

### 4.1 图像预处理

为避免图像放大失真，采用**只缩小不放大**策略：

```python
def resize_image(line_img, target_height=64):
    h, w = line_img.shape
    
    # 计算缩放比例（只缩小）
    scale = min(target_height / h, 1.0)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # 双线性插值缩放
    resized_img = cv2.resize(line_img, (new_w, new_h), 
                            interpolation=cv2.INTER_LINEAR)
    
    # 创建白色画布并居中放置
    canvas = np.ones((target_height, new_w), dtype=np.uint8) * 255
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset + new_h, :new_w] = resized_img
    
    return canvas, scale
```

### 4.2 标签生成

标签生成在 `dataset.py` 的 `LabelGenerator.generate()` 中实现，**已包含后处理合并逻辑**：

```python
# dataset.py 第84-134行

def generate(
    char_segments: List[Dict],
    image_width: int,
    image_height: int,
    scale: float = 1.0,
    merge_enabled: bool = True  # 默认启用合并
) -> np.ndarray:
    """从字符段生成标签数组"""
    label = np.zeros(image_width, dtype=np.float32)

    # 1. 提取区间并转换到缩放后的坐标
    intervals = []
    for char in char_segments:
        start = int(round(char.get('col_start', 0) * scale))
        end = int(round(char.get('col_end', 0) * scale))
        intervals.append((start, end))

    # 2. 执行合并逻辑（与 postprocess_merge_chars 一致）
    if merge_enabled and len(intervals) >= 2:
        intervals = LabelGenerator._merge_narrow_chars(
            intervals, image_height,
            min_gap=int(MERGE_MIN_GAP * scale),      # 默认3
            max_height_diff=MERGE_MAX_HEIGHT_DIFF,   # 默认0.2
            single_ratio=MERGE_SINGLE_ASPECT_RATIO,  # 默认0.7
            min_ratio=MERGE_MIN_ASPECT_RATIO,        # 默认0.5
            max_ratio=MERGE_MAX_ASPECT_RATIO         # 默认1.5
        )

    # 3. 生成标签
    for start, end in intervals:
        label[start:end+1] = 1.0

    return label
```

### 4.3 合并逻辑实现

合并逻辑在 `LabelGenerator._merge_narrow_chars()` 中实现，考虑多种因素：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_gap` | 3 | 最小间隙阈值 |
| `max_height_diff` | 0.2 | 最大高度差异比例 |
| `single_ratio` | 0.7 | 单字符宽高比阈值 |
| `min_ratio` | 0.5 | 最小宽高比 |
| `max_ratio` | 1.5 | 最大宽高比 |

**与 `IntervalExtractor.postprocess()` 使用相同的合并策略**，确保训练标签与推理结果一致。

### 4.4 标签生成流程图

```
规则分割结果 (char_segments)
        │
        ▼
┌─────────────────────────┐
│  1. 提取原始区间         │  [(10, 15), (18, 22), ...]
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  2. _merge_narrow_chars │  考虑高度差、宽高比、间隙
│     (合并逻辑)           │  与后处理一致
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  3. 生成1D标签           │  字符区域=1，间隙=0
└─────────────────────────┘
        │
        ▼
      标签数组
```

### 4.5 数据增强（可选）

| 增强方式 | 概率 | 参数 | 说明 |
|----------|------|------|------|
| 随机噪声 | 0.3 | sigma=5-15 | 模拟扫描噪声 |
| 对比度调整 | 0.3 | alpha=0.8-1.2 | 增强泛化能力 |
| 随机裁剪 | 0.2 | 保留>80%宽度 | 模拟局部文本 |

---

## 五、损失函数设计

### 5.1 混合损失

采用 **Dice Loss + BCE Loss** 组合：

```python
class DiceBCELoss(nn.Module):
    def forward(self, inputs, targets, smooth=1e-6):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Dice Loss - 关注边界精度
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / \
                    (inputs.sum() + targets.sum() + smooth)
        
        # BCE Loss - 关注全局分类
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return bce + dice_loss
```

### 5.2 损失设计理由

| 损失函数 | 优势 | 不足 |
|----------|------|------|
| BCE Loss | 训练稳定，全局优化 | 边界模糊，类别不平衡敏感 |
| Dice Loss | 边界精确，对不平衡鲁棒 | 训练不稳定，梯度消失 |
| **混合损失** | **兼顾两者优势** | **无明显不足** |

---

## 六、评估指标

### 6.1 列级准确率

```python
def column_accuracy(pred, target, threshold=0.5):
    pred_bin = (pred >= threshold).float()
    correct = (pred_bin == target).float().sum()
    total = target.numel()
    return correct / total
```

### 6.2 区间IoU（全局）

**设计目标**：评估预测区间与真实区间的整体重叠度，作为最基础的区间级评估指标。

**评估逻辑**：将预测和目标都二值化后，计算全局范围内的交集与并集之比。

**公式**：
```
IoU = (预测∩目标) / (预测∪目标)
```

**代码实现**：
```python
def interval_iou(pred, target, threshold=0.5):
    pred_bin = (pred >= threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)
```

**优缺点分析**：

| 维度 | 说明 |
|------|------|
| **优势** | 计算简单，反映整体区间质量 |
| **不足** | 首尾空白区域会拉高IoU，无法区分字符区和间隙区的错误 |
| **适用场景** | 快速评估、基线对比 |

**示例**：
```
目标：███  ████  ████（字符区）
预测：███ ████ ████（间隙区预测错误）
全局IoU：95%（首尾空白贡献较大）
```

### 6.3 ROI-IoU（有效区域）

为避免首尾空白区域影响，仅在字符区间并集 + 一个字符宽度扩展区域内计算：

```python
def roi_interval_iou(pred, target, char_width, threshold=0.5):
    pred_bin = (pred >= threshold).float()
    
    # 提取字符区间
    pred_intervals = extract_intervals(pred_bin)
    target_intervals = extract_intervals(target)
    
    # 计算有效区域
    all_intervals = pred_intervals + target_intervals
    if all_intervals:
        roi_start = max(0, min(s for s, e in all_intervals) - char_width)
        roi_end = min(len(target), max(e for s, e in all_intervals) + char_width)
    else:
        roi_start, roi_end = 0, len(target)
    
    # 仅在ROI内计算IOU
    pred_roi = pred_bin[roi_start:roi_end]
    target_roi = target[roi_start:roi_end]
    ...
```

### 6.4 拆分IoU（字符区+间隙区）

分别计算字符区和间隙区的IOU，更精细地评估模型性能：

```python
def split_interval_iou(pred, target, char_width, threshold=0.5):
    pred_bin = (pred >= threshold).float()
    
    # 计算字符区IoU（仅在真实字符位置）
    char_mask = target > 0.5
    char_iou = iou(pred_bin[char_mask], target[char_mask])
    
    # 计算间隙区IoU（仅在小于一个字符宽度的间隙）
    gap_mask = (target == 0) & (gap_width < char_width)
    gap_iou = iou(pred_bin[gap_mask], target[gap_mask])
    
    return {'char_iou': char_iou, 'gap_iou': gap_iou}
```

### 6.5 指标说明

| 指标 | 含义 | 目标值 |
|------|------|--------|
| 列级准确率 | 每列分类正确的比例 | >99% |
| 区间IoU | 预测区间与真实区间的重叠度（全局） | >95% |
| ROI-IoU | 仅在有效区域内的区间重叠度 | >95% |
| Char-IoU | 字符区域内的预测精度 | >98% |
| Gap-IoU | 间隙区域内的预测精度（<一个字符宽度） | >90% |

### 6.6 字符宽度自动计算

字符宽度从标注数据中自动统计，使用**中位数**而非均值，避免标点、数字等窄字符的长尾影响：

```python
# generate_dataset_split.py 中预计算
def compute_char_width_stats(data_base_path, line_ids):
    all_char_widths = []
    for line_id in line_ids:
        rule_data = load_rule_json(line_id)
        for char in rule_data['chars']:
            w = char.get('width', 0)
            if 3 <= w <= 100:  # 过滤异常值
                all_char_widths.append(w)
    
    return {
        'global_median_char_width': float(np.median(all_char_widths)),
        'global_mean_char_width': float(np.mean(all_char_widths)),
        'line_median_char_widths': {line_id: median_width},
    }
```

**数据流向**：
```
generate_dataset_split.py（预计算）→ dataset_split.json（存储）
       ↓
train.py（读取）→ CharSegmentDataset（加载）→ 评估指标（使用）
```

---

## 七、训练流程

### 7.1 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | Adam | 自适应学习率 |
| 初始学习率 | 1e-4 | 避免震荡 |
| 学习率衰减 | ReduceLROnPlateau | 验证损失停止下降时衰减 |
| 衰减因子 | 0.5 | 每次衰减50% |
| 耐心值 | 5 | 5个epoch无改善则衰减 |
| 批大小 | 8 | 根据GPU内存调整 |
| epoch数 | 50 | 早期停止 |
| 早停耐心 | 10 | 10个epoch无改善则停止 |

### 7.2 训练流程

```
1. 数据加载
   ├── 读取行图像列表 (4609个)
   ├── 80%训练集 / 20%验证集
   └── 构建DataLoader (collate_fn动态padding)

2. 模型初始化
   ├── 创建1D U-Net模型
   ├── 移动到GPU/CPU
   └── 初始化优化器和损失函数

3. 训练循环
   ├── 前向传播 → 计算损失 → 反向传播 → 参数更新
   ├── 每epoch计算训练/验证指标
   ├── 学习率自适应调整
   └── 保存最佳模型

4. 训练完成
   ├── 保存最终模型
   ├── 保存训练历史
   └── 输出验证集评估结果
```

### 7.3 训练脚本

```python
# ai_model/train/train.py

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        for batch in train_loader:
            features = torch.tensor(batch['features']).to(device)
            labels = torch.tensor(batch['labels']).to(device)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                ...
        
        # 早停和学习率调整
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), 'char_segment_1d_unet_best.pth')
```

---

## 八、推理流程

### 8.1 推理步骤

```
1. 加载模型
   ├── 读取训练好的模型权重
   └── 设置为eval模式

2. 图像预处理
   ├── 读取行图像（灰度）
   ├── 统一高度为64px（只缩小不放大）
   └── 提取6维1D特征

3. 模型预测
   ├── 特征padding到max_width
   ├── 前向传播获取概率图
   └── 截断到实际宽度

4. 后处理
   ├── 阈值二值化 (0.5)
   ├── 提取连续字符区间
   ├── 合并小间隙区间 (<=2px)
   └── 坐标反缩放（恢复原始尺寸）

5. 输出结果
   └── 返回字符区间列表 [(start_col, end_col), ...]
```

### 8.2 推理脚本

```python
# ai_model/inference/infer.py

class CharSegmentPredictor:
    def __init__(self, model_path, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet1D(n_channels=6, n_classes=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def predict(self, line_img):
        # 特征提取（含预处理）
        features, resized_w, scale = FeatureExtractor.extract(line_img)
        
        # 模型预测
        features_tensor = torch.tensor(features.transpose(1,0)[np.newaxis]).to(self.device)
        with torch.no_grad():
            output = self.model(features_tensor)
        pred_prob = output.squeeze().cpu().numpy()[:resized_w]
        
        # 后处理提取区间
        intervals = IntervalExtractor.extract(pred_prob)
        
        # 坐标反缩放
        inv_scale = 1.0 / scale
        intervals_orig = [(int(round(s*inv_scale)), int(round(e*inv_scale))) 
                         for s, e in intervals]
        
        return intervals_orig, pred_prob
```

### 8.3 后处理：区间提取

```python
def extract(pred_prob, threshold=0.5):
    binary = (pred_prob > threshold).astype(np.int32)
    char_cols = np.where(binary == 1)[0]
    
    # 提取连续区间
    intervals = []
    start = char_cols[0]
    for col in char_cols[1:]:
        if col != prev + 1:
            intervals.append((start, prev))
            start = col
        prev = col
    intervals.append((start, prev))
    
    # 合并小间隙
    merged = [intervals[0]]
    for current in intervals[1:]:
        gap = current[0] - merged[-1][1] - 1
        if gap <= 2:
            merged[-1] = (merged[-1][0], current[1])
        else:
            merged.append(current)
    
    return merged
```

---

## 九、代码结构

### 9.1 目录结构

```
ai_model/
├── models/                    # 模型定义
│   └── unet1d.py              # 1D U-Net模型、损失函数、评估指标
├── data/                      # 数据处理
│   └── dataset.py             # 特征提取、标签生成、数据集、IntervalExtractor
├── train/                     # 训练模块
│   └── train.py               # 训练主脚本
└── inference/                 # 推理模块
    └── infer.py               # 预测类和推理脚本
```

### 9.2 文件职责

| 文件 | 职责 | 核心类/函数 |
|------|------|-------------|
| `models/unet1d.py` | 模型架构 | UNet1D, DiceBCELoss |
| `data/dataset.py` | 数据处理 | FeatureExtractor, LabelGenerator, CharSegmentDataset |
| `train/train.py` | 训练流程 | train_model(), main() |
| `inference/infer.py` | 推理接口 | CharSegmentPredictor |

---

## 十、集成方案

### 10.1 与现有流程集成

```
现有segment_manager流程：
PDF → 页面图像 → 行图像 → 规则分割 → 后处理 → 血缘索引

集成AI模型后：
PDF → 页面图像 → 行图像 → AI模型分割 → 后处理 → 血缘索引

切换方式：通过配置参数控制使用规则还是AI模型
```

### 10.2 配置设计

```python
class Line2CharConfig:
    segmentation_method: str = "ai"  # "rule" 或 "ai"
    ai_model_path: str = "ai_model/models/char_segment_1d_unet_best.pth"
```

### 10.3 集成接口

```python
# 在segment_manager.py中添加
def line_to_chars_ai(self, line_img, line_id):
    predictor = CharSegmentPredictor(self.char_cfg.ai_model_path)
    intervals, _ = predictor.predict(line_img)
    
    chars = []
    for idx, (start, end) in enumerate(intervals):
        char_info = {
            'char_id': f'char_{line_id}_{idx}',
            'col_start': start,
            'col_end': end,
            'width': end - start,
            ...
        }
        chars.append(char_info)
    
    return chars
```

---

## 十一、迭代闭环

### 11.1 数据迭代流程

```
1. 规则分割生成训练数据
   ├── 行图像 → 规则分割 → rule_json
   └── 生成特征和标签

2. 训练AI模型
   ├── 80%训练 / 20%验证
   └── 保存最佳模型

3. AI分割生成新数据
   ├── 行图像 → AI分割 → 更精确结果
   └── 人工抽检修正错误

4. 构建新训练集
   ├── 混合规则数据和人工修正数据
   └── 重新训练模型

5. 迭代优化
   └── 重复步骤2-4，逐步提升精度
```

### 11.2 性能监控

| 监控项 | 频率 | 目标 |
|--------|------|------|
| 列级准确率 | 每epoch | >99% |
| 区间IoU | 每epoch | >95% |
| 字符分割错误率 | 每日抽检 | <0.1% |
| 推理耗时 | 批量测试 | <10ms/行 |

---

## 十二、部署方案

### 12.1 模型导出

```python
# 导出为ONNX格式（可选）
dummy_input = torch.randn(1, 6, 2048).to(device)
torch.onnx.export(model, dummy_input, 
                  "char_segment_1d_unet.onnx",
                  opset_version=11,
                  input_names=["features"],
                  output_names=["probability"])
```

### 12.2 推理服务（可选）

```
FastAPI服务：
├── POST /predict
│   ├── 输入：行图像base64
│   └── 输出：字符区间JSON
├── POST /batch_predict
│   ├── 输入：多行图像base64列表
│   └── 输出：多行行字符区间JSON
└── GET /health
    └── 输出：服务状态
```

---

## 附录：依赖列表

| 依赖 | 版本 | 用途 |
|------|------|------|
| torch | >=2.10.0 | 深度学习框架 |
| torchvision | >=0.15.0 | 视觉工具 |
| opencv-python | >=4.8.0 | 图像处理 |
| numpy | >=1.26.0 | 数值计算 |