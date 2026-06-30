# 字符切割 CLI 工具文档

## 概述

本项目提供一套统一的命令行工具，覆盖 PDF 分割、模型训练、推理对比等完整工作流。
所有命令均使用 **Click** 框架构建，支持自动生成帮助文档、参数验证和友好的错误提示。

## 快速入门

```bash
# 查看全局帮助
python cli.py --help

# 查看子命令帮助
python cli.py segment --help
python cli.py train --help
python cli.py predict --help
```

---

## 命令体系

```
cli.py
├── segment          # PDF 文本分割流程
│   ├── single       # 处理单个 PDF 文件
│   └── batch        # 批量处理 PDF 文件
├── train            # 训练深度学习分割模型
│   ├── start                # 启动模型训练
│   ├── split-dataset        # 生成数据集划分文件
│   ├── active-learn         # 主动学习（基于模型）：找出最需标注的行
│   └── rule-based-al        # 主动学习（基于规则）：仅用规则识别可能切割错误的样本
└── predict          # 使用模型进行推理
    ├── line         # 对单行图像进行字符分割
    └── compare      # 对比规则与模型的切割结果
```

---

## 1. 数据分割 (segment)

### segment single — 处理单个 PDF

```bash
python cli.py segment single <pdf_name> [options]
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pdf_name` | 必填 | — | PDF 文件名（位于 datahome/raw/pdf/ 下） |
| `--parallel` | flag | — | 页面级并行处理 |
| `--data-base-path` | path | ./datahome | 数据基础目录 |

**示例：**

```bash
python cli.py segment single document.pdf --parallel
python cli.py segment single document.pdf --data-base-path /path/to/datahome
```

---

### segment batch — 批量处理 PDF

```bash
python cli.py segment batch [options]
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--start` | int | 0 | 起始索引（从0开始） |
| `--end` | int | — | 结束索引（不包含，默认为全部） |
| `--parallel` | flag | — | PDF 级别并行处理 |
| `--max-workers` | int | 4 | 并行进程数 |
| `--pattern` | str | `*.pdf` | PDF 文件匹配模式 |
| `--data-base-path` | path | ./datahome | 数据基础目录 |

**示例：**

```bash
# 处理所有 PDF
python cli.py segment batch

# 处理前 10 个 PDF，4 进程并行
python cli.py segment batch --start 0 --end 10 --parallel --max-workers 4

# 指定数据目录
python cli.py segment batch --data-base-path /path/to/datahome
```

---

## 2. 模型训练 (train)

### train start — 启动训练

```bash
python cli.py train start [options]
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--batch-size` | int | 8 | 批大小 |
| `--lr` | float | 1e-4 | 学习率 |
| `--epochs` | int | 50 | 训练轮数 |
| `--train-ratio` | float | 0.8 | 训练集比例 |
| `--device` | str | auto | 训练设备 (auto/cuda/cpu) |
| `--num-workers` | int | 4 | DataLoader 并行数 |
| `--use-amp`/`--no-amp` | flag | True | 是否启用混合精度训练 |
| `--checkpoint-dir` | str | models | 模型保存目录 |
| `--data-base-path` | str | datahome | 数据基础目录（相对项目根） |
| `--split-file` | str | ai_model/data/dataset_split.json | 数据集划分文件路径 |
| `--seed` | int | 42 | 随机种子 |

**示例：**

```bash
# 默认配置训练
python cli.py train start

# 自定义配置
python cli.py train start --batch-size 64 --lr 1e-4 --epochs 100 \
    --device cuda --num-workers 8

# 关闭 AMP
python cli.py train start --no-amp
```

**训练输出指标：**

训练过程中会输出以下指标：

```
[INFO] 全局中位字符宽度: 16.50 px

Epoch [10/50] | LR=1.00e-04
  Train: Loss=0.0523, ColAcc=0.9982, IoU=0.9856
  Val:   Loss=0.0612, ColAcc=0.9978, IoU=0.9821
  ROI-IoU=0.9567, Char-IoU=0.9789, Gap-IoU=0.8912
```

| 指标 | 说明 |
|------|------|
| Loss | 混合损失（BCE + Dice） |
| ColAcc | 列级准确率 |
| IoU | 全局区间IoU |
| ROI-IoU | 有效区域内的区间IoU（排除首尾空白） |
| Char-IoU | 字符区域内的预测精度 |
| Gap-IoU | 间隙区域内的预测精度（<一个字符宽度） |

---

### train split-dataset — 生成数据集划分

```bash
python cli.py train split-dataset [options]
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train-ratio` | float | 0.8 | 训练集比例 |
| `--seed` | int | 42 | 随机种子 |
| `--sort-by-width` | flag | — | 按宽度排序后划分（减少 padding 浪费） |
| `--data-base-path` | str | datahome | 数据基础目录（相对项目根） |
| `--output` | str | ai_model/data/dataset_split.json | 输出文件路径 |

**示例：**

```bash
python cli.py train split-dataset --train-ratio 0.9 --sort-by-width
```

**输出说明：**

命令执行后会输出数据集划分信息和**字符宽度统计**：

```
[INFO] 数据集划分完成!
[INFO] 总样本数: 1200
[INFO] 训练集: 960
[INFO] 验证集: 240
[INFO] 训练集平均宽度: 480
[INFO] 验证集平均宽度: 478

[INFO] 字符宽度统计:
[INFO] 总字符数: 15600
[INFO] 全局中位字符宽度: 16.50 px
[INFO] 全局平均字符宽度: 16.23 px
[INFO] 字符宽度范围: 4 - 64 px
```

字符宽度统计会保存到划分文件中，训练时自动加载，避免重复计算。

---

### train active-learn — 主动学习（基于模型）

```bash
python cli.py train active-learn <top_n> [options]
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `top_n` | int (必填) | 100 | 返回最需要标注的 Top N 行 |
| `--model-path` | path | models/char_segment_1d_unet_best.pth | 模型权重路径 |
| `--data-base-path` | path | ./datahome | 数据基础目录 |
| `--output` | str | models/al_ranking.json | 排名结果输出路径 |
| `--batch-size` | int | 32 | GPU批量推理批大小 |
| `--num-workers` | int | 4 | 并行特征提取和评分的线程数 |
| `--chunk-size` | int | 1000 | 微批量大小，每处理完一个chunk更新一次排名 |

**示例：**

```bash
# 返回 Top 100 最需要标注的行（默认配置）
python cli.py train active-learn 100

# 使用自定义模型和配置
python cli.py train active-learn 200 --model-path /path/to/model.pth \
    --batch-size 64 --num-workers 8

# 更小的chunk，更快看到结果
python cli.py train active-learn 100 --chunk-size 500
```

**评分指标：**

| 指标 | 权重 | 说明 |
|------|------|------|
| 不确定性 | 0.2 | 预测概率接近0.5的程度 |
| 熵 | 0.2 | 预测概率分布的混乱程度 |
| 分歧 | 0.3 | 规则和模型分割的IoU差异 |
| 数量差异 | 0.15 | 字符数量差异比例 |
| 边界位移 | 0.15 | 边界位置平均差异 |

---

### train rule-based-al — 主动学习（基于规则）

```bash
python cli.py train rule-based-al <top_n> [options]
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `top_n` | int (必填) | 100 | 返回最可能切割错误的 Top N 行 |
| `--data-base-path` | path | ./datahome | 数据基础目录 |
| `--chunk-size` | int | 1000 | 微批量大小，每处理完一个chunk更新一次排名 |

**示例：**

```bash
# 返回 Top 100 最可能切割错误的行
python cli.py train rule-based-al 100

# 指定数据目录
python cli.py train rule-based-al 100 --data-base-path /path/to/datahome
```

**评分指标（以合并后结果为主）：**

算法先对原始切割结果执行合并后处理，再以合并后的结果为核心进行评分，重点检测三类问题。

**核心指标（高权重）：**

| 指标 | 权重 | 说明 |
|------|------|------|
| 粘连字符 | 0.25 | 合并后宽高比 1.5~3.0 的字符比例（两个汉字没切割开） |
| 合并过度 | 0.20 | 合并后宽高比 > 1.5 的字符比例（不该合并的被合并了） |
| 合并不足 | 0.15 | 合并后仍有相邻窄字符+小间隙的比例（该合并的没合并） |

**辅助指标（中权重）：**

| 指标 | 权重 | 说明 |
|------|------|------|
| 合并后宽字符 | 0.10 | 合并后 width > 平均宽度×2.0 的字符比例 |
| 合并后间隙异常 | 0.08 | 合并后 gap < 2px 或 gap > 平均间隙×3.0 |
| 合并后宽度变异 | 0.07 | 合并后宽度标准差/平均宽度 |

**参考指标（低权重）：**

| 指标 | 权重 | 说明 |
|------|------|------|
| 合并后窄字符 | 0.05 | 合并后 width < 平均宽度×0.5 的字符比例 |
| 合并后极端宽高比 | 0.05 | 合并后 aspect_ratio < 0.3 或 > 3.0 |
| 合并率 | 0.05 | 合并前后字符数量减少比例（参考） |

**输出示例：**

```
[INFO] Top 100 最可能切割错误的行：
--------------------------------------------------------------------------------------------------------------
排名   行ID                                    AL分数      粘连       过度合并   合并不足   合并率    
--------------------------------------------------------------------------------------------------------------
1      line_page_pdf_gwyb195624_3_0            0.7852      0.3500     0.2000     0.1500     0.4000
2      line_page_pdf_gwyb195401_11_5           0.6543      0.2800     0.1800     0.1200     0.2500
```

**输出字段说明：**

| 字段 | 说明 |
|------|------|
| AL分数 | 综合评分，越高越可能切割错误 |
| 粘连 | 合并后宽高比 1.5~3.0 的字符比例（两个汉字粘在一起） |
| 过度合并 | 合并后宽高比 > 1.5 的字符比例（合并错误） |
| 合并不足 | 合并后仍有相邻窄字符+小间隙的比例（该合并没合并） |
| 合并率 | 合并前后字符数量减少比例 |

**两种主动学习方式对比：**

| 特性 | `active-learn` | `rule-based-al` |
|------|----------------|-----------------|
| 需要模型 | ✅ 是 | ❌ 否 |
| GPU加速 | ✅ 支持 | ❌ 不需要 |
| 速度 | 较慢（需要推理） | 很快（仅分析JSON） |
| 适用场景 | 有训练好的模型时 | 无模型或快速筛选时 |

---

## 3. 模型推理 (predict)

### predict line — 单行推理

```bash
python cli.py predict line <image_path> [options]
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image_path` | path (必填) | — | 行图像路径 |
| `--model-path` | path | models/char_segment_1d_unet_best.pth | 模型权重路径 |
| `--output` | str | — | 可视化结果保存路径（默认不保存） |
| `--threshold` | float | 0.5 | 字符概率阈值 |

**示例：**

```bash
# 推理并保存可视化结果
python cli.py predict line ./datahome/lines/xxx.png --output result.png

# 调整阈值
python cli.py predict line ./datahome/lines/xxx.png --threshold 0.3
```

---

### predict compare — 对比可视化

```bash
python cli.py predict compare <line_id> [options]
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `line_id` | str (必填) | — | 行 ID（如 page_pdf_1_line_0） |
| `--data-base-path` | path | ./datahome | 数据基础目录 |
| `--model-path` | path | models/char_segment_1d_unet_best.pth | 模型权重路径 |
| `--image-path` | path | — | 直接指定行图像路径（替代 data_base_path + line_id） |
| `--rule-json-path` | path | — | 直接指定规则 JSON 路径 |
| `--save-dir` | str | ./visualization | 可视化结果保存目录 |

**示例：**

```bash
# 通过 data_base_path + line_id
python cli.py predict compare page_pdf_1_line_0 --data-base-path ./datahome

# 直接指定文件路径
python cli.py predict compare dummy \
    --image-path ./datahome/lines/xxx.png \
    --rule-json-path ./datahome/rule_jsons/xxx_rule.json
```

生成的对比图包含 4 行：

```
┌─────────────────────────────┐
│  第1行：原始行图像            │
├─ 紫色分隔线 ─────────────────┤
│  第2行：规则切割结果          │
├─ 紫色分隔线 ─────────────────┤
│  第3行：模型预测结果          │
├─ 紫色分隔线 ─────────────────┤
│  第4行：概率热力图            │
└─────────────────────────────┘
```

颜色约定：**红色竖线** = 字符起点，**绿色竖线** = 字符终点，**黄色竖条** = 模型概率。

---

## 各模块独立调用

每个模块也支持独立调用（无需通过 `cli.py`）：

```bash
# 数据分割
python image_tools/segment_manager.py single <pdf_name>
python image_tools/segment_manager.py batch --start 0 --end 10

# 模型训练
python ai_model/train/train.py start --batch-size 64 --epochs 100

# 数据集划分
python ai_model/data/generate_dataset_split.py split-dataset --train-ratio 0.8

# 主动学习
python ai_model/train/active_learning.py active-learn 100

# 单行推理
python ai_model/inference/infer.py predict ./datahome/lines/xxx.png --output result.png

# 对比可视化
python ai_model/inference/visualize_comparison.py compare page_pdf_1_line_0 --data-base-path ./datahome
```

---

## 环境要求

- Python 3.8+
- Click 8.0+（已随环境安装）
- 基础依赖：torch, torchvision, opencv-python, pillow, numpy
