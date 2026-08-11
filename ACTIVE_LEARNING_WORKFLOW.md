# 主动学习工作流操作指南

## 概述

本指南记录基于主动学习的数据筛选、项目同步、推理结果同步的完整流程，方便快速复现。

---

## 流程总览

```
1. 全量推理 → 2. 主动学习排序 → 3. 更新项目line_id → 4. 同步推理结果 → 5. 同步缓存 → 6. 前端标注
```

---

## 步骤详解

### 步骤1：全量推理（使用新模型）

对所有规则分割数据进行模型推理，生成推理版本目录。

```bash
cd D:\projects\work_projects\data_service

# 使用最新模型进行全量推理
d:\projects\work_projects\data_service\venv\Scripts\python.exe cli.py project infer-all --output-version v0724
```

**参数说明：**
- `--output-version`: 输出版本目录名（如 `v0724`）
- `--model-path`: 指定模型路径（默认使用 `models/char_segment_1d_unet_best.pth`）
- `--threshold`: 概率阈值（默认 0.5）
- `--save-detail`: 是否保存详细结果（默认不保存）

**输出目录：** `datahome/model_inference/v0724/`
- `metadata.json` - 推理元数据（模型路径、参数、统计信息）
- `inference.jsonl` - 轻量结果（字符区间列表）
- `detail/` - 详细结果（可选）

---

### 步骤2：主动学习排序

基于推理结果和规则结果的差异，计算主动学习分数，筛选最需要标注的样本。

```bash
cd D:\projects\work_projects\data_service

# 基于推理结果的主动学习排序
d:\projects\work_projects\data_service\venv\Scripts\python.exe -m ai_model.train.inference_based_al --top-n 2000 --inference-version v0724
```

**参数说明：**
- `--top-n`: 返回前N个最难样本（默认100）
- `--inference-version`: 使用的推理版本（默认最新版本）
- `--output`: 输出路径（默认 `ai_model/models/inference_al_ranking.json`）

**输出文件：** `ai_model/models/inference_al_ranking.json`
- 包含字段：`line_id`, `al_score`, `disagreement`, `count_diff`, `boundary_diff`
- 按 `al_score` 降序排列（分数越高越需要标注）

**主动学习评分公式：**
```
AL分数 = 0.5 × 分歧度(IoU差异) + 0.3 × 数量差异 + 0.2 × 边界位移
```

---

### 步骤3：更新项目 line_id

将筛选的 line_id 更新到标注项目，排除已标注的数据。

**操作脚本：** 创建 `sync_project.py` 脚本：

```python
import json
from pathlib import Path

# 配置
PROJECT_ID = "proj_20260725_225531"
EXISTING_PROJECT_ID = "proj_20260706_230417"  # 已标注项目（用于排除重复）
INFERENCE_VERSION = "v0724"

# 路径设置
project_root = Path(f"datahome/project/{PROJECT_ID}")
al_ranking_path = Path("ai_model/models/inference_al_ranking.json")
existing_project_path = Path(f"datahome/project/{EXISTING_PROJECT_ID}")
v0724_dir = Path(f"datahome/model_inference/{INFERENCE_VERSION}")

# 读取 AL 排名结果
with open(al_ranking_path, 'r', encoding='utf-8') as f:
    al_ranking = json.load(f)

all_line_ids = [item['line_id'] for item in al_ranking]
print(f"[INFO] AL排名文件包含 {len(all_line_ids)} 个 line_id")

# 获取已标注的 line_id（排除重复）
existing_annotated = set()
existing_annotations_dir = existing_project_path / "annotations"
if existing_annotations_dir.exists():
    for anno_file in existing_annotations_dir.glob('*.json'):
        line_id = anno_file.stem.replace('_annotation', '')
        existing_annotated.add(line_id)
print(f"[INFO] 已标注项目包含 {len(existing_annotated)} 个 line_id")

# 过滤已标注的 line_id
overlap = set(all_line_ids) & existing_annotated
print(f"[INFO] 重叠的 line_id: {len(overlap)} 个")
filtered_line_ids = [lid for lid in all_line_ids if lid not in overlap]
print(f"[INFO] 过滤后剩余 {len(filtered_line_ids)} 个 line_id")

# 更新 line_id_list.json
line_id_list_path = project_root / "line_id_list.json"
line_id_list_data = {
    'total_count': len(filtered_line_ids),
    'line_ids': filtered_line_ids
}
with open(line_id_list_path, 'w', encoding='utf-8') as f:
    json.dump(line_id_list_data, f, ensure_ascii=False, indent=2)
print(f"[INFO] 已更新 line_id_list.json")

# 更新 project.json
project_json_path = project_root / "project.json"
with open(project_json_path, 'r', encoding='utf-8') as f:
    project_data = json.load(f)

project_data['line_id_count'] = len(filtered_line_ids)
project_data['description'] = f"基于推理结果主动学习筛选的标注项目，共{len(filtered_line_ids)}条待标注数据（已排除{len(overlap)}条已标注数据）"
project_data['updated_at'] = "2026-07-26T00:00:00.000000"

with open(project_json_path, 'w', encoding='utf-8') as f:
    json.dump(project_data, f, ensure_ascii=False, indent=2)
print(f"[INFO] 已更新 project.json")
```

**执行命令：**
```bash
d:\projects\work_projects\data_service\venv\Scripts\python.exe sync_project.py
```

---

### 步骤4：同步模型推理结果

将指定版本的推理结果复制到项目的 `model_jsons` 目录。

**操作脚本：** 在步骤3的脚本基础上添加以下代码：

```python
# 同步推理结果
inference_jsonl_path = v0724_dir / "inference.jsonl"
metadata_path = v0724_dir / "metadata.json"
model_jsons_dir = project_root / "model_jsons"
rule_jsons_dir = Path("datahome/rule_jsons")

# 读取元数据
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)
model_path = metadata.get('model_path', 'char_segment_1d_unet_best.pth')
print(f"[INFO] 使用模型: {model_path}")

# 构建推理结果映射
inference_map = {}
with open(inference_jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            line_id = data.get('line_id')
            if line_id and line_id in set(filtered_line_ids):
                inference_map[line_id] = data
        except Exception:
            continue

print(f"[INFO] 从 inference.jsonl 中找到 {len(inference_map)} 个匹配的 line_id")

# 确保输出目录存在
model_jsons_dir.mkdir(parents=True, exist_ok=True)

# 复制推理结果
success_count = 0
fail_count = 0
missing_count = 0

for line_id in filtered_line_ids:
    if line_id not in inference_map:
        missing_count += 1
        continue
    
    inf_data = inference_map[line_id]
    intervals = inf_data.get('chars', [])
    
    # 获取图像尺寸（从规则文件）
    rule_file = rule_jsons_dir / f"{line_id}_rule.json"
    img_width = 0
    img_height = 0
    if rule_file.exists():
        try:
            with open(rule_file, 'r', encoding='utf-8') as f:
                rule_data = json.load(f)
                img_width = rule_data.get('width', 0)
                img_height = rule_data.get('height', 0)
        except Exception:
            pass
    
    # 构建字符列表
    chars = []
    for idx, (start, end) in enumerate(intervals):
        chars.append({
            'char_id': f"{line_id}_char_{idx}",
            'line_id': line_id,
            'char_idx': idx,
            'col_start': start,
            'col_end': end,
            'width': end - start
        })
    
    # 输出数据
    output_data = {
        'line_id': line_id,
        'chars': chars,
        'char_count': len(chars),
        'width': img_width,
        'height': img_height,
        'threshold': metadata.get('params', {}).get('threshold', 0.5),
        'model_path': model_path
    }
    
    # 保存文件
    output_path = model_jsons_dir / f"{line_id}_model.json"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        success_count += 1
    except Exception as e:
        print(f"[ERROR] 保存失败 {line_id}: {e}")
        fail_count += 1

# 输出统计
print(f"\n[INFO] 复制完成!")
print(f"[INFO] 成功: {success_count}")
print(f"[INFO] 失败: {fail_count}")
print(f"[INFO] inference.jsonl 中缺失: {missing_count}")
print(f"[INFO] 输出目录: {model_jsons_dir}")
```

**执行命令：**
```bash
d:\projects\work_projects\data_service\venv\Scripts\python.exe sync_project.py
```

---

### 步骤5：同步缓存

生成项目的 lineage 缓存，用于前端展示字符到原始 PDF 的溯源信息。

```bash
cd D:\projects\work_projects\data_service

# 生成 lineage 缓存
d:\projects\work_projects\data_service\venv\Scripts\python.exe cli.py project cache-lineage proj_20260725_225531
```

**参数说明：**
- `project_id`: 目标项目ID
- `--force`: 强制重新生成缓存（默认不启用）

**输出文件：** `datahome/project/proj_20260725_225531/lineage_cache.json`
- 包含从字符ID到原始PDF坐标的完整溯源信息
- 支持前端快速查询字符所属的页面、PDF文件等信息

---

### 步骤6：前端标注

启动后端服务，在前端进行标注。

```bash
# 启动后端（推荐方式）
cd D:\projects\work_projects\data_service
d:\projects\work_projects\data_service\venv\Scripts\python.exe -m uvicorn cutting_labeling_system.backend.app:app --host 0.0.0.0 --port 8000 --reload

# 启动前端
cd D:\projects\work_projects\data_service\cutting_labeling_system\frontend
npm install
npm run dev
```

**前端访问：** http://localhost:5173

---

## 完整操作脚本

将步骤3和步骤4合并为一个完整脚本 `sync_al_project.py`：

```python
#!/usr/bin/env python3
"""
主动学习项目同步脚本

功能：
1. 从 AL 排名文件读取 line_id
2. 排除已标注的数据
3. 更新项目的 line_id_list.json 和 project.json
4. 从指定推理版本复制模型推理结果
5. 生成 lineage 缓存（可选）

使用方法：
    python sync_al_project.py --project-id proj_20260725_225531 --inference-version v0724 --with-cache
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="同步主动学习项目")
    parser.add_argument("--project-id", type=str, required=True, help="目标项目ID")
    parser.add_argument("--existing-project-id", type=str, default="proj_20260706_230417",
                        help="已标注项目ID（用于排除重复）")
    parser.add_argument("--inference-version", type=str, required=True, help="推理版本（如 v0724）")
    parser.add_argument("--al-ranking-path", type=str, default="ai_model/models/inference_al_ranking.json",
                        help="AL排名文件路径")
    parser.add_argument("--with-cache", action="store_true", default=False,
                        help="是否生成 lineage 缓存")
    args = parser.parse_args()

    # 路径设置
    project_root = Path(f"datahome/project/{args.project_id}")
    al_ranking_path = Path(args.al_ranking_path)
    existing_project_path = Path(f"datahome/project/{args.existing_project_id}")
    v0724_dir = Path(f"datahome/model_inference/{args.inference_version}")
    rule_jsons_dir = Path("datahome/rule_jsons")

    # ----------------------
    # 步骤1：读取并过滤 line_id
    # ----------------------
    print("\n" + "="*60)
    print("步骤1：读取并过滤 line_id")
    print("="*60)

    with open(al_ranking_path, 'r', encoding='utf-8') as f:
        al_ranking = json.load(f)

    all_line_ids = [item['line_id'] for item in al_ranking]
    print(f"[INFO] AL排名文件包含 {len(all_line_ids)} 个 line_id")

    # 获取已标注的 line_id
    existing_annotated = set()
    existing_annotations_dir = existing_project_path / "annotations"
    if existing_annotations_dir.exists():
        for anno_file in existing_annotations_dir.glob('*.json'):
            line_id = anno_file.stem.replace('_annotation', '')
            existing_annotated.add(line_id)
    print(f"[INFO] 已标注项目包含 {len(existing_annotated)} 个 line_id")

    # 过滤
    overlap = set(all_line_ids) & existing_annotated
    print(f"[INFO] 重叠的 line_id: {len(overlap)} 个")
    filtered_line_ids = [lid for lid in all_line_ids if lid not in overlap]
    print(f"[INFO] 过滤后剩余 {len(filtered_line_ids)} 个 line_id")

    # ----------------------
    # 步骤2：更新项目配置
    # ----------------------
    print("\n" + "="*60)
    print("步骤2：更新项目配置")
    print("="*60)

    # 更新 line_id_list.json
    line_id_list_path = project_root / "line_id_list.json"
    line_id_list_data = {
        'total_count': len(filtered_line_ids),
        'line_ids': filtered_line_ids
    }
    with open(line_id_list_path, 'w', encoding='utf-8') as f:
        json.dump(line_id_list_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已更新 line_id_list.json")

    # 更新 project.json
    project_json_path = project_root / "project.json"
    with open(project_json_path, 'r', encoding='utf-8') as f:
        project_data = json.load(f)

    project_data['line_id_count'] = len(filtered_line_ids)
    project_data['description'] = (
        f"基于推理结果主动学习筛选的标注项目，"
        f"共{len(filtered_line_ids)}条待标注数据（已排除{len(overlap)}条已标注数据）"
    )
    project_data['updated_at'] = "2026-07-26T00:00:00.000000"

    with open(project_json_path, 'w', encoding='utf-8') as f:
        json.dump(project_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已更新 project.json")

    # ----------------------
    # 步骤3：同步推理结果
    # ----------------------
    print("\n" + "="*60)
    print("步骤3：同步推理结果")
    print("="*60)

    inference_jsonl_path = v0724_dir / "inference.jsonl"
    metadata_path = v0724_dir / "metadata.json"
    model_jsons_dir = project_root / "model_jsons"

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    model_path = metadata.get('model_path', 'char_segment_1d_unet_best.pth')
    print(f"[INFO] 使用模型: {model_path}")

    # 构建推理结果映射
    inference_map = {}
    with open(inference_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                line_id = data.get('line_id')
                if line_id and line_id in set(filtered_line_ids):
                    inference_map[line_id] = data
            except Exception:
                continue

    print(f"[INFO] 从 inference.jsonl 中找到 {len(inference_map)} 个匹配的 line_id")

    # 复制推理结果
    model_jsons_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0
    fail_count = 0
    missing_count = 0

    for line_id in filtered_line_ids:
        if line_id not in inference_map:
            missing_count += 1
            continue

        inf_data = inference_map[line_id]
        intervals = inf_data.get('chars', [])

        # 获取图像尺寸
        rule_file = rule_jsons_dir / f"{line_id}_rule.json"
        img_width = 0
        img_height = 0
        if rule_file.exists():
            try:
                with open(rule_file, 'r', encoding='utf-8') as f:
                    rule_data = json.load(f)
                    img_width = rule_data.get('width', 0)
                    img_height = rule_data.get('height', 0)
            except Exception:
                pass

        # 构建字符列表
        chars = []
        for idx, (start, end) in enumerate(intervals):
            chars.append({
                'char_id': f"{line_id}_char_{idx}",
                'line_id': line_id,
                'char_idx': idx,
                'col_start': start,
                'col_end': end,
                'width': end - start
            })

        # 输出数据
        output_data = {
            'line_id': line_id,
            'chars': chars,
            'char_count': len(chars),
            'width': img_width,
            'height': img_height,
            'threshold': metadata.get('params', {}).get('threshold', 0.5),
            'model_path': model_path
        }

        # 保存文件
        output_path = model_jsons_dir / f"{line_id}_model.json"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            success_count += 1
        except Exception as e:
            print(f"[ERROR] 保存失败 {line_id}: {e}")
            fail_count += 1

    # ----------------------
    # 步骤4：输出统计
    # ----------------------
    print("\n" + "="*60)
    print("步骤4：输出统计")
    print("="*60)
    print(f"[INFO] 成功: {success_count}")
    print(f"[INFO] 失败: {fail_count}")
    print(f"[INFO] inference.jsonl 中缺失: {missing_count}")
    print(f"[INFO] 输出目录: {model_jsons_dir}")
    print(f"[INFO] 项目ID: {args.project_id}")
    print(f"[INFO] 推理版本: {args.inference_version}")

    # ----------------------
    # 步骤5：生成 lineage 缓存（可选）
    # ----------------------
    if args.with_cache:
        print("\n" + "="*60)
        print("步骤5：生成 lineage 缓存")
        print("="*60)
        
        try:
            import subprocess
            cmd = [
                str(Path(__file__).parent / "cli.py"),
                "project",
                "cache-lineage",
                args.project_id
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
            if result.returncode == 0:
                print(f"[INFO] lineage 缓存生成成功")
                print(f"[INFO] {result.stdout.strip()}")
            else:
                print(f"[ERROR] lineage 缓存生成失败")
                print(f"[ERROR] {result.stderr.strip()}")
        except Exception as e:
            print(f"[ERROR] 调用 cache-lineage 失败: {e}")


if __name__ == "__main__":
    main()
```

**使用方式：**
```bash
cd D:\projects\work_projects\data_service

# 执行同步（包含缓存）
d:\projects\work_projects\data_service\venv\Scripts\python.exe sync_al_project.py \
    --project-id proj_20260725_225531 \
    --inference-version v0724 \
    --with-cache
```

---

## 常用命令速查

### 全量推理
```bash
python cli.py project infer-all --output-version v0724
```

### 主动学习排序
```bash
python -m ai_model.train.inference_based_al --top-n 2000 --inference-version v0724
```

### 项目推理（重新生成推理结果）
```bash
python cli.py project infer proj_20260725_225531 --model-path models/char_segment_1d_unet_best_0724.pth
```

### 生成 lineage 缓存
```bash
python cli.py project cache-lineage proj_20260725_225531
```

### 启动后端服务
```bash
python -m uvicorn cutting_labeling_system.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 启动前端服务
```bash
cd cutting_labeling_system/frontend
npm run dev
```

---

## 数据流向图

```
全量推理                    主动学习排序               项目同步                前端标注
─────────────────────────────────────────────────────────────────────────────────────
rule_jsons/          ───────→  inference.jsonl  ────→  inference_al_ranking.json
    │                              │                          │
    ↓                              ↓                          ↓
    │                     规则结果 vs 模型结果            筛选难样本
    │                          (计算AL分数)                   │
    ↓                              ↓                          ↓
lines/*.png          ───────→  CharSegmentPredictor  ───→  line_id_list.json
    │                              │                          │
    ↓                              ↓                          ↓
    │                      metadata.json              project.json
    │                              │                          │
    ↓                              ↓                          ↓
    └──────────────────────────────┴──────────────→  model_jsons/*.json
                                                             │
                                                             ↓
                                                      lineage_cache.json
                                                             │
                                                             ↓
                                                      前端标注界面
                                                             │
                                                             ↓
                                                      annotations/*.json
```

---

## 注意事项

1. **模型版本一致性**：确保全量推理和项目同步使用相同的模型版本
2. **数据排除**：更新项目时会自动排除已标注的 line_id，避免重复标注
3. **推理结果格式**：`inference.jsonl` 使用轻量格式 `[[start,end],...]`，需要转换为项目所需的完整格式
4. **图像尺寸**：从 `rule_jsons` 文件中读取图像尺寸信息
5. **增量更新**：全量推理支持断点续传（不启用 `--overwrite` 时跳过已完成的 line_id）
6. **缓存同步**：更新项目 line_id 后必须重新生成 lineage 缓存，否则前端无法正确展示溯源信息

---

## 下次使用流程（快速参考）

```bash
# 1. 全量推理（使用新模型）
python cli.py project infer-all --output-version v0726

# 2. 主动学习排序
python -m ai_model.train.inference_based_al --top-n 2000 --inference-version v0726

# 3. 同步项目（更新 line_id + 同步推理结果 + 生成缓存）
python sync_al_project.py --project-id proj_20260725_225531 --inference-version v0726 --with-cache

# 4. 启动服务进行标注
python -m uvicorn cutting_labeling_system.backend.app:app --host 0.0.0.0 --port 8000 --reload
```
