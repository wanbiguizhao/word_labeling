#!/usr/bin/env python3
"""
后处理链框架（Post-Processing Chain）

将规则切割后的字符处理步骤抽象为可配置的处理链节点。
每个节点统一接口：输入 chars → 输出 chars，通过 JSON 配置文件编排。

配置格式示例:
{
    "version": "v2_merge_gap3",
    "description": "规则切割后处理：合并碎片",
    "steps": [
        {
            "name": "merge_fragments",
            "enabled": true,
            "params": {
                "width_ratio": 0.5,
                "max_gap": 3,
                "strategy": "nearest"
            }
        },
        {
            "name": "filter_by_width",
            "enabled": false,
            "params": {"min_width": 3}
        }
    ]
}
"""
import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_model import merge_fragments as _merge_fragments


# ============================================================
# 步骤注册机制
# ============================================================

_STEPS: Dict[str, Callable] = {}


def register_step(name: str):
    """装饰器：注册一个后处理步骤"""
    def decorator(func: Callable):
        _STEPS[name] = func
        return func
    return decorator


def get_step(name: str) -> Optional[Callable]:
    """获取已注册的步骤"""
    return _STEPS.get(name)


def list_steps() -> List[str]:
    """列出所有已注册的步骤名"""
    return sorted(_STEPS.keys())


# ============================================================
# 内置步骤
# ============================================================

@register_step("merge_fragments")
def step_merge_fragments(chars: List[Dict], params: Dict, context: Dict) -> List[Dict]:
    """
    合并窄碎片到相邻字符

    针对规则切割在汉字内部空白处误切产生的碎片，
    按宽度比例判定并合并到左/右相邻字符。

    params:
        width_ratio: 碎片判定阈值，宽度 < 行中位宽 × ratio 视为碎片（默认0.5）
        strategy: 合并策略 'nearest' | 'larger' | 'left' | 'right'（默认'nearest'）
        max_gap: 合并最大允许间隙（像素），0表示不限制（默认0）
    """
    if len(chars) < 2:
        return chars

    width_ratio = params.get("width_ratio", 0.5)
    strategy = params.get("strategy", "nearest")
    max_gap = params.get("max_gap", 0)

    # 转换为元组列表调用核心函数
    intervals = [(c["col_start"], c["col_end"]) for c in chars]
    merged = _merge_fragments(
        intervals,
        width_ratio=width_ratio,
        strategy=strategy,
        max_gap=max_gap if max_gap > 0 else None,
    )

    # 重建 char dicts
    result = []
    for start, end in merged:
        result.append({"col_start": start, "col_end": end, "width": end - start})
    return result


@register_step("filter_by_width")
def step_filter_by_width(chars: List[Dict], params: Dict, context: Dict) -> List[Dict]:
    """
    按宽度过滤字符

    过滤掉宽度小于阈值的字符（通常是脏点、噪声）。

    params:
        min_width: 最小允许宽度（像素），小于此值的字符被过滤（默认3）
    """
    min_width = params.get("min_width", 3)
    return [c for c in chars if (c["col_end"] - c["col_start"]) >= min_width]


@register_step("noop")
def step_noop(chars: List[Dict], params: Dict, context: Dict) -> List[Dict]:
    """空操作，用于基线对比"""
    return chars


# ============================================================
# 处理链执行器
# ============================================================

def load_config(config_path: Path) -> Dict:
    """加载后处理链配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_chain(chars: List[Dict], config: Dict, context: Optional[Dict] = None, verbose: bool = True) -> List[Dict]:
    """
    按配置顺序执行后处理链

    Args:
        chars: 规则切割产生的字符列表
        config: 处理链配置 (dict 或 config_path)
        context: 上下文信息 (line_id, image_width, image_height)
        verbose: 是否打印每步处理日志

    Returns:
        处理后的字符列表
    """
    if context is None:
        context = {}

    if isinstance(config, (str, Path)):
        config = load_config(Path(config))

    steps = config.get("steps", [])
    for step_cfg in steps:
        if not step_cfg.get("enabled", True):
            continue

        step_name = step_cfg["name"]
        step_func = get_step(step_name)
        if step_func is None:
            print(f"  [WARN] 未知的后处理步骤: {step_name}，跳过")
            continue

        params = step_cfg.get("params", {})
        before_count = len(chars)
        chars = step_func(chars, params, context)
        after_count = len(chars)

        if before_count != after_count and verbose:
            step_label = f"{step_name}({params})" if params else step_name
            print(f"  [POSTCHAIN] {step_label}: {before_count} → {after_count} 字符")

    return chars


def get_default_config() -> Dict:
    """默认配置：不做任何后处理"""
    return {
        "version": "baseline",
        "description": "无后处理（基线）",
        "steps": []
    }


# ============================================================
# 配置目录
# ============================================================

CONFIGS_DIR = Path(__file__).parent / "postprocess_configs"


def resolve_config_path(config_name: str) -> Path:
    """
    根据配置名解析配置文件路径

    支持以下形式:
        - "baseline" → postprocess_configs/baseline.json
        - "merge_fragments_gap3" → postprocess_configs/merge_fragments_gap3.json
        - "/absolute/path/to/config.json" → 直接使用绝对路径
    """
    p = Path(config_name)
    if p.is_absolute() and p.exists():
        return p
    return CONFIGS_DIR / f"{config_name}.json"
