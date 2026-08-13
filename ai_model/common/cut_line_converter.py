"""
切割线转换工具

提供字符区间与切割线之间的转换逻辑，供后端API和可视化工具共同使用，
确保前后端显示一致。

共享边界定义：
    1. 完全共享边界：char1.col_end == char2.col_start
       该位置同时是上一个字符的结束和下一个字符的开始。
    
    2. 相邻共享边界：char1.col_end + 1 == char2.col_start
       两个相邻的边界点表示两个字符之间的共享边界。
"""

from typing import List, Tuple, Dict, Union


# ===================== 颜色常量 =====================

START_COLOR_HEX = "#ff0000"
END_COLOR_HEX = "#00ff00"
SHARED_COLOR_HEX = "#9932cc"

START_COLOR_RGB = (255, 0, 0)
END_COLOR_RGB = (0, 255, 0)
SHARED_COLOR_RGB = (153, 50, 204)


# ===================== 区间转切割线 =====================

def chars_to_lines(
    chars: List[Dict],
    color_format: str = "hex"
) -> List[Dict]:
    """
    将字符区间转换为切割线（支持共享边界）
    
    Args:
        chars: 字符列表，每个包含 col_start, col_end
        color_format: 颜色格式，"hex" 或 "rgb"
    
    Returns:
        lines: 切割线列表，每个包含 pos, color
    """
    if color_format == "hex":
        start_color = START_COLOR_HEX
        end_color = END_COLOR_HEX
        shared_color = SHARED_COLOR_HEX
    else:
        start_color = START_COLOR_RGB
        end_color = END_COLOR_RGB
        shared_color = SHARED_COLOR_RGB
    
    all_starts = set()
    all_ends = set()
    
    for char in chars:
        col_start = char.get("col_start")
        col_end = char.get("col_end")
        if col_start is not None:
            all_starts.add(col_start)
        if col_end is not None:
            all_ends.add(col_end)
    
    shared_boundaries = all_starts & all_ends
    
    adjacent_shared = set()
    for end_pos in all_ends:
        if end_pos + 1 in all_starts:
            adjacent_shared.add(end_pos)
            adjacent_shared.add(end_pos + 1)
    
    line_dict = {}
    
    for char in chars:
        col_start = char.get("col_start")
        col_end = char.get("col_end")
        
        if col_start is not None:
            if col_start in shared_boundaries or col_start in adjacent_shared:
                line_dict[col_start] = shared_color
            elif col_start not in line_dict:
                line_dict[col_start] = start_color
        
        if col_end is not None:
            if col_end in shared_boundaries or col_end in adjacent_shared:
                line_dict[col_end] = shared_color
            elif col_end not in line_dict:
                line_dict[col_end] = end_color
    
    lines = [{"pos": pos, "color": color} for pos, color in line_dict.items()]
    lines.sort(key=lambda x: x["pos"])
    return lines


def intervals_to_lines(
    intervals: List[Tuple[int, int]],
    color_format: str = "rgb"
) -> List[Dict]:
    """
    将区间元组列表转换为切割线
    
    Args:
        intervals: 区间列表，每个为 (start, end) 元组
        color_format: 颜色格式，"hex" 或 "rgb"
    
    Returns:
        lines: 切割线列表，每个包含 pos, color
    """
    chars = [{"col_start": s, "col_end": e} for s, e in intervals]
    return chars_to_lines(chars, color_format)


def intervals_to_positions(
    intervals: List[Tuple[int, int]]
) -> Tuple[List[int], List[int], List[int]]:
    """
    从区间中分离出起始位置、结束位置和共享边界位置
    
    Args:
        intervals: 区间列表，每个为 (start, end) 元组
    
    Returns:
        (start_positions, end_positions, shared_positions)
    """
    all_starts = set(s for s, e in intervals)
    all_ends = set(e for s, e in intervals)
    
    shared_boundaries = all_starts & all_ends
    
    adjacent_shared = set()
    for end_pos in all_ends:
        if end_pos + 1 in all_starts:
            adjacent_shared.add(end_pos)
            adjacent_shared.add(end_pos + 1)
    
    shared_positions = shared_boundaries | adjacent_shared
    start_positions = [s for s in all_starts if s not in shared_positions]
    end_positions = [e for e in all_ends if e not in shared_positions]
    
    return sorted(start_positions), sorted(end_positions), sorted(shared_positions)


# ===================== 切割线转区间 =====================

def lines_to_chars(lines: List[Dict], image_width: int) -> List[Dict]:
    """
    将切割线转换为字符区间
    
    转换逻辑：
        1. 遍历排序后的切割线
        2. 遇到红色或紫色线 → 开始新字符的起始位置
        3. 遇到绿色或紫色线 → 当前字符结束
        4. 支持共享边界：紫色线同时作为上一个字符的结束和下一个字符的开始
        5. 跳过宽度 <= 1 的无效区间
    
    Args:
        lines: 切割线列表，每个包含 pos, color
        image_width: 图像宽度
    
    Returns:
        chars: 字符列表，每个包含 col_start, col_end
    """
    lines_sorted = sorted(lines, key=lambda x: x["pos"])
    
    chars = []
    
    i = 0
    n = len(lines_sorted)
    
    while i < n:
        current_line = lines_sorted[i]
        current_color = current_line.get("color", "")
        
        is_start = current_color in (START_COLOR_HEX, "red")
        is_shared = current_color in (SHARED_COLOR_HEX, "purple", "#9932cc")
        
        if is_start or is_shared:
            col_start = current_line["pos"]
            
            j = i + 1
            found_end = False
            while j < n:
                next_line = lines_sorted[j]
                next_color = next_line.get("color", "")
                
                is_end = next_color in (END_COLOR_HEX, "green")
                is_next_shared = next_color in (SHARED_COLOR_HEX, "purple", "#9932cc")
                
                if is_end or is_next_shared:
                    col_end = next_line["pos"]
                    
                    if col_end - col_start > 1:
                        chars.append({"col_start": col_start, "col_end": col_end})
                    
                    if is_next_shared:
                        i = j
                    else:
                        i = j + 1
                    found_end = True
                    break
                
                j += 1
            
            if not found_end:
                i += 1
        else:
            i += 1
    
    return chars
