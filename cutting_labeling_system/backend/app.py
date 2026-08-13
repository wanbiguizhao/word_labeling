import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from collections import defaultdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from .config import project_manager, MERGED_ANNOTATIONS_PATH
from .annotation_registry import annotation_registry
from .project_service import project_service
from .selector_service import selector_service

from ai_model.common.cut_line_converter import (
    chars_to_lines as chars_to_lines_impl,
    lines_to_chars as lines_to_chars_impl
)

app = FastAPI(title="汉字切割标注系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.middleware("http")
async def set_encoding(request: Request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


def load_json_file(file_path: Path) -> Optional[dict]:
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def chars_to_lines(chars: List[dict], image_width: int) -> List[dict]:
    """
    将字符区间转换为切割线（支持共享边界）
    
    使用 ai_model.common.cut_line_converter 中的公共实现，
    确保前后端显示一致。
    
    Args:
        chars: 字符列表，每个包含 col_start, col_end
        image_width: 图像宽度（兼容参数，实际未使用）
    
    Returns:
        lines: 切割线列表，每个包含 pos, color
    """
    return chars_to_lines_impl(chars, color_format="hex")


def lines_to_chars(lines: List[dict], image_width: int = 0) -> List[dict]:
    """
    将切割线转换为字符区间（支持共享边界）
    
    使用 ai_model.common.cut_line_converter 中的公共实现，
    确保前后端显示一致。
    
    Args:
        lines: 切割线列表，每个包含 pos, color
        image_width: 图像宽度（兼容参数）
    
    Returns:
        chars: 字符列表，每个包含 col_start, col_end, width
    """
    chars = lines_to_chars_impl(lines, image_width)
    for char in chars:
        char["width"] = char["col_end"] - char["col_start"]
    return chars


def get_project(project_id: Optional[str] = None):
    if project_id:
        project = project_manager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"项目不存在: {project_id}")
        return project
    project = project_manager.current_project
    if not project:
        raise HTTPException(status_code=500, detail="未配置任何项目")
    return project


@app.get("/api/projects")
async def list_projects():
    projects = project_service.list_projects()
    current = project_manager.current_project
    project_list = []
    for proj in projects:
        stats = project_service.get_project_stats(proj["project_id"])
        project_list.append({
            "project_id": proj["project_id"],
            "project_name": proj["project_name"],
            "description": proj.get("description", ""),
            "line_id_count": proj.get("line_id_count", 0),
            "pdf_count": len(proj.get("pdf_files", [])),
            "selector": proj.get("selector", {}),
            **stats
        })
    return {
        "code": 0,
        "msg": "success",
        "data": {
            "projects": project_list,
            "current_project": current.project_id if current else None
        }
    }


class CreateProjectRequest(BaseModel):
    project_name: str
    strategy: str = "random"
    pdf_ids: Optional[List[str]] = None
    count: int = 100
    description: str = ""


@app.post("/api/projects")
async def create_project(body: CreateProjectRequest):
    if body.strategy == "pdf" and (not body.pdf_ids or len(body.pdf_ids) == 0):
        raise HTTPException(status_code=400, detail="PDF选择器需要指定pdf_ids")

    config = project_service.create_project(
        project_name=body.project_name,
        strategy=body.strategy,
        pdf_ids=body.pdf_ids,
        count=body.count,
        description=body.description
    )

    project_manager._load_projects()
    project_manager.set_current_project(config["project_id"])

    return {
        "code": 0,
        "msg": "项目创建成功",
        "data": config
    }


@app.get("/api/projects/{project_id}")
async def get_project_detail(project_id: str):
    config = project_service.get_project_config(project_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"项目不存在: {project_id}")
    stats = project_service.get_project_stats(project_id)
    return {
        "code": 0,
        "msg": "success",
        "data": {**config, **stats}
    }


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    success = project_service.delete_project(project_id)
    if success:
        project_manager._load_projects()
        return {"code": 0, "msg": "项目删除成功"}
    raise HTTPException(status_code=404, detail=f"项目不存在: {project_id}")


@app.post("/api/projects/{project_id}/switch")
async def switch_project(project_id: str):
    success = project_manager.set_current_project(project_id)
    if success:
        return {"code": 0, "msg": "切换成功"}
    raise HTTPException(status_code=404, detail=f"项目不存在: {project_id}")


@app.get("/api/selectors/strategies")
async def list_selector_strategies():
    return {
        "code": 0,
        "msg": "success",
        "data": [
            {"strategy": "pdf", "name": "按PDF文件选择", "description": "根据指定的PDF文件ID选择待标注行"},
            {"strategy": "random", "name": "随机选择", "description": "从所有未标注数据中随机选择"},
            {"strategy": "al", "name": "主动学习选择", "description": "根据主动学习分数排序选择最有价值的数据"}
        ]
    }


class SelectorPreviewRequest(BaseModel):
    strategy: str = "random"
    pdf_ids: Optional[List[str]] = None
    count: int = 20


@app.post("/api/selectors/preview")
async def preview_selector(body: SelectorPreviewRequest):
    result = selector_service.preview(body.strategy, body.pdf_ids, body.count)
    return {
        "code": 0,
        "msg": "success",
        "data": result
    }


@app.get("/api/annotation/registry-stats")
async def get_registry_stats():
    stats = annotation_registry.get_stats()
    return {
        "code": 0,
        "msg": "success",
        "data": stats
    }


class LineDataRequest(BaseModel):
    line_id: str
    project_id: Optional[str] = None


class ImageListRequest(BaseModel):
    source: Optional[str] = None
    page: int = 1
    page_size: int = 20
    sort_by: Optional[str] = None
    sort_order: str = "desc"
    is_annotated: Optional[bool] = None
    project_id: Optional[str] = None


_image_cache = {}
_cache_timestamp = {}


def _load_image_cache(project):
    project_id = project.project_id
    now = datetime.now().timestamp()
    if project_id in _image_cache and (now - _cache_timestamp.get(project_id, 0)) < 300:
        return _image_cache[project_id]

    rule_dir = project.rule_jsons_dir
    if not rule_dir.exists():
        _image_cache[project_id] = []
        _cache_timestamp[project_id] = now
        return []

    items = []

    # 优化路径：当项目有 line_id_list 时，直接按 line_id 定位文件，
    # 避免扫描共享目录中的全部 37649 个 rule.json 文件
    if project.line_id_list_file:
        for line_id in project.line_ids:
            rule_file = rule_dir / f"{line_id}_rule.json"
            if not rule_file.exists():
                continue
            rule_data = load_json_file(rule_file)
            if not rule_data:
                continue
            image_path = rule_data.get("image_path", "")
            chars = rule_data.get("chars", [])
            items.append({
                "id": line_id,
                "line_id": line_id,
                "image_path": image_path,
                "char_count": len(chars),
                "_anno_file": str(project.annotations_dir / f"{line_id}.json")
            })
    else:
        # 回退路径：无 line_id_list 时扫描全部文件
        for rule_file in rule_dir.glob("*_rule.json"):
            rule_data = load_json_file(rule_file)
            if not rule_data:
                continue
            line_id = rule_data.get("line_id", "")
            image_path = rule_data.get("image_path", "")
            chars = rule_data.get("chars", [])
            items.append({
                "id": line_id,
                "line_id": line_id,
                "image_path": image_path,
                "char_count": len(chars),
                "_anno_file": str(project.annotations_dir / f"{line_id}.json")
            })

    _image_cache[project_id] = items
    _cache_timestamp[project_id] = now
    return items


@app.post("/api/images")
async def get_images(body: ImageListRequest):
    project = get_project(body.project_id)

    all_items = _load_image_cache(project)

    filtered_items = []
    for item in all_items:
        is_annotated = annotation_registry.is_annotated(item["line_id"])

        is_postponed = annotation_registry.is_postponed(item["line_id"])

        if body.is_annotated is not None:
            if body.is_annotated and not is_annotated:
                continue
            if not body.is_annotated and (is_annotated or is_postponed):
                continue

        filtered_items.append({
            "id": item["id"],
            "line_id": item["line_id"],
            "image_path": item["image_path"],
            "char_count": item["char_count"],
            "is_annotated": is_annotated,
            "is_postponed": is_postponed
        })

    total = len(filtered_items)
    start = (body.page - 1) * body.page_size
    end = start + body.page_size
    paginated_items = filtered_items[start:end]

    return {
        "code": 0,
        "msg": "success",
        "data": paginated_items,
        "total": total,
        "page": body.page,
        "page_size": body.page_size,
        "total_pages": (total + body.page_size - 1) // body.page_size
    }


# ===================== lineage 缓存机制 =====================
# 全局内存缓存：首次加载 lineage.json 后常驻内存，避免重复读取 1.3GB 文件
_global_lineage_cache: Optional[dict] = None
_global_lineage_loaded: bool = False


def _load_global_lineage(project) -> dict:
    """加载全局 lineage.json 到内存缓存（仅首次调用时实际读取文件）"""
    global _global_lineage_cache, _global_lineage_loaded
    if _global_lineage_loaded:
        return _global_lineage_cache or {}

    # 尝试两个可能的路径
    for lineage_path in [
        project.project_root.parent.parent / "lineage.json",
        project.project_root.parent / "lineage.json",
    ]:
        if lineage_path.exists():
            try:
                logger.info(f"正在加载 lineage.json: {lineage_path} (可能需要数秒)...")
                with open(lineage_path, "r", encoding="utf-8") as f:
                    _global_lineage_cache = json.load(f)
                _global_lineage_loaded = True
                logger.info(f"lineage.json 加载完成: {len(_global_lineage_cache.get('lines', {}))} 行")
                return _global_lineage_cache
            except Exception as e:
                logger.error(f"加载 lineage.json 失败: {e}")
                break

    _global_lineage_loaded = True
    _global_lineage_cache = {}
    return {}


def _get_project_lineage_cache_path(project) -> Path:
    """项目级 lineage 缓存文件路径"""
    return project.project_root / "lineage_cache.json"


def _load_project_lineage_cache(project) -> dict:
    """加载项目级 lineage 缓存"""
    cache_path = _get_project_lineage_cache_path(project)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"lines": {}, "chars": {}}


def _save_project_lineage_cache(project, cache_data: dict) -> None:
    """保存项目级 lineage 缓存"""
    cache_path = _get_project_lineage_cache_path(project)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False)
    except Exception as e:
        logger.error(f"保存项目级 lineage 缓存失败: {e}")


def _try_line_id_variants(line_id: str) -> list:
    """生成 line_id 的格式变体（处理 pdf_ 前缀差异）"""
    variants = [line_id]
    if not line_id.startswith("line_page_pdf_"):
        variants.append(line_id.replace("line_page_", "line_page_pdf_", 1))
    elif line_id.startswith("line_page_pdf_"):
        variants.append(line_id.replace("line_page_pdf_", "line_page_", 1))
    return variants


def get_lineage_chars_with_cache(project, line_id: str) -> list:
    """
    带缓存地获取行的合并后字符数据

    查询流程：
    1. 优先从项目级缓存读取 → 命中则直接返回
    2. 从全局内存缓存（首次加载全局 lineage.json）读取
    3. 将该行数据写入项目级缓存，加速下次查询
    """
    # 步骤1：尝试项目级缓存
    proj_cache = _load_project_lineage_cache(project)
    for variant_id in _try_line_id_variants(line_id):
        if variant_id in proj_cache.get("lines", {}):
            line_info = proj_cache["lines"][variant_id]
            char_ids = line_info.get("chars", [])
            result = []
            for cid in char_ids:
                char_info = proj_cache.get("chars", {}).get(cid)
                if char_info:
                    result.append({
                        "col_start": char_info.get("col_start", 0),
                        "col_end": char_info.get("col_end", 0),
                        "width": char_info.get("width", 0),
                        "height": char_info.get("height", 0)
                    })
            return result

    # 步骤2：从全局内存缓存读取
    global_lineage = _load_global_lineage(project)
    if not global_lineage:
        return []

    global_lines = global_lineage.get("lines", {})
    global_chars = global_lineage.get("chars", {})

    matched_line_id = None
    line_info = None
    for variant_id in _try_line_id_variants(line_id):
        if variant_id in global_lines:
            matched_line_id = variant_id
            line_info = global_lines[variant_id]
            break

    if not line_info:
        return []

    # 提取字符数据
    char_ids = line_info.get("chars", [])
    result = []
    for cid in char_ids:
        char_info = global_chars.get(cid)
        if char_info:
            result.append({
                "col_start": char_info.get("col_start", 0),
                "col_end": char_info.get("col_end", 0),
                "width": char_info.get("width", 0),
                "height": char_info.get("height", 0)
            })

    # 步骤3：写入项目级缓存
    proj_cache.setdefault("lines", {})[matched_line_id] = line_info
    for cid in char_ids:
        if cid in global_chars:
            proj_cache.setdefault("chars", {})[cid] = global_chars[cid]
    _save_project_lineage_cache(project, proj_cache)

    return result


@app.get("/api/images/{line_id:path}/detail")
async def get_detail(line_id: str, project_id: Optional[str] = None):
    project = get_project(project_id)

    if not project.has_line_id(line_id):
        raise HTTPException(status_code=404, detail=f"line_id 不属于当前项目: {line_id}")

    rule_file = project.rule_jsons_dir / f"{line_id}_rule.json"
    model_file = project.model_jsons_dir / f"{line_id}_model.json"
    fusion_file = project.fusion_jsons_dir / f"{line_id}_fusion.json"
    anno_file = project.annotations_dir / f"{line_id}.json"

    rule_data = load_json_file(rule_file)
    model_data = load_json_file(model_file)
    fusion_data = load_json_file(fusion_file)
    anno_data = load_json_file(anno_file)

    if not rule_data:
        raise HTTPException(status_code=404, detail="规则切割数据不存在")

    image_path = rule_data.get("image_path", "")
    chars = rule_data.get("chars", [])

    if image_path.startswith("lines/") or image_path.startswith("lines\\"):
        image_path = image_path[6:]

    img_width = 0
    for ext in [".png", ".jpg", ".jpeg"]:
        img_file = project.lines_dir / image_path
        if not img_file.exists():
            img_file = project.project_root / image_path
        if img_file.exists():
            from PIL import Image
            with Image.open(img_file) as img:
                img_width = img.width
            break

    lineage_chars = get_lineage_chars_with_cache(project, line_id)

    if lineage_chars:
        rule_lines = chars_to_lines(lineage_chars, img_width)
    else:
        rule_lines = chars_to_lines(rule_data.get("chars", []), img_width) if rule_data else []
    
    model_lines = chars_to_lines(model_data.get("chars", []), img_width) if model_data else []
    fusion_lines = chars_to_lines(fusion_data.get("chars", []), img_width) if fusion_data else []

    annotation_lines = []
    is_annotated = annotation_registry.is_annotated(line_id)
    is_postponed = annotation_registry.is_postponed(line_id)

    if anno_data:
        annotation_lines = chars_to_lines(anno_data.get("chars", []), img_width)

    chars_with_id = []
    for idx, char in enumerate(chars):
        chars_with_id.append({
            "id": f"{line_id}_char_{idx}",
            "col_start": char.get("col_start", 0),
            "col_end": char.get("col_end", 0),
            "width": char.get("width", 0),
            "label": char.get("label", "")
        })

    return {
        "code": 0,
        "msg": "success",
        "data": {
            "id": line_id,
            "line_id": line_id,
            "image_path": image_path,
            "image_url": f"/api/images/{line_id}/raw?project_id={project.project_id}",
            "char_count": len(chars),
            "rule_lines": rule_lines,
            "model_lines": model_lines,
            "fusion_lines": fusion_lines,
            "annotation": {
                "lines": annotation_lines,
                "chars": chars_with_id
            },
            "is_annotated": is_annotated,
            "is_postponed": is_postponed
        }
    }


@app.get("/api/images/{line_id:path}/raw")
async def get_raw(line_id: str, project_id: Optional[str] = None):
    project = get_project(project_id)

    rule_file = project.rule_jsons_dir / f"{line_id}_rule.json"
    rule_data = load_json_file(rule_file)

    if not rule_data:
        raise HTTPException(status_code=404, detail="规则切割数据不存在")

    image_path = rule_data.get("image_path", "")

    if image_path.startswith("lines/") or image_path.startswith("lines\\"):
        image_path = image_path[6:]

    img_file = project.lines_dir / image_path
    if not img_file.exists():
        img_file = project.project_root / image_path

    if not img_file.exists():
        raise HTTPException(status_code=404, detail="图片文件不存在")

    with open(img_file, "rb") as f:
        content = f.read()

    response = Response(content)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Content-Type"] = "image/png"
    return response


class AnnotateRequest(BaseModel):
    lines: List[dict]
    project_id: Optional[str] = None


@app.post("/api/images/{line_id:path}/annotate")
async def annotate(line_id: str, body: AnnotateRequest):
    project = get_project(body.project_id)

    chars = lines_to_chars(body.lines)

    anno_data = {
        "image_name": f"{line_id}.png",
        "line_id": line_id,
        "is_annotated": True,
        "is_postponed": False,
        "updated_at": datetime.now().isoformat(),
        "chars": chars
    }

    project.annotations_dir.mkdir(parents=True, exist_ok=True)
    anno_file = project.annotations_dir / f"{line_id}.json"

    with open(anno_file, "w", encoding="utf-8") as f:
        json.dump(anno_data, f, ensure_ascii=False, indent=2)

    annotation_registry.mark_annotated(line_id, project.project_id)

    return {"code": 0, "msg": "保存成功"}


@app.post("/api/images/{line_id:path}/postpone")
async def postpone(line_id: str, project_id: Optional[str] = None):
    project = get_project(project_id)

    anno_file = project.annotations_dir / f"{line_id}.json"
    anno_data = load_json_file(anno_file) or {}

    anno_data["is_postponed"] = True
    anno_data["is_annotated"] = False
    anno_data["updated_at"] = datetime.now().isoformat()

    project.annotations_dir.mkdir(parents=True, exist_ok=True)

    with open(anno_file, "w", encoding="utf-8") as f:
        json.dump(anno_data, f, ensure_ascii=False, indent=2)

    annotation_registry.mark_postponed(line_id, project.project_id)

    return {"code": 0, "msg": "已标记为暂不标注"}


@app.post("/api/images/{line_id:path}/unpostpone")
async def unpostpone(line_id: str, project_id: Optional[str] = None):
    project = get_project(project_id)

    anno_file = project.annotations_dir / f"{line_id}.json"
    anno_data = load_json_file(anno_file) or {}

    anno_data["is_postponed"] = False
    anno_data["updated_at"] = datetime.now().isoformat()

    project.annotations_dir.mkdir(parents=True, exist_ok=True)

    with open(anno_file, "w", encoding="utf-8") as f:
        json.dump(anno_data, f, ensure_ascii=False, indent=2)

    annotation_registry.mark_unpostponed(line_id)

    return {"code": 0, "msg": "已取消暂不标注"}


@app.get("/api/annotation/stats")
async def get_annotation_stats(project_id: Optional[str] = None):
    project = get_project(project_id)

    all_items = _load_image_cache(project)

    total_samples = len(all_items)
    annotated_count = 0
    postponed_count = 0

    for item in all_items:
        line_id = item["line_id"]
        if annotation_registry.is_annotated(line_id):
            annotated_count += 1
        elif annotation_registry.is_postponed(line_id):
            postponed_count += 1

    unannotated_count = total_samples - annotated_count - postponed_count

    return {
        "code": 0,
        "msg": "success",
        "data": {
            "total_samples": total_samples,
            "annotated_count": annotated_count,
            "postponed_count": postponed_count,
            "unannotated_count": unannotated_count
        }
    }


@app.get("/api/annotation/priority-queue")
async def get_priority_queue(
    limit: int = 20,
    skip_annotated: bool = True,
    project_id: Optional[str] = None
):
    project = get_project(project_id)

    all_items = _load_image_cache(project)

    items = []

    for item in all_items:
        line_id = item["line_id"]

        if skip_annotated:
            if annotation_registry.is_annotated(line_id) or annotation_registry.is_postponed(line_id):
                continue

        items.append({
            "line_id": line_id,
            "char_count": item["char_count"],
            "priority_score": 0.0
        })

    items.sort(key=lambda x: x["char_count"], reverse=True)
    items = items[:limit]

    return {
        "code": 0,
        "msg": "success",
        "data": items
    }


@app.get("/api/annotation/next")
async def get_next_annotation(project_id: Optional[str] = None):
    project = get_project(project_id)

    all_items = _load_image_cache(project)

    for item in all_items:
        line_id = item["line_id"]

        if annotation_registry.is_annotated(line_id) or annotation_registry.is_postponed(line_id):
            continue

        return {
            "code": 0,
            "msg": "success",
            "data": {
                "line_id": line_id,
                "image_url": f"/api/images/{line_id}/raw?project_id={project.project_id}"
            }
        }

    return {"code": 0, "msg": "没有更多待标注样本", "data": None}