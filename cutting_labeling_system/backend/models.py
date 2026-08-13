from pydantic import BaseModel
from typing import List, Optional


class Line(BaseModel):
    pos: int
    color: str


class AnnotationSubmit(BaseModel):
    lines: List[Line]


class ImageListRequest(BaseModel):
    """图片列表筛选请求"""
    is_annotated: Optional[str] = None  # "true" / "false" / "pending"
    page: int = 1
    page_size: int = 10


class ImageListItem(BaseModel):
    """图片列表项"""
    id: str
    image_name: str
    score: float = 0.0
    is_annotated: bool = False
    is_postponed: bool = False
    updated_at: str = ""


class ImageDetail(BaseModel):
    """图片详情（含三层切割线）"""
    id: str
    image_name: str
    image_url: str
    is_annotated: bool = False
    is_postponed: bool = False
    rule_lines: List[Line] = []
    model_lines: List[Line] = []
    fusion_lines: List[Line] = []
    annotation: dict = {"lines": []}


class PriorityItem(BaseModel):
    """优先级队列项"""
    image_id: str
    image_name: str = ""
    al_score: float = 0.0
    priority_score: float = 0.0
    is_annotated: bool = False
    is_postponed: bool = False
    updated_at: str = ""


class AnnotationStats(BaseModel):
    """标注统计"""
    total_samples: int = 0
    annotated_count: int = 0
    postponed_count: int = 0
    unannotated_count: int = 0
    progress: float = 0.0


class LineStatusItem(BaseModel):
    """行状态项"""
    line_name: str
    total: int = 0
    labeled: int = 0
    ratio: float = 0.0
    status: str = "unlabeled"  # full / partial / unlabeled
    chars: list = []


class LineStatusResponse(BaseModel):
    """行状态响应"""
    code: int = 0
    data: List[LineStatusItem] = []
