# 汉字切割标注系统 - UML 4+1 视图模型

## 版本记录

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| v1.0 | 2026-07-10 | System | 基于当前系统实际架构编写，反映 lineage.json 后处理数据接入、五区域标注页面等最新改动 |

---

## 一、文档说明

### 1.1 文档目的

本文档使用 **UML 4+1 视图模型**（Kruchten 4+1 View Model）描述汉字切割标注系统的架构，从五个互补的视角全面刻画系统结构：

- **逻辑视图（Logical View）**：系统提供的功能与领域抽象
- **进程视图（Process View）**：运行期进程、并发与同步
- **开发视图（Development View）**：代码组织与模块划分
- **物理视图（Physical View）**：部署节点与网络拓扑
- **场景视图（Scenarios，+1）**：关键用例串联各视图

### 1.2 4+1 视图与系统映射总览

```
┌─────────────────────────────────────────────────────────────────┐
│                     4+1 视图模型总览                            │
├──────────────┬──────────────────────────────────────────────────┤
│ 场景视图     │  标注员标注一行汉字 / 创建项目并选择样本         │
│  (+1)        │  → 驱动并串联下面四个视图                        │
├──────────────┼──────────────────────────────────────────────────┤
│ 逻辑视图     │  ProjectConfig / ProjectManager / ProjectService │
│              │  SelectorService / AnnotationRegistry / API 层   │
│              │  LineCanvas / LabelPage / HomePage               │
├──────────────┼──────────────────────────────────────────────────┤
│ 进程视图     │  Vite Dev Server(5173) / FastAPI(8000) / 文件系统│
│              │  单进程多请求 / 内存缓存 / 单例注册表            │
├──────────────┼──────────────────────────────────────────────────┤
│ 开发视图     │  cutting_labeling_system/                        │
│              │   ├── backend/  (FastAPI + 单例服务)             │
│              │   ├── frontend/ (Vue 3 SPA)                      │
│              │   └── docs/                                     │
├──────────────┼──────────────────────────────────────────────────┤
│ 物理视图     │  开发者主机 (Windows)                            │
│              │   ├── 浏览器 → Vite(5173) → FastAPI(8000)        │
│              │   └── datahome/ (共享数据池)                     │
└──────────────┴──────────────────────────────────────────────────┘
```

---

## 二、逻辑视图（Logical View）

### 2.1 逻辑架构分层

系统采用分层架构，自上而下分为四层：

```
┌───────────────────────────────────────────────────────────┐
│ 表现层 (Presentation Layer)                               │
│   HomePage / SegmentLabelPage / LabelPage / LineCanvas    │
│   职责：用户交互、视图渲染、画布操作                       │
├───────────────────────────────────────────────────────────┤
│ 接口层 (API Layer)                                        │
│   app.py: FastAPI 路由                                    │
│   职责：请求路由、参数校验、JSON 响应、图片流响应          │
├───────────────────────────────────────────────────────────┤
│ 服务层 (Service Layer)                                    │
│   ProjectManager / ProjectService / SelectorService       │
│   AnnotationRegistry                                      │
│   职责：项目管理、样本选择、标注状态跟踪                   │
├───────────────────────────────────────────────────────────┤
│ 数据层 (Data Layer)                                       │
│   datahome/ 共享数据池                                    │
│   lineage.json / rule_jsons/ / lines/ / annotations/      │
│   职责：持久化存储与读取                                   │
└───────────────────────────────────────────────────────────┘
```

### 2.2 核心类图

#### 2.2.1 后端核心类图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           后端核心类图                                  │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐         ┌──────────────────────────┐
  │   <<singleton>>      │ 1    *  │      ProjectConfig       │
  │  ProjectManager      │────────<│  - project_id: str       │
  │──────────────────────│         │  - project_name: str     │
  │ - _projects: Dict    │         │  - paths: dict           │
  │ - _current_project   │         │  - _project_root: Path   │
  │──────────────────────│         │──────────────────────────│
  │ + get_project(id)    │         │ + rule_jsons_dir: Path   │
  │ + set_current(id)    │         │ + model_jsons_dir: Path  │
  │ + current_project    │         │ + fusion_jsons_dir: Path │
  │ + list_projects()    │         │ + annotations_dir: Path  │
  └──────────────────────┘         │ + lines_dir: Path        │
          │                        │ + line_ids: List[str]    │
          │ manages                 │ + line_id_set: Set[str]  │
          ▼                        │ + has_line_id(id): bool  │
  ┌──────────────────────┐         └──────────────────────────┘
  │   ProjectService     │                  │
  │──────────────────────│                  │ reads
  │ - _projects_base_dir │                  ▼
  │──────────────────────│         ┌──────────────────────────┐
  │ + create_project()   │         │      File System         │
  │ + delete_project()   │         │  (datahome/)             │
  │ + get_project_config│          │  - project/<id>/         │
  │ + list_projects()    │         │    - project.json        │
  │ + get_project_stats()│         │    - line_id_list.json   │
  └──────────┬───────────┘         │    - annotations/        │
             │ uses                  │    - model_jsons/        │
             ▼                      │    - fusion_jsons/       │
  ┌──────────────────────┐         │  - rule_jsons/ (共享)    │
  │  SelectorService     │         │  - lines/ (共享)         │
  │──────────────────────│         │  - lineage.json (共享)   │
  │ - _global_rule_dir   │         │  - annotation_registry   │
  │ - _al_ranking_path   │         │    .json                 │
  │──────────────────────│         └──────────────────────────┘
  │ + select(strategy)   │
  │ + select_by_pdf()    │                  ▲
  │ + select_random()    │                  │ queries
  │ + select_by_al()     │         ┌────────┴─────────────────┐
  │ + preview()          │         │  <<singleton>>           │
  │ + get_all_line_ids() │─────────│  AnnotationRegistry      │
  └──────────────────────┘         │──────────────────────────│
                                   │ - _data: Dict            │
                                   │ - _file_path: Path       │
                                   │──────────────────────────│
                                   │ + mark_annotated()       │
                                   │ + mark_postponed()       │
                                   │ + mark_unpostponed()     │
                                   │ + is_annotated(id): bool │
                                   │ + is_postponed(id): bool │
                                   │ + get_stats(): dict      │
                                   └──────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │                    app.py (FastAPI 应用)                    │
  │─────────────────────────────────────────────────────────────│
  │  工具函数:                                                   │
  │  + load_json_file(path) → Optional[dict]                    │
  │  + chars_to_lines(chars, img_width) → List[dict]            │
  │  + lines_to_chars(lines) → List[dict]                       │
  │  + load_lineage_data(project) → dict                        │
  │  + get_chars_from_lineage(lineage, line_id) → list          │
  │  + _load_image_cache(project) → list                        │
  │                                                              │
  │  API 路由:                                                   │
  │  GET  /api/projects              项目列表                    │
  │  POST /api/projects              创建项目                    │
  │  GET  /api/projects/{id}         项目详情                    │
  │  DELETE /api/projects/{id}       删除项目                    │
  │  POST /api/projects/{id}/switch  切换项目                    │
  │  GET  /api/selectors/strategies  选择策略列表                │
  │  POST /api/selectors/preview     预览选择结果                │
  │  POST /api/images                图片列表(分页)              │
  │  GET  /api/images/{id}/detail    行详情(含切割线)            │
  │  GET  /api/images/{id}/raw       原始图片流                  │
  │  POST /api/images/{id}/annotate  保存标注                    │
  │  POST /api/images/{id}/postpone  暂不标注                    │
  │  POST /api/images/{id}/unpostpone 取消暂不标注               │
  │  GET  /api/annotation/stats      标注统计                    │
  │  GET  /api/annotation/registry-stats 注册表统计              │
  │  GET  /api/annotation/priority-queue 优先队列                │
  │  GET  /api/annotation/next       下一个待标注                │
  └─────────────────────────────────────────────────────────────┘
```

#### 2.2.2 前端核心类/组件图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          前端核心组件图                             │
└─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐      ┌──────────────────────┐
  │   <<router>>    │      │     App.vue          │
  │  Vue Router     │─────>│  (根组件/路由出口)    │
  │─────────────────│      └──────────┬───────────┘
  │ / → HomePage    │                 │
  │ /segment        │                 ▼
  │ /label/:id      │      ┌──────────────────────┐
  └─────────────────┘      │     HomePage         │
                           │──────────────────────│
                           │ 项目列表 + 创建项目  │
                           │ 选择策略 + 预览      │
                           │ 标注统计概览         │
                           └──────────┬───────────┘
                                      │ 跳转 /label/:id
                                      ▼
                           ┌──────────────────────┐
                           │    LabelPage         │
                           │──────────────────────│
                           │ 五区域布局:          │
                           │  1. 原始行图片       │
                           │  2. 规则切割预览     │
                           │  3. 模型切割预览     │
                           │  4. 融合切割预览     │
                           │  5. 可编辑切割线     │
                           │ 操作按钮:            │
                           │  复制/使用/删除/     │
                           │  清空/保存/暂不标注  │
                           └──────────┬───────────┘
                                      │ 使用 4 次
                          ┌───────────┴───────────┐
                          ▼                       ▼
              ┌──────────────────────┐  ┌──────────────────────┐
              │     LineCanvas       │  │   <<static>>         │
              │──────────────────────│  │   api/index.js       │
              │ Props:               │  │──────────────────────│
              │  - imageUrl          │  │ projectsApi          │
              │  - lines[]           │  │ selectorsApi         │
              │  - selectedIndexes[] │  │ imagesApi            │
              │  - readonly          │  │ annotationApi        │
              │ Emits:               │  └──────────────────────┘
              │  - update:lines      │
              │  - update:selected-  │
              │    Indexes           │
              │ 交互:                │
              │  单击选择(8px)       │
              │  Ctrl+拖拽框选       │
              │  Delete删除          │
              │  方向键微调          │
              │  空白点击添加        │
              └──────────────────────┘
```

### 2.3 关键数据结构

```
┌─────────────────────────────────────────────────────────────────┐
│                    核心数据结构关系图                            │
└─────────────────────────────────────────────────────────────────┘

  project.json (项目配置)
  ┌─────────────────────────────────┐
  │ project_id: str                 │
  │ project_name: str               │
  │ line_id_list: "line_id_list.json"│
  │ selector: {strategy, pdf_ids}   │
  │ paths: {                         │
  │   rule_jsons: "../../rule_jsons",│───> datahome/rule_jsons/<line_id>_rule.json
  │   model_jsons: "model_jsons",   │───> project/<id>/model_jsons/<line_id>_model.json
  │   fusion_jsons: "fusion_jsons", │───> project/<id>/fusion_jsons/<line_id>_fusion.json
  │   annotations: "annotations",   │───> project/<id>/annotations/<line_id>.json
  │   lines: "../../lines"          │───> datahome/lines/<line_id>.png
  │ }                                │
  └─────────────────────────────────┘

  lineage.json (全局后处理数据)
  ┌─────────────────────────────────┐
  │ metadata: {...}                 │
  │ pdfs: {pdf_id: {...}}           │
  │ pages: {page_id: {...}}         │
  │ lines: {line_id: {              │
  │   chars: [char_id, ...],        │───> chars 中的键
  │   image_path, width, height...  │
  │ }}                              │
  │ chars: {char_id: {              │
  │   col_start, col_end, width,    │
  │   height, merged_from[], ...    │
  │ }}                              │
  └─────────────────────────────────┘

  annotation_registry.json (全局标注状态)
  ┌─────────────────────────────────┐
  │ annotations: {                  │
  │   line_id: {                    │
  │     annotated: bool,            │
  │     postponed: bool,            │
  │     project_id: str,            │
  │     annotated_at: str           │
  │   }                             │
  │ }                               │
  └─────────────────────────────────┘

  切割线对象 (前后端传递)
  ┌─────────────────────────────────┐
  │ { pos: int, color: "red"|"green"|"#ff0000" } │
  │  - red:   字符起始位置           │
  │  - green: 字符结束位置           │
  └─────────────────────────────────┘
```

---

## 三、进程视图（Process View）

### 3.1 运行时进程拓扑

```
┌─────────────────────────────────────────────────────────────────────┐
│                       开发者主机 (Windows)                          │
│                                                                     │
│  ┌───────────────┐        ┌─────────────────┐                      │
│  │   浏览器进程   │        │  Vite Dev Server │                      │
│  │  (用户操作)    │        │   (Node.js)      │                      │
│  │───────────────│        │   端口: 5173     │                      │
│  │ Vue 3 SPA     │◀──────>│  反向代理 /api   │                      │
│  │ - 渲染 DOM    │  HTTP  │  → localhost:8000│                      │
│  │ - Canvas 绘制 │        └────────┬─────────┘                      │
│  │ - 键鼠事件    │                 │                                 │
│  └───────────────┘                 │ HTTP                           │
│                                     ▼                                │
│                          ┌─────────────────────┐                    │
│                          │  FastAPI 服务进程    │                    │
│                          │  (Python venv)      │                    │
│                          │  端口: 8000         │                    │
│                          │─────────────────────│                    │
│                          │ 单进程 + asyncio    │                    │
│                          │ uvicorn 事件循环    │                    │
│                          │                     │                    │
│                          │ 内存中常驻:         │                    │
│                          │  - ProjectManager   │                    │
│                          │  - AnnotationReg.   │                    │
│                          │  - ProjectService   │                    │
│                          │  - SelectorService  │                    │
│                          │  - _image_cache{}   │                    │
│                          └──────────┬──────────┘                    │
│                                     │                               │
│                          ┌──────────▼──────────┐                    │
│                          │    文件系统 I/O     │                    │
│                          │  (同步阻塞调用)     │                    │
│                          │  datahome/ 目录树   │                    │
│                          └─────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 并发与同步模型

```
┌─────────────────────────────────────────────────────────────────┐
│                    并发模型与共享状态                            │
└─────────────────────────────────────────────────────────────────┘

  FastAPI 进程内
  ┌───────────────────────────────────────────────────────────┐
  │                  asyncio 事件循环                         │
  │                                                           │
  │   请求1: GET /detail ──┐                                  │
  │   请求2: POST /annotate├──> 协程并发调度                  │
  │   请求3: GET /raw     ─┘    (I/O 等待时让出)             │
  │                                                           │
  │   共享单例 (进程级共享, 非线程安全):                      │
  │   ┌─────────────────────────────────────────────┐         │
  │   │ ProjectManager._projects    (读多写少)      │         │
  │   │ AnnotationRegistry._data    (读写频繁)      │         │
  │   │ _image_cache{} (5分钟TTL)   (读多写少)      │         │
  │   └─────────────────────────────────────────────┘         │
  │                                                           │
  │   同步点:                                                 │
  │   - AnnotationRegistry._save(): 每次标注都同步写文件      │
  │   - _image_cache: 5分钟过期重建, 无锁                     │
  │   - ProjectManager._load_projects(): 重新加载项目列表     │
  └───────────────────────────────────────────────────────────┘

  潜在并发风险:
  1. AnnotationRegistry 并发写入可能丢失 (单进程下风险较低)
  2. _image_cache 并发重建可能重复加载 (可接受, 幂等)
  3. 文件 I/O 阻塞事件循环 (大图片读取时影响吞吐)
```

### 3.3 关键时序：标注详情加载

```
  标注员           Vue前端        Vite代理       FastAPI        文件系统
    │                │              │             │               │
    │ 访问 /label/:id│              │             │               │
    │───────────────>│              │             │               │
    │                │ GET /detail  │             │               │
    │                │─────────────>│ GET /detail │               │
    │                │              │────────────>│               │
    │                │              │             │ load_json_file│
    │                │              │             │ (rule_json)   │
    │                │              │             │──────────────>│
    │                │              │             │<──────────────│
    │                │              │             │               │
    │                │              │             │ load_lineage_ │
    │                │              │             │ data(project) │
    │                │              │             │──────────────>│
    │                │              │             │<──────────────│
    │                │              │             │               │
    │                │              │             │ get_chars_from│
    │                │              │             │ _lineage()    │
    │                │              │             │ (内存解析)    │
    │                │              │             │               │
    │                │              │             │ chars_to_lines│
    │                │              │             │ (转切割线)    │
    │                │              │             │               │
    │                │              │             │ 读图片宽度    │
    │                │              │             │ (PIL.Image)   │
    │                │              │             │──────────────>│
    │                │              │             │<──────────────│
    │                │              │<────────────│ JSON响应      │
    │                │<─────────────│             │               │
    │  渲染五区域    │              │             │               │
    │<───────────────│              │             │               │
    │                │              │             │               │
    │ (后续异步)     │ GET /raw     │             │               │
    │                │─────────────>│────────────>│ 读图片文件    │
    │                │              │             │──────────────>│
    │                │              │             │<──────────────│
    │                │              │<────────────│ image/png     │
    │  Canvas绘制    │<─────────────│             │               │
    │<───────────────│              │             │               │
```

---

## 四、开发视图（Development View）

### 4.1 源代码目录结构

```
cutting_labeling_system/
│
├── backend/                          # 后端服务 (Python + FastAPI)
│   ├── app.py                        # 主应用: 路由定义、工具函数、lineage 加载
│   ├── config.py                     # 项目配置: ProjectConfig + ProjectManager 单例
│   ├── project_service.py            # 项目服务: 创建/删除/统计/模板复制
│   ├── selector_service.py           # 选择器服务: random/pdf/al 三种策略
│   ├── annotation_registry.py        # 标注注册表: 全局标注状态跟踪 (单例)
│   ├── models.py                     # Pydantic 数据模型
│   ├── utils.py                      # 通用工具函数
│   ├── config.json                   # 后端运行时配置
│   ├── requirements.txt              # Python 依赖
│   ├── start_server.py               # uvicorn 启动入口
│   └── __pycache__/                  # 字节码缓存
│
├── frontend/                         # 前端应用 (Vue 3 + Vite)
│   ├── src/
│   │   ├── main.ts                   # 应用入口
│   │   ├── App.vue                   # 根组件 (路由出口)
│   │   ├── style.css                 # 全局样式
│   │   ├── api/
│   │   │   └── index.js              # Axios 封装 + 四组 API (projects/selectors/images/annotation)
│   │   ├── router/
│   │   │   └── index.js              # Vue Router 路由配置 (3条路由)
│   │   ├── views/
│   │   │   ├── HomePage.vue          # 首页: 项目管理 + 创建 + 统计
│   │   │   ├── SegmentLabelPage.vue  # 分割标注页 (辅助页面)
│   │   │   ├── LabelPage.vue         # 标注详情页: 五区域布局 + 操作按钮
│   │   │   ├── LineDetailPage.vue    # 行详情页 (辅助)
│   │   │   └── LineCheckPage.vue     # 行检查页 (辅助)
│   │   └── components/
│   │       └── LineCanvas.vue        # 画布组件: 图像渲染 + 切割线交互
│   ├── index.html                    # HTML 模板
│   ├── vite.config.ts                # Vite 配置 (含 /api 代理到 :8000)
│   ├── package.json                  # npm 依赖与脚本
│   ├── package-lock.json             # 依赖锁定
│   └── dist/                         # 构建产物
│
└── docs/                             # 文档目录
    ├── 汉字切割标注系统-需求说明书.md
    ├── 需求说明书.md
    ├── 概要设计说明书.md
    ├── 详细设计说明书.md
    └── UML_4+1视图模型.md            # 本文档
```

### 4.2 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                      后端模块依赖图                              │
└─────────────────────────────────────────────────────────────────┘

  start_server.py
       │
       ▼
    app.py  ──────────────────> FastAPI / uvicorn / pydantic
       │
       ├──> config.py           (ProjectManager 单例)
       │        │
       │        └──> 数据: datahome/project/*/project.json
       │
       ├──> annotation_registry.py  (AnnotationRegistry 单例)
       │        │
       │        └──> 数据: datahome/annotation_registry.json
       │
       ├──> project_service.py      (ProjectService)
       │        │
       │        └──> selector_service.py  (SelectorService)
       │                 │
       │                 └──> annotation_registry  (查询未标注)
       │
       └──> 数据读取 (内联于 app.py):
            ├──> datahome/rule_jsons/<id>_rule.json
            ├──> datahome/lines/<id>.png
            ├──> datahome/lineage.json               ← 后处理合并数据
            └──> project/<id>/annotations/<id>.json


┌─────────────────────────────────────────────────────────────────┐
│                      前端模块依赖图                              │
└─────────────────────────────────────────────────────────────────┘

  main.ts
    │
    ├──> App.vue
    │      │
    │      └──> router/index.js
    │             │
    │             ├──> views/HomePage.vue ───> api/index.js ──> axios
    │             ├──> views/SegmentLabelPage.vue
    │             └──> views/LabelPage.vue
    │                    │
    │                    └──> components/LineCanvas.vue
    │
    └──> ant-design-vue (UI 组件库)
    └──> vue-router (路由)
```

### 4.3 分层职责矩阵

| 模块 | 表现层 | 接口层 | 服务层 | 数据层 |
|------|--------|--------|--------|--------|
| LabelPage.vue | ● | ○ | | |
| LineCanvas.vue | ● | | | |
| api/index.js | | ● | | |
| app.py (路由) | | ● | ○ | |
| app.py (工具) | | ○ | | ● |
| ProjectManager | | | ● | ○ |
| ProjectService | | | ● | ○ |
| SelectorService | | | ● | ○ |
| AnnotationRegistry | | | ● | ○ |
| 文件系统 | | | | ● |

> ● = 主要职责  ○ = 次要职责

### 4.4 技术栈

| 层次 | 技术 | 版本/说明 |
|------|------|-----------|
| 前端框架 | Vue 3 | Composition API (`<script setup>`) |
| 前端 UI | Ant Design Vue | Button/Tag/Spin/Table |
| 前端构建 | Vite | 开发服务器 + 反向代理 |
| 前端路由 | Vue Router 4 | createWebHistory |
| HTTP 客户端 | Axios | baseURL=/api |
| 后端框架 | FastAPI | 异步 ASGI 框架 |
| 后端运行 | Uvicorn | ASGI 服务器 |
| 后端校验 | Pydantic | 请求体模型 |
| 图像处理 | Pillow (PIL) | 读取图片宽度 |
| 数据格式 | JSON | 全部数据持久化 |

---

## 五、物理视图（Physical View）

### 5.1 部署节点图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    物理部署视图 (开发环境)                          │
└─────────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────┐
  │              开发者主机 (Windows 11)                          │
  │                                                               │
  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │
  │  │  浏览器     │    │ Node.js     │    │ Python venv     │   │
  │  │ (Chrome等)  │    │ (Vite)      │    │ (uvicorn)       │   │
  │  │             │    │             │    │                 │   │
  │  │ 端口: 任意  │    │ 端口: 5173  │    │ 端口: 8000      │   │
  │  └──────┬──────┘    └──────┬──────┘    └────────┬────────┘   │
  │         │                  │                    │            │
  │         │   HTTP (同源)    │  HTTP /api 代理    │            │
  │         │                  │                    │            │
  │         └──────────────────┴────────────────────┘            │
  │                            │                                 │
  │                            ▼                                 │
  │  ┌─────────────────────────────────────────────────────────┐ │
  │  │              本地文件系统 (磁盘)                        │ │
  │  │                                                         │ │
  │  │  d:\projects\work_projects\data_service\               │ │
  │  │  ├── cutting_labeling_system\  (应用代码)              │ │
  │  │  │   ├── backend\            (FastAPI)                 │ │
  │  │  │   └── frontend\           (Vue SPA)                 │ │
  │  │  ├── datahome\               (共享数据池)              │ │
  │  │  │   ├── project\<id>\       (项目视图)                │ │
  │  │  │   │   ├── project.json                             │ │
  │  │  │   │   ├── line_id_list.json                        │ │
  │  │  │   │   ├── annotations\                             │ │
  │  │  │   │   ├── model_jsons\                             │ │
  │  │  │   │   └── fusion_jsons\                            │ │
  │  │  │   ├── rule_jsons\         (全局规则切割结果)        │ │
  │  │  │   ├── lines\              (全局行图片)              │ │
  │  │  │   ├── lineage.json        (全局后处理合并数据)      │ │
  │  │  │   └── annotation_registry.json (标注状态)           │ │
  │  │  └── venv\                   (Python 虚拟环境)         │ │
  │  └─────────────────────────────────────────────────────────┘ │
  └───────────────────────────────────────────────────────────────┘

  网络通信:
  - 浏览器 ↔ Vite:      localhost:5173 (HTTP)
  - Vite ↔ FastAPI:     localhost:8000 (HTTP, 反向代理 /api)
  - FastAPI ↔ 文件系统:  本地磁盘 I/O (同步阻塞)
```

### 5.2 数据存储分布

```
┌─────────────────────────────────────────────────────────────────┐
│                    数据存储物理分布                              │
└─────────────────────────────────────────────────────────────────┘

  共享数据池 (datahome/, 跨项目共享)
  ┌────────────────────────────────────────────────────┐
  │ rule_jsons/                                        │
  │   ├── <line_id>_rule.json   (规则切割原始结果)     │ ← 区域二数据源(原始)
  │   └── ...                                          │
  │                                                    │
  │ lines/                                             │
  │   ├── <line_id>.png        (行图片)                │ ← 区域一数据源
  │   └── ...                                          │
  │                                                    │
  │ lineage.json                (后处理合并数据)        │ ← 区域二数据源(合并后)
  │   ├── lines: {line_id: {chars: [...]}}             │   优先级高于 rule_jsons
  │   └── chars: {char_id: {col_start, col_end, ...}}  │
  │                                                    │
  │ annotation_registry.json    (全局标注状态)          │
  │   └── annotations: {line_id: {annotated, ...}}     │
  └────────────────────────────────────────────────────┘

  项目视图 (datahome/project/<id>/, 项目隔离)
  ┌────────────────────────────────────────────────────┐
  │ project.json                (项目配置)              │
  │ line_id_list.json           (项目包含的行ID列表)    │
  │                                                    │
  │ annotations/                                        │
  │   └── <line_id>.json        (标注结果)              │ ← 区域五数据源(历史标注)
  │                                                    │
  │ model_jsons/                                        │
  │   └── <line_id>_model.json  (模型切割结果)          │ ← 区域三数据源
  │                                                    │
  │ fusion_jsons/                                       │
  │   └── <line_id>_fusion.json (融合切割结果)          │ ← 区域四数据源
  └────────────────────────────────────────────────────┘
```

### 5.3 启动与访问

| 组件 | 启动命令 | 端口 | 访问地址 |
|------|---------|------|---------|
| 后端 | `venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000` | 8000 | http://localhost:8000/api |
| 前端 | `npm run dev` (frontend 目录) | 5173 | http://localhost:5173 |

> 注意：后端必须使用项目 venv 的 Python 启动，避免使用系统 Python 导致依赖缺失。

---

## 六、场景视图（Scenarios，+1）

### 6.1 场景一：标注员标注一行汉字

```
┌─────────────────────────────────────────────────────────────────┐
│  场景: 标注员标注一行汉字 (主流程)                               │
└─────────────────────────────────────────────────────────────────┘

  参与者: 标注员
  前置条件: 已有项目, 项目包含待标注行
  
  主流程:
  1. 标注员打开首页 HomePage
     → GET /api/projects 获取项目列表
     → GET /api/annotation/stats 获取标注统计
  
  2. 标注员点击"开始标注"进入 LabelPage
     → GET /api/images/{line_id}/detail?project_id=xxx
     → 后端加载 rule_json + lineage.json + annotations
     → 返回: rule_lines(来自lineage合并数据) + model_lines + 
             fusion_lines + annotation(历史标注) + 图片URL
  
  3. 前端渲染五区域:
     - 区域一: <img src="image_url"> 原始行图片
     - 区域二: LineCanvas(readonly) 规则切割预览 (合并后数据)
     - 区域三: LineCanvas(readonly) 模型切割预览
     - 区域四: LineCanvas(readonly) 融合切割预览
     - 区域五: LineCanvas(editable) 可编辑切割线 (初始=历史标注, 支持颜色选择:无/红线/绿线, 支持缩放:50%-300%)
  
  4. 标注员选择方案:
     方式A: 点击"使用规则切割" → 一键覆盖区域五
     方式B: 在区域二选中线条 → "复制选中分割线" → 追加到区域五
     方式C: 直接在区域五空白处点击 → 添加新切割线（需先选择颜色:无/红线/绿线，选"无"时点击不画线，防止误操作）
  
  5. 标注员微调切割线:
     - 单击选中 (8px范围内)
     - Ctrl+拖拽框选多条
     - 方向键左/右 1px 微调
     - Delete 删除选中
  
  6. 标注员点击"保存"
     → POST /api/images/{line_id}/annotate
     → 后端 lines_to_chars() 转换为字符列表
     → 写入 annotations/<line_id>.json
     → AnnotationRegistry.mark_annotated() 更新全局状态
  
  异常分支:
  - 样本难以确定 → 点击"暂不标注" → POST /postpone
  - 已暂不标注 → 点击"取消暂不标注" → POST /unpostpone
```

### 6.2 场景二：创建项目并选择样本

```
┌─────────────────────────────────────────────────────────────────┐
│  场景: 创建项目并选择样本                                        │
└─────────────────────────────────────────────────────────────────┘

  参与者: 标注管理员
  前置条件: datahome/rule_jsons/ 已有切割结果数据
  
  主流程:
  1. 管理员在 HomePage 填写项目信息:
     - 项目名称
     - 选择策略: random / pdf / al
     - 样本数量 count
  
  2. (可选) 预览选择结果
     → POST /api/selectors/preview
     → 返回: 策略说明 + 选中数量 + 样本示例
  
  3. 管理员确认创建
     → POST /api/projects
     → ProjectService.create_project():
       a. 生成 project_id (proj_yyyyMMdd_HHmmss)
       b. 创建项目目录 (复制 _template 模板)
       c. SelectorService.select() 选择 line_ids
       d. 写入 line_id_list.json
       e. 写入 project.json (含 paths 配置)
     → ProjectManager._load_projects() 重新加载
     → ProjectManager.set_current_project() 设为当前
  
  4. 项目创建完成, 标注员可开始标注
  
  涉及视图:
  - 逻辑: ProjectService + SelectorService + AnnotationRegistry 协作
  - 进程: 同步文件 I/O, 单次请求完成
  - 开发: project_service.py → selector_service.py → config.py
  - 物理: 创建 datahome/project/<id>/ 目录结构
```

### 6.3 场景三：规则切割后处理数据接入

```
┌─────────────────────────────────────────────────────────────────┐
│  场景: 区域二规则切割预览使用后处理合并数据                      │
└─────────────────────────────────────────────────────────────────┘

  背景: rule_jsons 存储原始规则切割结果(可能过细分割),
        lineage.json 存储后处理合并后的结果(粘连字符已合并)
  
  主流程:
  1. GET /api/images/{line_id}/detail
  
  2. app.py get_detail():
     a. rule_data = load_json_file(rule_file)         # 原始数据
     b. lineage_data = load_lineage_data(project)     # 合并数据
        - 路径1: project_root.parent.parent / lineage.json
        - 路径2: project_root.parent / lineage.json (兜底)
     c. lineage_chars = get_chars_from_lineage(lineage_data, line_id)
        - 从 lines[line_id].chars 获取 char_id 列表
        - 从 chars[char_id] 获取 col_start/col_end
        - 支持 line_id 格式转换 (line_page_ ↔ line_page_pdf_)
  
  3. 判断逻辑:
     if lineage_chars:                                # 优先用合并数据
         rule_lines = chars_to_lines(lineage_chars)
     else:                                            # 回退到原始数据
         rule_lines = chars_to_lines(rule_data.chars)
  
  4. 返回前端: rule_lines (合并后的76条 = 38字符 × 2)
  
  涉及视图:
  - 逻辑: load_lineage_data + get_chars_from_lineage + chars_to_lines
  - 进程: 每次请求重新加载 lineage.json (无缓存, 大文件读取)
  - 开发: app.py 内联实现, 依赖 datahome/lineage.json
  - 物理: lineage.json 约 43772 行, 单次读取约数十 MB
```

---

## 七、视图间关系与一致性检查

### 7.1 视图追踪矩阵

| 场景 | 逻辑视图元素 | 进程视图元素 | 开发视图元素 | 物理视图元素 |
|------|-------------|-------------|-------------|-------------|
| 标注一行 | LabelPage + LineCanvas + get_detail + annotate | FastAPI协程 + 文件I/O | app.py + LabelPage.vue + LineCanvas.vue | 浏览器→5173→8000→磁盘 |
| 创建项目 | ProjectService + SelectorService | 同步I/O阻塞事件循环 | project_service.py + selector_service.py | datahome/project/ 目录创建 |
| 后处理数据接入 | load_lineage_data + get_chars_from_lineage | 每请求读取大文件 | app.py 内联函数 | lineage.json 单文件 |

### 7.2 架构约束与权衡

| 约束 | 当前实现 | 权衡说明 |
|------|---------|---------|
| 单机部署 | 全部组件在同一主机 | 开发期可接受, 生产需独立部署 |
| 文件存储 | JSON 文件, 无数据库 | 简单直接, 但并发写有风险 |
| 内存缓存 | _image_cache 5分钟TTL | 减少目录扫描, 但内存占用增长 |
| lineage.json 加载 | 每请求全量读取 | 数据量大时影响响应延迟 |
| 标注注册表 | 每次标注同步写文件 | 保证持久化, 但并发写可能丢失 |
| 前端画布 | 原生 Canvas API | 灵活可控, 但无虚拟DOM优化 |

### 7.3 演进建议

1. **lineage.json 缓存**：当前每次请求全量读取数十 MB 文件，建议加内存缓存（类似 _image_cache）
2. **标注注册表并发**：单进程下风险低，多 worker 部署需引入文件锁或迁移到数据库
3. **图片流响应**：大图片同步读取阻塞事件循环，建议改为流式响应或异步读取
4. **前端构建**：dist/ 已有构建产物，生产环境应使用 Nginx 托管静态资源 + 反向代理 API

---

## 八、附录

### 8.1 术语表

| 术语 | 说明 |
|------|------|
| line_id | 行唯一标识, 格式: line_page_pdf_<pdf_id>_<page>_<idx> |
| rule_jsons | 规则切割结果目录, 每行一个 JSON |
| lineage.json | 全局后处理数据, 含粘连字符合并结果 |
| annotation_registry | 全局标注状态注册表 |
| 五区域 | LabelPage 的五个布局区域: 原始图/规则预览/模型预览/融合预览/可编辑 |
| 切割线 | {pos, color}, pos 为 x 坐标, red=字符起始, green=字符结束 |
| 共享数据池 | datahome/ 下跨项目共享的数据 (rule_jsons, lines, lineage) |
| 项目视图 | datahome/project/<id>/ 下项目隔离的数据 (annotations, model_jsons) |

### 8.2 参考文档

- [需求说明书](file:///d:/projects/work_projects/data_service/cutting_labeling_system/docs/需求说明书.md)
- [概要设计说明书](file:///d:/projects/work_projects/data_service/cutting_labeling_system/docs/概要设计说明书.md)
- [详细设计说明书](file:///d:/projects/work_projects/data_service/cutting_labeling_system/docs/详细设计说明书.md)
- [app.py](file:///d:/projects/work_projects/data_service/cutting_labeling_system/backend/app.py)
- [config.py](file:///d:/projects/work_projects/data_service/cutting_labeling_system/backend/config.py)
- [LabelPage.vue](file:///d:/projects/work_projects/data_service/cutting_labeling_system/frontend/src/views/LabelPage.vue)
- [LineCanvas.vue](file:///d:/projects/work_projects/data_service/cutting_labeling_system/frontend/src/components/LineCanvas.vue)
