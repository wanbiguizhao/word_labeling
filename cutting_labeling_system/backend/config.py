from pathlib import Path
import json
from typing import Optional, Dict, List, Set


class ProjectConfig:
    def __init__(self, project_id: str, config_data: dict):
        self.project_id = project_id
        self.project_name = config_data.get("project_name", project_id)
        self.description = config_data.get("description", "")
        self.pdf_files = config_data.get("pdf_files", [])
        self.paths = config_data.get("paths", {})
        self.created_at = config_data.get("created_at")
        self.updated_at = config_data.get("updated_at")
        self._project_root = Path(config_data.get("project_root", "."))
        self._line_id_list_field = config_data.get("line_id_list")
        self._line_id_set: Optional[Set[str]] = None
        self._line_id_list: Optional[List[str]] = None
    
    def get_path(self, key: str) -> Path:
        rel_path = self.paths.get(key)
        if rel_path:
            return (self._project_root / rel_path).resolve()
        return self._project_root
    
    @property
    def rule_jsons_dir(self) -> Path:
        return self.get_path("rule_jsons")
    
    @property
    def model_jsons_dir(self) -> Path:
        return self.get_path("model_jsons")
    
    @property
    def fusion_jsons_dir(self) -> Path:
        return self.get_path("fusion_jsons")
    
    @property
    def annotations_dir(self) -> Path:
        return self.get_path("annotations")
    
    @property
    def lines_dir(self) -> Path:
        return self.get_path("lines")
    
    @property
    def project_root(self) -> Path:
        return self._project_root
    
    @property
    def line_id_list_file(self) -> Optional[Path]:
        if not self._line_id_list_field:
            return None
        return (self._project_root / self._line_id_list_field).resolve()
    
    @property
    def line_ids(self) -> List[str]:
        if self._line_id_list is None:
            self._load_line_id_list()
        return self._line_id_list or []
    
    @property
    def line_id_set(self) -> Set[str]:
        if self._line_id_set is None:
            self._line_id_set = set(self.line_ids)
        return self._line_id_set
    
    def has_line_id(self, line_id: str) -> bool:
        if not self._line_id_list_field:
            return True
        return line_id in self.line_id_set
    
    def _load_line_id_list(self):
        list_file = self.line_id_list_file
        if not list_file or not list_file.exists():
            self._line_id_list = []
            self._line_id_set = set()
            return
        try:
            with open(list_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._line_id_list = data
            elif isinstance(data, dict):
                self._line_id_list = data.get("line_ids", [])
            else:
                self._line_id_list = []
            self._line_id_set = set(self._line_id_list)
        except Exception as e:
            print(f"加载 line_id_list 失败: {list_file}, {e}")
            self._line_id_list = []
            self._line_id_set = set()


class ProjectManager:
    _instance = None
    _projects: Dict[str, ProjectConfig] = {}
    _current_project: Optional[ProjectConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_projects()
        return cls._instance
    
    def _load_projects(self):
        projects_base_dir = Path(__file__).parent.parent.parent / "datahome" / "project"
        if projects_base_dir.exists():
            for project_dir in projects_base_dir.iterdir():
                if not project_dir.is_dir():
                    continue
                if project_dir.name.startswith("_"):
                    continue
                project_file = project_dir / "project.json"
                if not project_file.exists():
                    continue
                try:
                    with open(project_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    project_id = config_data.get("project_id")
                    if project_id:
                        config_data["project_root"] = str(project_dir)
                        self._projects[project_id] = ProjectConfig(project_id, config_data)
                except Exception as e:
                    print(f"加载项目配置失败: {project_file}, {e}")
        
        if self._projects:
            self._current_project = next(iter(self._projects.values()))
    
    def get_project(self, project_id: str) -> Optional[ProjectConfig]:
        return self._projects.get(project_id)
    
    def list_projects(self) -> Dict[str, str]:
        return {pid: p.project_name for pid, p in self._projects.items()}
    
    def set_current_project(self, project_id: str) -> bool:
        project = self._projects.get(project_id)
        if project:
            self._current_project = project
            return True
        return False
    
    @property
    def current_project(self) -> Optional[ProjectConfig]:
        return self._current_project


project_manager = ProjectManager()

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent

DATAHOME_DIR = PROJECT_ROOT / "datahome"

MERGED_ANNOTATIONS_PATH = DATAHOME_DIR / "datasets" / "merged_annotations.json"