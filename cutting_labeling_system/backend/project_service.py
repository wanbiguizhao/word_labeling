import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from .selector_service import selector_service


class ProjectService:
    def __init__(self):
        self._projects_base_dir = Path(__file__).parent.parent.parent / "datahome" / "project"
        self._template_dir = self._projects_base_dir / "_template"

    def create_project(self, project_name: str, strategy: str, pdf_ids: Optional[List[str]] = None, count: int = 100, description: str = "") -> dict:
        project_id = f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_dir = self._projects_base_dir / project_id

        project_dir.mkdir(parents=True, exist_ok=True)

        if self._template_dir.exists():
            for item in self._template_dir.iterdir():
                if item.is_file():
                    shutil.copy(item, project_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, project_dir / item.name, dirs_exist_ok=True)
        else:
            (project_dir / "annotations").mkdir(exist_ok=True)
            (project_dir / "model_jsons").mkdir(exist_ok=True)
            (project_dir / "fusion_jsons").mkdir(exist_ok=True)

        selected_line_ids = selector_service.select(strategy, pdf_ids, count)

        line_id_list_file = project_dir / "line_id_list.json"
        with open(line_id_list_file, "w", encoding="utf-8") as f:
            json.dump({
                "project_id": project_id,
                "total_count": len(selected_line_ids),
                "line_ids": selected_line_ids
            }, f, ensure_ascii=False, indent=2)

        pdf_files = []
        if pdf_ids:
            for idx, pdf_id in enumerate(pdf_ids):
                pdf_files.append({
                    "pdf_id": pdf_id,
                    "pdf_name": f"PDF_{pdf_id}.pdf",
                    "page_count": 0,
                    "total_lines": 0
                })

        now = datetime.now().isoformat()
        pdf_ids_json = json.dumps(pdf_ids or [])

        template_file = project_dir / "project.json.template"
        if template_file.exists():
            with open(template_file, "r", encoding="utf-8") as f:
                template_content = f.read()

            project_config = template_content.replace("{{project_id}}", project_id)
            project_config = project_config.replace("{{project_name}}", project_name)
            project_config = project_config.replace("{{description}}", description)
            project_config = project_config.replace("{{line_id_count}}", str(len(selected_line_ids)))
            project_config = project_config.replace("{{strategy}}", strategy)
            project_config = project_config.replace("{{pdf_ids}}", pdf_ids_json)
            project_config = project_config.replace("{{count}}", str(count))
            project_config = project_config.replace("{{created_at}}", now)
            project_config = project_config.replace("{{updated_at}}", now)

            project_config = json.loads(project_config)
            template_file.unlink()
        else:
            project_config = {
                "project_id": project_id,
                "project_name": project_name,
                "description": description,
                "pdf_files": [],
                "line_id_list": "line_id_list.json",
                "line_id_count": len(selected_line_ids),
                "selector": {
                    "strategy": strategy,
                    "pdf_ids": pdf_ids or [],
                    "count": count,
                    "created_at": now
                },
                "paths": {
                    "rule_jsons": "../../rule_jsons",
                    "model_jsons": "model_jsons",
                    "fusion_jsons": "fusion_jsons",
                    "annotations": "annotations",
                    "lines": "../../lines"
                },
                "created_at": now,
                "updated_at": now
            }

        project_config["pdf_files"] = pdf_files

        project_file = project_dir / "project.json"
        with open(project_file, "w", encoding="utf-8") as f:
            json.dump(project_config, f, ensure_ascii=False, indent=2)

        return project_config

    def update_project(self, project_id: str, **kwargs) -> Optional[dict]:
        project_dir = self._projects_base_dir / project_id
        project_file = project_dir / "project.json"

        if not project_file.exists():
            return None

        with open(project_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        for key, value in kwargs.items():
            if key in config:
                config[key] = value

        config["updated_at"] = datetime.now().isoformat()

        with open(project_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        return config

    def delete_project(self, project_id: str) -> bool:
        project_dir = self._projects_base_dir / project_id
        if not project_dir.exists():
            return False

        for item in project_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                for sub_item in item.iterdir():
                    sub_item.unlink()
                item.rmdir()

        project_dir.rmdir()
        return True

    def get_project_config(self, project_id: str) -> Optional[dict]:
        project_file = self._projects_base_dir / project_id / "project.json"
        if not project_file.exists():
            return None

        with open(project_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_projects(self) -> List[dict]:
        projects = []
        if not self._projects_base_dir.exists():
            return projects

        for project_dir in self._projects_base_dir.iterdir():
            if not project_dir.is_dir():
                continue
            if project_dir.name.startswith("_"):
                continue

            project_file = project_dir / "project.json"
            if not project_file.exists():
                continue

            try:
                with open(project_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                projects.append(config)
            except Exception:
                continue

        return projects

    def get_project_stats(self, project_id: str) -> dict:
        project_file = self._projects_base_dir / project_id / "project.json"
        line_id_list_file = self._projects_base_dir / project_id / "line_id_list.json"
        annotations_dir = self._projects_base_dir / project_id / "annotations"

        total_lines = 0
        annotated_count = 0

        if line_id_list_file.exists():
            with open(line_id_list_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                total_lines = data.get("total_count", 0)

        if annotations_dir.exists():
            annotated_count = len(list(annotations_dir.glob("*.json")))

        return {
            "project_id": project_id,
            "total_lines": total_lines,
            "annotated_count": annotated_count,
            "unannotated_count": total_lines - annotated_count
        }


project_service = ProjectService()