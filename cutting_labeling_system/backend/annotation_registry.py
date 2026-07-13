from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional, Set


class AnnotationRegistry:
    _instance = None
    _data: Dict[str, dict] = {}
    _file_path: Path = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        self._file_path = Path(__file__).parent.parent.parent / "datahome" / "annotation_registry.json"
        if self._file_path.exists():
            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._data = data.get("annotations", {})
            except Exception as e:
                print(f"加载标注注册表失败: {e}")
                self._data = {}
        else:
            self._data = {}
            self._save()

    def _save(self):
        try:
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "version": "1.0",
                    "created_at": "2026-07-05T10:00:00",
                    "updated_at": datetime.now().isoformat(),
                    "annotations": self._data
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存标注注册表失败: {e}")

    def mark_annotated(self, line_id: str, project_id: str):
        self._data[line_id] = {
            "annotated": True,
            "project_id": project_id,
            "annotated_at": datetime.now().isoformat(),
            "postponed": False
        }
        self._save()

    def mark_postponed(self, line_id: str, project_id: str):
        entry = self._data.get(line_id, {})
        entry.update({
            "annotated": False,
            "postponed": True,
            "project_id": project_id,
            "updated_at": datetime.now().isoformat()
        })
        self._data[line_id] = entry
        self._save()

    def mark_unpostponed(self, line_id: str):
        entry = self._data.get(line_id)
        if entry:
            entry["postponed"] = False
            entry["updated_at"] = datetime.now().isoformat()
            self._save()

    def is_annotated(self, line_id: str) -> bool:
        entry = self._data.get(line_id)
        return entry is not None and entry.get("annotated", False)

    def is_postponed(self, line_id: str) -> bool:
        entry = self._data.get(line_id)
        return entry is not None and entry.get("postponed", False)

    def get_entry(self, line_id: str) -> Optional[dict]:
        return self._data.get(line_id)

    def get_annotated_line_ids(self) -> Set[str]:
        return {line_id for line_id, entry in self._data.items() if entry.get("annotated", False)}

    def get_postponed_line_ids(self) -> Set[str]:
        return {line_id for line_id, entry in self._data.items() if entry.get("postponed", False)}

    def get_unannotated_line_ids(self, all_line_ids: Set[str]) -> Set[str]:
        annotated = self.get_annotated_line_ids()
        postponed = self.get_postponed_line_ids()
        return all_line_ids - annotated - postponed

    def get_stats(self) -> dict:
        total = len(self._data)
        annotated = len(self.get_annotated_line_ids())
        postponed = len(self.get_postponed_line_ids())
        return {
            "total_tracked": total,
            "annotated": annotated,
            "postponed": postponed
        }


annotation_registry = AnnotationRegistry()