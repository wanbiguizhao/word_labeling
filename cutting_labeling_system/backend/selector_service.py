import random
import json
from pathlib import Path
from typing import List, Set, Optional, Dict
from annotation_registry import annotation_registry


class SelectorService:
    def __init__(self):
        self._global_rule_dir = Path(__file__).parent.parent.parent / "datahome" / "rule_jsons"
        self._al_ranking_path = Path(__file__).parent.parent.parent / "ai_model" / "models" / "merged_annotation_al_ranking.json"

    def get_all_line_ids(self) -> Set[str]:
        line_ids = set()
        if self._global_rule_dir.exists():
            for rule_file in self._global_rule_dir.glob("*_rule.json"):
                try:
                    with open(rule_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    line_id = data.get("line_id")
                    if line_id:
                        line_ids.add(line_id)
                except Exception:
                    continue
        return line_ids

    def select_by_pdf(self, pdf_ids: List[str], count: int = 100) -> List[str]:
        all_line_ids = self.get_all_line_ids()
        unannotated = annotation_registry.get_unannotated_line_ids(all_line_ids)

        candidates = []
        for line_id in unannotated:
            for pdf_id in pdf_ids:
                if pdf_id in line_id:
                    candidates.append(line_id)
                    break

        return candidates[:count]

    def select_random(self, count: int = 100) -> List[str]:
        all_line_ids = self.get_all_line_ids()
        unannotated = annotation_registry.get_unannotated_line_ids(all_line_ids)
        candidates = list(unannotated)
        random.shuffle(candidates)
        return candidates[:count]

    def select_by_al(self, count: int = 100) -> List[str]:
        if not self._al_ranking_path.exists():
            return self.select_random(count)

        all_line_ids = self.get_all_line_ids()
        unannotated = annotation_registry.get_unannotated_line_ids(all_line_ids)

        try:
            with open(self._al_ranking_path, "r", encoding="utf-8") as f:
                al_data = json.load(f)

            al_items = []
            for item in al_data:
                line_id = item.get("line_id") or item.get("id")
                if line_id and line_id in unannotated:
                    al_items.append({
                        "line_id": line_id,
                        "al_score": item.get("al_score", 0)
                    })

            al_items.sort(key=lambda x: x["al_score"], reverse=True)
            return [item["line_id"] for item in al_items[:count]]
        except Exception as e:
            print(f"AL选择器加载失败: {e}")
            return self.select_random(count)

    def select(self, strategy: str, pdf_ids: Optional[List[str]] = None, count: int = 100) -> List[str]:
        if strategy == "pdf":
            return self.select_by_pdf(pdf_ids or [], count)
        elif strategy == "al":
            return self.select_by_al(count)
        else:
            return self.select_random(count)

    def preview(self, strategy: str, pdf_ids: Optional[List[str]] = None, count: int = 20) -> dict:
        selected = self.select(strategy, pdf_ids, count)
        all_line_ids = self.get_all_line_ids()
        unannotated = annotation_registry.get_unannotated_line_ids(all_line_ids)

        return {
            "strategy": strategy,
            "total_unannotated": len(unannotated),
            "selected_count": len(selected),
            "sample_line_ids": selected[:5],
            "selected_line_ids": selected
        }


selector_service = SelectorService()