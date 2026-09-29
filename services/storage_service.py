"""Local filesystem storage adapter with atomic writes and safe index IDs."""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from config import settings
from models.schemas import GraphData, INDEX_ID_PATTERN
from utils.logger import log, log_exception


_INDEX_ID_RE = re.compile(INDEX_ID_PATTERN)


class StorageService:
    def __init__(self):
        settings.ensure_directories()
        self.output_dir = settings.OUTPUT_DIR
        self.llm_logs_dir = settings.LLM_LOGS_DIR
        self.graphs_dir = settings.GRAPHS_DIR
        self.communities_dir = settings.COMMUNITIES_DIR

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

    @staticmethod
    def validate_index_id(index_id: str) -> str:
        if not _INDEX_ID_RE.fullmatch(index_id):
            raise ValueError(
                "index_id must match /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/"
            )
        return index_id

    def _graph_path(self, index_id: str) -> Path:
        safe_id = self.validate_index_id(index_id)
        return self.graphs_dir / f"{safe_id}_graph.json"

    @staticmethod
    def _safe_write_json(filepath: Path, data: Any) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{filepath.name}.", suffix=".tmp", dir=filepath.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, filepath)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _safe_read_json(filepath: Path) -> Any:
        with filepath.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_llm_response(self, task: str, prompt: str, response: str, metadata: Dict | None = None) -> Path:
        filepath = self.llm_logs_dir / f"{self._timestamp()}_{task}.json"
        self._safe_write_json(
            filepath,
            {
                "timestamp": self._timestamp(),
                "task": task,
                "prompt": prompt,
                "response": response,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "metadata": metadata or {},
            },
        )
        return filepath

    def save_graph(self, index_id: str, graph_data: GraphData) -> Path:
        filepath = self._graph_path(index_id)
        payload = {
            "schema_version": settings.INDEX_SCHEMA_VERSION,
            "index_id": index_id,
            "timestamp": self._timestamp(),
            "graph": graph_data.model_dump(mode="json"),
        }
        self._safe_write_json(filepath, payload)
        log.info("Saved index {} -> {}", index_id, filepath)
        return filepath

    def load_graph(self, index_id: str) -> GraphData:
        filepath = self._graph_path(index_id)
        if not filepath.exists():
            raise FileNotFoundError(f"Index not found: {index_id}")

        try:
            data = self._safe_read_json(filepath)
            if "graph" in data:
                return GraphData.model_validate(data["graph"])
            return GraphData.model_validate(data)
        except Exception as exc:
            log_exception(exc, f"load_graph({index_id})")
            raise

    def index_exists(self, index_id: str) -> bool:
        return self._graph_path(index_id).exists()

    def delete_index(self, index_id: str) -> bool:
        filepath = self._graph_path(index_id)
        if not filepath.exists():
            return False
        filepath.unlink()
        return True

    def list_indexes(self) -> list[str]:
        suffix = "_graph.json"
        return sorted(
            path.name[: -len(suffix)]
            for path in self.graphs_dir.glob(f"*{suffix}")
            if path.is_file()
        )


storage_service = StorageService()
