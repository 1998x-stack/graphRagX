"""
文件存储服务
保存所有 LLM 输出、图谱数据、社区数据
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
from config import settings
from models.schemas import GraphData
from utils.logger import log, log_exception


class StorageService:
    """存储服务类"""
    
    def __init__(self):
        """初始化存储服务"""
        self.output_dir = settings.OUTPUT_DIR
        self.llm_logs_dir = settings.LLM_LOGS_DIR
        self.graphs_dir = settings.GRAPHS_DIR
        self.communities_dir = settings.COMMUNITIES_DIR
        
        log.info("StorageService initialized")
    
    def _generate_timestamp(self) -> str:
        """生成时间戳字符串"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    def _safe_write_json(self, filepath: Path, data: Any):
        """
        安全写入 JSON 文件
        
        Args:
            filepath: 文件路径
            data: 数据（可序列化为 JSON）
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            log.debug(f"Saved file: {filepath}")
        except Exception as e:
            log_exception(e, f"_safe_write_json({filepath})")
            raise
    
    def _safe_read_json(self, filepath: Path) -> Any:
        """
        安全读取 JSON 文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            JSON 数据
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            log.debug(f"Loaded file: {filepath}")
            return data
        except Exception as e:
            log_exception(e, f"_safe_read_json({filepath})")
            raise
    
    def save_llm_response(
        self,
        task: str,
        prompt: str,
        response: str,
        metadata: Dict = None
    ) -> Path:
        """
        保存 LLM 响应到文件
        
        Args:
            task: 任务类型（如 'entity_extraction', 'community_summary'）
            prompt: 输入提示词
            response: LLM 响应
            metadata: 元数据
            
        Returns:
            保存的文件路径
        """
        timestamp = self._generate_timestamp()
        filename = f"{timestamp}_{task}.json"
        filepath = self.llm_logs_dir / filename
        
        data = {
            "timestamp": timestamp,
            "task": task,
            "prompt": prompt,
            "response": response,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "metadata": metadata or {}
        }
        
        self._safe_write_json(filepath, data)
        log.info(f"Saved LLM response: {task} -> {filepath}")
        return filepath
    
    def save_graph(self, index_id: str, graph_data: GraphData) -> Path:
        """
        保存图谱数据
        
        Args:
            index_id: 索引ID
            graph_data: 图谱数据
            
        Returns:
            保存的文件路径
        """
        filename = f"{index_id}_graph.json"
        filepath = self.graphs_dir / filename
        
        # 转换为可序列化的格式
        data = {
            "index_id": index_id,
            "timestamp": self._generate_timestamp(),
            "entities": {name: entity.dict() for name, entity in graph_data.entities.items()},
            "relations": [rel.dict() for rel in graph_data.relations],
            "communities": [comm.dict() for comm in graph_data.communities],
            "metadata": graph_data.metadata
        }
        
        self._safe_write_json(filepath, data)
        log.info(f"Saved graph: {index_id} -> {filepath}")
        return filepath
    
    def load_graph(self, index_id: str) -> GraphData:
        """
        加载图谱数据
        
        Args:
            index_id: 索引ID
            
        Returns:
            图谱数据
        """
        filename = f"{index_id}_graph.json"
        filepath = self.graphs_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        
        data = self._safe_read_json(filepath)
        
        # 重建对象
        from models.schemas import Entity, Relation, Community
        
        graph_data = GraphData(
            entities={name: Entity(**entity_data) for name, entity_data in data["entities"].items()},
            relations=[Relation(**rel_data) for rel_data in data["relations"]],
            communities=[Community(**comm_data) for comm_data in data["communities"]],
            metadata=data.get("metadata", {})
        )
        
        log.info(f"Loaded graph: {index_id} from {filepath}")
        return graph_data
    
    def save_communities(self, index_id: str, communities: list) -> Path:
        """
        保存社区数据
        
        Args:
            index_id: 索引ID
            communities: 社区列表
            
        Returns:
            保存的文件路径
        """
        filename = f"{index_id}_communities.json"
        filepath = self.communities_dir / filename
        
        data = {
            "index_id": index_id,
            "timestamp": self._generate_timestamp(),
            "num_communities": len(communities),
            "communities": [comm.dict() if hasattr(comm, 'dict') else comm for comm in communities]
        }
        
        self._safe_write_json(filepath, data)
        log.info(f"Saved communities: {index_id} -> {filepath}")
        return filepath
    
    def list_indexes(self) -> list:
        """
        列出所有索引
        
        Returns:
            索引ID列表
        """
        graph_files = list(self.graphs_dir.glob("*_graph.json"))
        index_ids = [f.stem.replace("_graph", "") for f in graph_files]
        return index_ids


# 全局存储服务实例
storage_service = StorageService()