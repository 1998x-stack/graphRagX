"""
图谱构建器
将提取结果合并为统一的知识图谱
"""
from typing import List
from models.schemas import ExtractionResult
from models.graph import KnowledgeGraph
from utils.logger import log


class GraphBuilder:
    """图谱构建器"""
    
    def __init__(self):
        """初始化构建器"""
        log.info("GraphBuilder initialized")
    
    def build_graph(
        self,
        extraction_results: List[ExtractionResult]
    ) -> KnowledgeGraph:
        """
        从提取结果构建知识图谱
        
        核心策略：
        1. 实体消歧：同名同类型的实体合并，描述拼接
        2. 关系合并：同源同目标的关系权重累加
        
        Args:
            extraction_results: 提取结果列表
            
        Returns:
            知识图谱
        """
        log.info(f"Building graph from {len(extraction_results)} extraction results")
        
        kg = KnowledgeGraph()
        
        # 阶段1：添加所有实体（自动合并同名实体）
        for result in extraction_results:
            for entity in result.entities:
                kg.add_entity(entity)
        
        log.info(f"Added {len(kg.entities)} unique entities")
        
        # 阶段2：添加所有关系（自动合并重复关系）
        for result in extraction_results:
            for relation in result.relations:
                kg.add_relation(relation)
        
        log.info(f"Added {len(kg.relations)} relations")
        
        # 打印统计信息
        stats = kg.get_statistics()
        log.info(f"Graph statistics: {stats}")
        
        return kg
    
    def merge_graphs(
        self,
        graphs: List[KnowledgeGraph]
    ) -> KnowledgeGraph:
        """
        合并多个图谱
        
        Args:
            graphs: 图谱列表
            
        Returns:
            合并后的图谱
        """
        log.info(f"Merging {len(graphs)} graphs")
        
        merged = KnowledgeGraph()
        
        for graph in graphs:
            # 合并实体
            for entity in graph.entities.values():
                merged.add_entity(entity)
            
            # 合并关系
            for relation in graph.relations:
                merged.add_relation(relation)
        
        log.info(f"Merged graph: {len(merged.entities)} entities, {len(merged.relations)} relations")
        return merged


# 全局构建器实例
graph_builder = GraphBuilder()