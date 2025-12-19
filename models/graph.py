"""
图数据结构
使用 NetworkX 构建和操作图
"""
import networkx as nx
from typing import List, Dict, Set, Optional
from models.schemas import Entity, Relation, Community, GraphData
from utils.logger import log


class KnowledgeGraph:
    """知识图谱类"""
    
    def __init__(self):
        """初始化知识图谱"""
        self.graph = nx.Graph()
        self.entities: Dict[str, Entity] = {}  # {name: Entity}
        self.relations: List[Relation] = []
        self.communities: List[Community] = []
        
        log.info("KnowledgeGraph initialized")
    
    def add_entity(self, entity: Entity):
        """
        添加实体到图谱
        
        Args:
            entity: 实体对象
        """
        if entity.name not in self.entities:
            self.entities[entity.name] = entity
            self.graph.add_node(entity.name, **entity.dict())
            log.debug(f"Added entity: {entity.name} ({entity.type})")
        else:
            # 合并描述（实体消歧）
            existing = self.entities[entity.name]
            existing.description += f" | {entity.description}"
            existing.source_chunk_ids.extend(entity.source_chunk_ids)
            existing.source_chunk_ids = list(set(existing.source_chunk_ids))
            log.debug(f"Merged entity: {entity.name}")
    
    def add_relation(self, relation: Relation):
        """
        添加关系到图谱
        
        Args:
            relation: 关系对象
        """
        # 确保实体存在
        if relation.source not in self.entities or relation.target not in self.entities:
            log.warning(f"Relation {relation.source}->{relation.target} references non-existent entities")
            return
        
        # 添加边
        if self.graph.has_edge(relation.source, relation.target):
            # 更新权重和描述
            edge_data = self.graph[relation.source][relation.target]
            edge_data['weight'] = edge_data.get('weight', 1.0) + relation.weight
            edge_data['descriptions'] = edge_data.get('descriptions', [])
            edge_data['descriptions'].append(relation.description)
            log.debug(f"Updated relation: {relation.source} -> {relation.target}")
        else:
            self.graph.add_edge(
                relation.source,
                relation.target,
                weight=relation.weight,
                relation_type=relation.relation_type,
                description=relation.description,
                descriptions=[relation.description]
            )
            log.debug(f"Added relation: {relation.source} -> {relation.target}")
        
        self.relations.append(relation)
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """获取实体"""
        return self.entities.get(name)
    
    def get_neighbors(self, entity_name: str, depth: int = 1) -> Set[str]:
        """
        获取实体的邻居（支持多跳）
        
        Args:
            entity_name: 实体名称
            depth: 邻居深度
            
        Returns:
            邻居实体名称集合
        """
        if entity_name not in self.graph:
            return set()
        
        neighbors = set()
        current_level = {entity_name}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.neighbors(node))
            neighbors.update(next_level)
            current_level = next_level
        
        neighbors.discard(entity_name)  # 移除自己
        return neighbors
    
    def get_subgraph(self, entity_names: List[str]) -> nx.Graph:
        """
        获取子图
        
        Args:
            entity_names: 实体名称列表
            
        Returns:
            子图
        """
        return self.graph.subgraph(entity_names).copy()
    
    def set_communities(self, communities: List[Community]):
        """设置社区列表"""
        self.communities = communities
        log.info(f"Set {len(communities)} communities")
    
    def get_community_by_entity(self, entity_name: str) -> Optional[Community]:
        """根据实体名称查找所属社区"""
        for community in self.communities:
            if entity_name in community.entities:
                return community
        return None
    
    def to_graph_data(self) -> GraphData:
        """转换为 GraphData 对象（用于序列化）"""
        return GraphData(
            entities=self.entities,
            relations=self.relations,
            communities=self.communities,
            metadata={
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "num_communities": len(self.communities)
            }
        )
    
    @classmethod
    def from_graph_data(cls, graph_data: GraphData) -> 'KnowledgeGraph':
        """从 GraphData 对象重建图谱"""
        kg = cls()
        
        # 添加实体
        for entity in graph_data.entities.values():
            kg.add_entity(entity)
        
        # 添加关系
        for relation in graph_data.relations:
            kg.add_relation(relation)
        
        # 添加社区
        kg.set_communities(graph_data.communities)
        
        log.info(f"Loaded graph: {len(kg.entities)} entities, {len(kg.relations)} relations")
        return kg
    
    def get_statistics(self) -> Dict:
        """获取图谱统计信息"""
        return {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "num_communities": len(self.communities),
            "avg_degree": sum(dict(self.graph.degree()).values()) / len(self.graph.nodes()) if self.graph.nodes() else 0,
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph)
        }