"""
社区检测算法
使用 Leiden 算法进行层次化社区检测
"""
from typing import List, Dict
import networkx as nx
from models.graph import KnowledgeGraph
from models.schemas import Community
from config import settings
from utils.logger import log, log_exception


class CommunityDetector:
    """社区检测器"""
    
    def __init__(
        self,
        resolution: float = None,
        max_iterations: int = None,
        seed: int = None
    ):
        """
        初始化社区检测器
        
        Args:
            resolution: 分辨率参数（控制社区粒度）
            max_iterations: 最大迭代次数
            seed: 随机种子
        """
        self.resolution = resolution or settings.LEIDEN_RESOLUTION
        self.max_iterations = max_iterations or settings.LEIDEN_MAX_ITERATIONS
        self.seed = seed or settings.LEIDEN_SEED
        
        log.info(
            f"CommunityDetector initialized: "
            f"resolution={self.resolution}, max_iterations={self.max_iterations}"
        )
    
    def detect_communities(
        self,
        kg: KnowledgeGraph,
        algorithm: str = "louvain"  # "louvain" or "leiden" (需要 igraph)
    ) -> List[Community]:
        """
        检测社区
        
        注意：真正的 Leiden 算法需要 igraph 库，这里使用 NetworkX 的 Louvain 作为简化实现
        
        Args:
            kg: 知识图谱
            algorithm: 算法类型
            
        Returns:
            社区列表
        """
        try:
            log.info(f"Detecting communities using {algorithm} algorithm")
            
            if kg.graph.number_of_nodes() == 0:
                log.warning("Empty graph, no communities to detect")
                return []
            
            # 使用 Louvain 算法（NetworkX 内置）
            # TODO: 如果需要真正的 Leiden，安装 igraph 和 leidenalg 库
            import community as community_louvain  # python-louvain 库
            
            try:
                # Louvain 算法
                partition = community_louvain.best_partition(
                    kg.graph,
                    resolution=self.resolution,
                    random_state=self.seed
                )
            except ImportError:
                log.warning("python-louvain not installed, using greedy modularity")
                # 降级方案：使用 NetworkX 的贪婪模块度算法
                from networkx.algorithms import community as nx_community
                communities_gen = nx_community.greedy_modularity_communities(
                    kg.graph,
                    resolution=self.resolution
                )
                # 转换为 partition 格式
                partition = {}
                for comm_id, nodes in enumerate(communities_gen):
                    for node in nodes:
                        partition[node] = comm_id
            
            # 转换为 Community 对象
            communities_dict: Dict[int, List[str]] = {}
            for node, comm_id in partition.items():
                if comm_id not in communities_dict:
                    communities_dict[comm_id] = []
                communities_dict[comm_id].append(node)
            
            communities = []
            for comm_id, entity_names in communities_dict.items():
                community = Community(
                    id=f"community_{comm_id}",
                    level=0,  # 单层社区
                    entities=entity_names,
                    summary=None,  # 稍后生成
                    parent_id=None,
                    size=len(entity_names)
                )
                communities.append(community)
            
            log.info(f"Detected {len(communities)} communities")
            
            # 打印社区统计
            sizes = [c.size for c in communities]
            log.info(
                f"Community sizes - min: {min(sizes)}, max: {max(sizes)}, "
                f"avg: {sum(sizes) / len(sizes):.1f}"
            )
            
            return communities
            
        except Exception as e:
            log_exception(e, "detect_communities")
            raise
    
    def detect_hierarchical_communities(
        self,
        kg: KnowledgeGraph,
        max_levels: int = 3
    ) -> List[Community]:
        """
        检测层次化社区（递归社区检测）
        
        Args:
            kg: 知识图谱
            max_levels: 最大层级数
            
        Returns:
            所有层级的社区列表
        """
        log.info(f"Detecting hierarchical communities (max_levels={max_levels})")
        
        all_communities = []
        current_graph = kg.graph.copy()
        
        for level in range(max_levels):
            # 检测当前层级的社区
            temp_kg = KnowledgeGraph()
            temp_kg.graph = current_graph
            
            level_communities = self.detect_communities(temp_kg)
            
            if len(level_communities) <= 1:
                log.info(f"Stopping at level {level}: only 1 community")
                break
            
            # 设置层级
            for community in level_communities:
                community.id = f"L{level}_{community.id}"
                community.level = level
            
            all_communities.extend(level_communities)
            
            # 构建下一层级的图（社区作为超节点）
            if level < max_levels - 1:
                next_graph = self._aggregate_graph(current_graph, level_communities)
                if next_graph.number_of_nodes() <= 1:
                    break
                current_graph = next_graph
        
        log.info(f"Detected {len(all_communities)} communities across {level + 1} levels")
        return all_communities
    
    def _aggregate_graph(
        self,
        graph: nx.Graph,
        communities: List[Community]
    ) -> nx.Graph:
        """
        聚合图：将社区转换为超节点
        
        Args:
            graph: 原始图
            communities: 社区列表
            
        Returns:
            聚合后的图
        """
        # 创建节点到社区的映射
        node_to_comm = {}
        for comm in communities:
            for entity in comm.entities:
                node_to_comm[entity] = comm.id
        
        # 构建新图
        agg_graph = nx.Graph()
        
        # 添加社区间的边
        for u, v, data in graph.edges(data=True):
            comm_u = node_to_comm.get(u)
            comm_v = node_to_comm.get(v)
            
            if comm_u and comm_v and comm_u != comm_v:
                # 跨社区的边
                if agg_graph.has_edge(comm_u, comm_v):
                    agg_graph[comm_u][comm_v]['weight'] += data.get('weight', 1.0)
                else:
                    agg_graph.add_edge(comm_u, comm_v, weight=data.get('weight', 1.0))
        
        return agg_graph


# 全局检测器实例
community_detector = CommunityDetector()