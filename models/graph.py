"""In-memory knowledge graph preserving relation direction and type."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from models.schemas import Community, Entity, GraphData, Relation
from utils.logger import log


class KnowledgeGraph:
    def __init__(self):
        # MultiDiGraph preserves direction and multiple relation types between the
        # same entity pair. Community detection uses a weighted undirected view.
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.communities: List[Community] = []
        self._relation_index: Dict[Tuple[str, str, str], Relation] = {}

    @staticmethod
    def _merge_text(left: str, right: str) -> str:
        parts = [part.strip() for part in (left, right) if part and part.strip()]
        deduped = list(dict.fromkeys(parts))
        return " | ".join(deduped)

    def add_entity(self, entity: Entity) -> None:
        existing = self.entities.get(entity.name)
        if existing is None:
            cloned = entity.model_copy(deep=True)
            self.entities[entity.name] = cloned
            self.graph.add_node(entity.name, **cloned.model_dump())
            return

        if existing.type != entity.type:
            log.warning(
                "Entity type conflict for '{}': existing={} incoming={}",
                entity.name,
                existing.type,
                entity.type,
            )
        existing.description = self._merge_text(existing.description, entity.description)
        existing.source_chunk_ids = sorted(set(existing.source_chunk_ids + entity.source_chunk_ids))
        self.graph.nodes[entity.name].update(existing.model_dump())

    def add_relation(self, relation: Relation) -> None:
        if relation.source not in self.entities or relation.target not in self.entities:
            log.warning(
                "Skipping relation {} -> {} (missing entity)",
                relation.source,
                relation.target,
            )
            return

        key = (relation.source, relation.target, relation.relation_type)
        existing = self._relation_index.get(key)
        if existing is None:
            cloned = relation.model_copy(deep=True)
            self._relation_index[key] = cloned
            self.relations.append(cloned)
            self.graph.add_edge(
                cloned.source,
                cloned.target,
                key=cloned.relation_type,
                weight=cloned.weight,
                relation_type=cloned.relation_type,
                description=cloned.description,
            )
            return

        existing.weight += relation.weight
        existing.description = self._merge_text(existing.description, relation.description)
        existing.source_chunk_ids = sorted(set(existing.source_chunk_ids + relation.source_chunk_ids))
        edge = self.graph[existing.source][existing.target][existing.relation_type]
        edge.update(weight=existing.weight, description=existing.description)

    def get_entity(self, name: str) -> Optional[Entity]:
        return self.entities.get(name)

    def get_neighbors(self, entity_name: str, depth: int = 1) -> Set[str]:
        if entity_name not in self.graph or depth <= 0:
            return set()
        visited = {entity_name}
        frontier = {entity_name}
        for _ in range(depth):
            next_frontier: Set[str] = set()
            for node in frontier:
                next_frontier.update(self.graph.successors(node))
                next_frontier.update(self.graph.predecessors(node))
            next_frontier -= visited
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        visited.discard(entity_name)
        return visited

    def get_subgraph(self, entity_names: List[str]) -> nx.MultiDiGraph:
        return self.graph.subgraph(entity_names).copy()

    def to_undirected_weighted(self) -> nx.Graph:
        projection = nx.Graph()
        projection.add_nodes_from(self.graph.nodes())
        for source, target, data in self.graph.edges(data=True):
            weight = float(data.get("weight", 1.0))
            if projection.has_edge(source, target):
                projection[source][target]["weight"] += weight
            else:
                projection.add_edge(source, target, weight=weight)
        return projection

    def set_communities(self, communities: List[Community]) -> None:
        self.communities = communities

    def get_community_by_entity(self, entity_name: str) -> Optional[Community]:
        return next((community for community in self.communities if entity_name in community.entities), None)

    def to_graph_data(self) -> GraphData:
        return GraphData(
            entities=self.entities,
            relations=self.relations,
            communities=self.communities,
            metadata=self.get_statistics(),
        )

    @classmethod
    def from_graph_data(cls, graph_data: GraphData) -> "KnowledgeGraph":
        kg = cls()
        for entity in graph_data.entities.values():
            kg.add_entity(entity)
        for relation in graph_data.relations:
            kg.add_relation(relation)
        kg.set_communities(graph_data.communities)
        return kg

    def get_statistics(self) -> Dict[str, float | int]:
        projection = self.to_undirected_weighted()
        node_count = projection.number_of_nodes()
        return {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "num_communities": len(self.communities),
            "avg_degree": (sum(dict(projection.degree()).values()) / node_count) if node_count else 0.0,
            "density": nx.density(projection),
            "connected_components": nx.number_connected_components(projection) if node_count else 0,
        }
