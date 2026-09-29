"""In-memory knowledge graph preserving relation direction and type."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from models.schemas import Community, Entity, GraphData, Relation
from utils.logger import log


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.communities: List[Community] = []
        self._relation_index: Dict[Tuple[str, str, str], Relation] = {}

    @staticmethod
    def _merge_text(left: str, right: str) -> str:
        parts = [part.strip() for part in (left, right) if part and part.strip()]
        return " | ".join(dict.fromkeys(parts))

    @staticmethod
    def _merge_list(left: List[str], right: List[str]) -> List[str]:
        return list(dict.fromkeys(item for item in left + right if item))

    def add_entity(self, entity: Entity) -> None:
        incoming = entity.model_copy(deep=True)
        incoming.aliases = self._merge_list(incoming.aliases, [incoming.name])
        if incoming.description and not incoming.description_mentions:
            incoming.description_mentions = [incoming.description]

        existing = self.entities.get(incoming.name)
        if existing is None:
            self.entities[incoming.name] = incoming
            self.graph.add_node(incoming.name, **incoming.model_dump())
            return

        if existing.type != incoming.type:
            log.warning(
                "Entity type conflict for '{}': existing={} incoming={}",
                incoming.name,
                existing.type,
                incoming.type,
            )
        existing.description = self._merge_text(existing.description, incoming.description)
        existing.description_mentions = self._merge_list(
            existing.description_mentions,
            incoming.description_mentions,
        )
        existing.aliases = self._merge_list(existing.aliases, incoming.aliases)
        existing.mention_count += incoming.mention_count
        existing.source_chunk_ids = sorted(
            set(existing.source_chunk_ids + incoming.source_chunk_ids)
        )
        self.graph.nodes[incoming.name].update(existing.model_dump())

    def add_relation(self, relation: Relation) -> None:
        if relation.source not in self.entities or relation.target not in self.entities:
            log.warning(
                "Skipping relation {} -> {} (missing entity)",
                relation.source,
                relation.target,
            )
            return

        incoming = relation.model_copy(deep=True)
        if incoming.description and not incoming.description_mentions:
            incoming.description_mentions = [incoming.description]

        key = (incoming.source, incoming.target, incoming.relation_type)
        existing = self._relation_index.get(key)
        if existing is None:
            self._relation_index[key] = incoming
            self.relations.append(incoming)
            self.graph.add_edge(
                incoming.source,
                incoming.target,
                key=incoming.relation_type,
                weight=incoming.weight,
                relation_type=incoming.relation_type,
                description=incoming.description,
                mention_count=incoming.mention_count,
            )
            return

        existing.weight += incoming.weight
        existing.mention_count += incoming.mention_count
        existing.description = self._merge_text(existing.description, incoming.description)
        existing.description_mentions = self._merge_list(
            existing.description_mentions,
            incoming.description_mentions,
        )
        existing.source_chunk_ids = sorted(
            set(existing.source_chunk_ids + incoming.source_chunk_ids)
        )
        edge = self.graph[existing.source][existing.target][existing.relation_type]
        edge.update(
            weight=existing.weight,
            description=existing.description,
            mention_count=existing.mention_count,
        )

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
        return next(
            (
                community
                for community in self.communities
                if entity_name in community.entities
            ),
            None,
        )

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
            "entity_mentions": sum(
                entity.mention_count for entity in self.entities.values()
            ),
            "relation_mentions": sum(
                relation.mention_count for relation in self.relations
            ),
            "avg_degree": (
                sum(dict(projection.degree()).values()) / node_count
            ) if node_count else 0.0,
            "density": nx.density(projection),
            "connected_components": (
                nx.number_connected_components(projection)
                if node_count
                else 0
            ),
        }
