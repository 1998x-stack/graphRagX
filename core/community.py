"""Hierarchical community detection for the graph's weighted projection."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Set

import networkx as nx

from config import settings
from models.graph import KnowledgeGraph
from models.schemas import Community
from utils.logger import log


class CommunityDetector:
    def __init__(
        self,
        resolution: float | None = None,
        seed: int | None = None,
        max_levels: int | None = None,
        max_cluster_size: int | None = None,
        resolution_multiplier: float | None = None,
    ):
        self.resolution = resolution or settings.LEIDEN_RESOLUTION
        self.seed = settings.LEIDEN_SEED if seed is None else seed
        self.max_levels = max_levels or settings.COMMUNITY_MAX_LEVELS
        self.max_cluster_size = max_cluster_size or settings.COMMUNITY_MAX_CLUSTER_SIZE
        self.resolution_multiplier = (
            resolution_multiplier or settings.COMMUNITY_RESOLUTION_MULTIPLIER
        )

    def _partition_louvain(self, graph: nx.Graph, resolution: float) -> List[Set[str]]:
        if graph.number_of_edges() == 0:
            return [{node} for node in sorted(graph.nodes())]
        groups = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
            seed=self.seed,
        )
        return [set(group) for group in groups]

    def _partition_leiden(self, graph: nx.Graph, resolution: float) -> List[Set[str]]:
        try:
            import igraph as ig
        except ImportError as exc:
            raise RuntimeError(
                "COMMUNITY_ALGORITHM=leiden requires python-igraph; "
                "install requirements-leiden.txt"
            ) from exc

        nodes = sorted(graph.nodes())
        if not nodes:
            return []
        if graph.number_of_edges() == 0:
            return [{node} for node in nodes]

        index = {name: position for position, name in enumerate(nodes)}
        edge_pairs = list(graph.edges())
        edges = [(index[source], index[target]) for source, target in edge_pairs]
        weights = [
            float(graph[source][target].get("weight", 1.0))
            for source, target in edge_pairs
        ]
        ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
        ig_graph.es["weight"] = weights

        ig.set_random_number_generator(random.Random(self.seed))
        try:
            clustering = ig_graph.community_leiden(
                objective_function="modularity",
                weights="weight",
                resolution=resolution,
                n_iterations=settings.LEIDEN_MAX_ITERATIONS,
            )
        finally:
            ig.set_random_number_generator(None)

        return [
            {nodes[position] for position in cluster}
            for cluster in clustering
        ]

    def _partition_graph(
        self,
        graph: nx.Graph,
        resolution: float,
        algorithm: str,
    ) -> List[Set[str]]:
        if graph.number_of_nodes() == 0:
            return []
        if algorithm == "louvain":
            groups = self._partition_louvain(graph, resolution)
        elif algorithm == "leiden":
            groups = self._partition_leiden(graph, resolution)
        else:
            raise ValueError(f"Unsupported community algorithm: {algorithm}")

        normalized = [set(group) for group in groups if group]
        normalized.sort(key=lambda group: tuple(sorted(group)))
        covered: set[str] = set()
        for group in normalized:
            if covered.intersection(group):
                raise ValueError("Community partition contains overlapping groups")
            covered.update(group)
        if covered != set(graph.nodes()):
            raise ValueError("Community partition does not cover every graph node")
        return normalized

    def detect_communities(
        self,
        kg: KnowledgeGraph,
        algorithm: str | None = None,
    ) -> List[Community]:
        """Build a broad-to-fine hierarchy."""
        selected = algorithm or settings.COMMUNITY_ALGORITHM
        graph = kg.to_undirected_weighted()
        if graph.number_of_nodes() == 0:
            return []

        communities: List[Community] = []
        counters: Dict[int, int] = defaultdict(int)

        def add_partition(
            subgraph: nx.Graph,
            parent_id: str | None,
            level: int,
            resolution: float,
            prepared_groups: List[Set[str]] | None = None,
        ) -> None:
            groups = prepared_groups if prepared_groups is not None else self._partition_graph(
                subgraph, resolution, selected
            )
            for group in groups:
                community_id = f"L{level}_C{counters[level]}"
                counters[level] += 1
                communities.append(
                    Community(
                        id=community_id,
                        level=level,
                        parent_id=parent_id,
                        entities=sorted(group),
                        size=len(group),
                    )
                )

                if (
                    level + 1 >= self.max_levels
                    or len(group) <= self.max_cluster_size
                    or len(group) <= 1
                ):
                    continue

                child_graph = graph.subgraph(group).copy()
                child_groups = self._partition_graph(
                    child_graph,
                    resolution * self.resolution_multiplier,
                    selected,
                )
                if len(child_groups) <= 1:
                    continue

                add_partition(
                    child_graph,
                    community_id,
                    level + 1,
                    resolution * self.resolution_multiplier,
                    prepared_groups=child_groups,
                )

        add_partition(graph, None, 0, self.resolution)
        log.info(
            "Detected {} communities across {} level(s) with {}",
            len(communities),
            1 + max(community.level for community in communities),
            selected,
        )
        return communities

    @staticmethod
    def project_level(
        communities: List[Community], requested_level: int
    ) -> List[Community]:
        """Select a complete hierarchy cut, retaining unsplit ancestors.

        A pure exact-level filter loses nodes whenever a branch terminates early.
        This projection matches the C0-C3 community semantics in Edge et al.
        """
        if requested_level < 0:
            raise ValueError("requested_level must be non-negative")
        by_id = {community.id: community for community in communities}
        children: Dict[str, List[Community]] = defaultdict(list)
        roots: List[Community] = []
        for community in communities:
            if community.parent_id and community.parent_id in by_id:
                children[community.parent_id].append(community)
            else:
                roots.append(community)

        projected: List[Community] = []
        pending = list(reversed(sorted(roots, key=lambda item: item.id)))
        visited: set[str] = set()
        while pending:
            parent = pending.pop()
            if parent.id in visited:
                raise ValueError("Community hierarchy contains a cycle or duplicate")
            visited.add(parent.id)
            eligible_children = sorted(
                (child for child in children[parent.id] if child.level <= requested_level),
                key=lambda item: item.id,
            )
            if eligible_children:
                pending.extend(reversed(eligible_children))
            else:
                projected.append(parent)
        return sorted(projected, key=lambda item: (item.level, item.id))

    @staticmethod
    def available_levels(communities: List[Community]) -> List[int]:
        return sorted({community.level for community in communities})

    @staticmethod
    def choose_level(communities: List[Community], requested: int) -> int:
        levels = CommunityDetector.available_levels(communities)
        if not levels:
            return 0
        if requested in levels:
            return requested
        lower_or_equal = [level for level in levels if level <= requested]
        return max(lower_or_equal) if lower_or_equal else min(levels)


community_detector = CommunityDetector()
