"""Community detection over the undirected weighted graph projection."""
from typing import List

import networkx as nx

from config import settings
from models.graph import KnowledgeGraph
from models.schemas import Community
from utils.logger import log


class CommunityDetector:
    def __init__(self, resolution: float | None = None, seed: int | None = None):
        self.resolution = resolution or settings.LEIDEN_RESOLUTION
        self.seed = settings.LEIDEN_SEED if seed is None else seed

    def detect_communities(self, kg: KnowledgeGraph, algorithm: str | None = None) -> List[Community]:
        selected = algorithm or settings.COMMUNITY_ALGORITHM
        if selected != "louvain":
            raise ValueError(f"Unsupported community algorithm: {selected}")

        graph = kg.to_undirected_weighted()
        if graph.number_of_nodes() == 0:
            return []
        if graph.number_of_edges() == 0:
            groups = [{node} for node in sorted(graph.nodes())]
        else:
            groups = nx.community.louvain_communities(
                graph,
                weight="weight",
                resolution=self.resolution,
                seed=self.seed,
            )

        communities = [
            Community(
                id=f"community_{index}",
                level=0,
                entities=sorted(group),
                size=len(group),
            )
            for index, group in enumerate(groups)
        ]
        log.info("Detected {} communities with Louvain", len(communities))
        return communities


community_detector = CommunityDetector()
