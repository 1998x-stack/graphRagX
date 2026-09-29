"""Deterministic graph pruning with auditable before/after statistics."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np

from config import settings
from models.graph import KnowledgeGraph
from utils.logger import log


class GraphPruner:
    def prune(self, kg: KnowledgeGraph) -> Tuple[KnowledgeGraph, Dict[str, Any]]:
        before = kg.get_statistics()
        if not settings.GRAPH_PRUNING_ENABLED or not kg.entities:
            return kg, {
                "enabled": settings.GRAPH_PRUNING_ENABLED,
                "before": before,
                "after": before,
                "removed_entities": 0,
                "removed_relations": 0,
            }

        projection = kg.to_undirected_weighted()
        degrees = dict(projection.degree())
        frequencies = {
            name: max(
                entity.mention_count,
                len(set(entity.source_chunk_ids)),
            )
            for name, entity in kg.entities.items()
        }

        freq_values = np.asarray(list(frequencies.values()), dtype=np.float64)
        degree_values = np.asarray(list(degrees.values()), dtype=np.float64)
        max_freq = None
        max_degree = None
        if (
            settings.PRUNE_MAX_NODE_FREQ_STD is not None
            and len(freq_values) > 1
        ):
            max_freq = float(
                freq_values.mean()
                + settings.PRUNE_MAX_NODE_FREQ_STD * freq_values.std()
            )
        if (
            settings.PRUNE_MAX_NODE_DEGREE_STD is not None
            and len(degree_values) > 1
        ):
            max_degree = float(
                degree_values.mean()
                + settings.PRUNE_MAX_NODE_DEGREE_STD * degree_values.std()
            )

        keep_nodes = set()
        removed_reasons: Dict[str, list[str]] = {}
        for name in sorted(kg.entities):
            reasons = []
            frequency = frequencies[name]
            degree = degrees.get(name, 0)
            if frequency < settings.PRUNE_MIN_NODE_FREQ:
                reasons.append("min_node_freq")
            if max_freq is not None and frequency > max_freq:
                reasons.append("max_node_freq_std")
            if degree < settings.PRUNE_MIN_NODE_DEGREE:
                reasons.append("min_node_degree")
            if max_degree is not None and degree > max_degree:
                reasons.append("max_node_degree_std")
            if (
                settings.PRUNE_REMOVE_EGO_NODES
                and projection.number_of_nodes() > 2
                and degree >= projection.number_of_nodes() - 1
            ):
                reasons.append("ego_node")

            if reasons:
                removed_reasons[name] = reasons
            else:
                keep_nodes.add(name)

        relation_weights = np.asarray(
            [relation.weight for relation in kg.relations],
            dtype=np.float64,
        )
        edge_threshold = 0.0
        if len(relation_weights):
            edge_threshold = float(
                np.percentile(
                    relation_weights,
                    settings.PRUNE_MIN_EDGE_WEIGHT_PCT,
                )
            )

        candidate_edges = [
            relation
            for relation in kg.relations
            if relation.source in keep_nodes
            and relation.target in keep_nodes
            and relation.weight >= edge_threshold
        ]

        if settings.PRUNE_LCC_ONLY and keep_nodes:
            filtered = nx.Graph()
            filtered.add_nodes_from(keep_nodes)
            filtered.add_edges_from(
                (relation.source, relation.target)
                for relation in candidate_edges
            )
            components = list(nx.connected_components(filtered))
            if components:
                components.sort(
                    key=lambda component: (-len(component), tuple(sorted(component)))
                )
                keep_nodes = set(components[0])
                candidate_edges = [
                    relation
                    for relation in candidate_edges
                    if relation.source in keep_nodes
                    and relation.target in keep_nodes
                ]

        pruned = KnowledgeGraph()
        for name in sorted(keep_nodes):
            pruned.add_entity(kg.entities[name])
        for relation in candidate_edges:
            pruned.add_relation(relation)

        after = pruned.get_statistics()
        report = {
            "enabled": True,
            "before": before,
            "after": after,
            "edge_weight_threshold": edge_threshold,
            "removed_entities": len(kg.entities) - len(pruned.entities),
            "removed_relations": len(kg.relations) - len(pruned.relations),
            "removed_entity_reasons": removed_reasons,
        }
        log.info("Graph pruning report: {}", report)
        return pruned, report


graph_pruner = GraphPruner()
