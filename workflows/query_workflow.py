"""Unified query workflow for local, global, basic, and DRIFT retrieval."""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from config import settings
from core.community import community_detector
from models.graph import KnowledgeGraph
from models.schemas import (
    DriftEvidence,
    DriftFollowUp,
    DriftTrace,
    GraphData,
    QueryMode,
    TextChunk,
)
from prompts.drift_prompts import (
    create_drift_followup_prompt,
    create_drift_primer_prompt,
    create_drift_reduce_prompt,
)
from prompts.summary_prompts import (
    create_basic_search_prompt,
    create_global_map_prompt,
    create_global_reduce_prompt,
    create_local_search_prompt,
)
from services.embedding_service import embedding_service
from services.llm_service import llm_service
from services.storage_service import storage_service
from utils.concurrency import concurrency_controller
from utils.json_extractor import extract_json
from utils.logger import log, log_exception
from utils.token_budget import token_budget


class QueryWorkflowState(TypedDict, total=False):
    query: str
    index_id: str
    mode: QueryMode
    top_k: int
    community_level: int | None
    graph_data: GraphData
    kg_object: KnowledgeGraph
    relevant_context: Dict[str, Any]
    drift_trace: DriftTrace
    answer: str
    current_step: str
    error: str | None


class QueryWorkflow:
    def __init__(self):
        self.graph = self._build_graph()

    @staticmethod
    def _failure(exc: Exception, context: str) -> dict:
        log_exception(exc, context)
        return {"current_step": "error", "error": str(exc)}

    @staticmethod
    def _route_mode(state: QueryWorkflowState) -> str:
        if state.get("error"):
            return "end"
        return state["mode"]

    @staticmethod
    def _route_after_retrieval(state: QueryWorkflowState) -> str:
        return "end" if state.get("error") else "continue"

    def _build_graph(self):
        workflow = StateGraph(QueryWorkflowState)
        workflow.add_node("load_graph", self.load_graph)
        workflow.add_node("local_search", self.local_search)
        workflow.add_node("global_search", self.global_search)
        workflow.add_node("basic_search", self.basic_search)
        workflow.add_node("drift_search", self.drift_search)
        workflow.add_node("generate_local_answer", self.generate_local_answer)
        workflow.add_node("generate_global_answer", self.generate_global_answer)
        workflow.add_node("generate_basic_answer", self.generate_basic_answer)
        workflow.add_node("generate_drift_answer", self.generate_drift_answer)

        workflow.add_edge(START, "load_graph")
        workflow.add_conditional_edges(
            "load_graph",
            self._route_mode,
            {
                "local": "local_search",
                "global": "global_search",
                "basic": "basic_search",
                "drift": "drift_search",
                "end": END,
            },
        )
        for retrieval, generator in (
            ("local_search", "generate_local_answer"),
            ("global_search", "generate_global_answer"),
            ("basic_search", "generate_basic_answer"),
            ("drift_search", "generate_drift_answer"),
        ):
            workflow.add_conditional_edges(
                retrieval,
                self._route_after_retrieval,
                {"continue": generator, "end": END},
            )
            workflow.add_edge(generator, END)
        return workflow.compile()

    async def load_graph(self, state: QueryWorkflowState) -> dict:
        try:
            graph_data = storage_service.load_graph(state["index_id"])
            return {
                "graph_data": graph_data,
                "kg_object": KnowledgeGraph.from_graph_data(graph_data),
                "current_step": "graph_loaded",
                "error": None,
            }
        except Exception as exc:
            return self._failure(exc, "load_graph")

    @staticmethod
    def _stored_embeddings_compatible(graph_data: GraphData) -> bool:
        metadata = graph_data.metadata
        return (
            metadata.get("embedding_provider") == settings.EMBEDDING_PROVIDER
            and metadata.get("embedding_model") == embedding_service.model
            and metadata.get("embedding_dimension") == embedding_service.dimension
        )

    async def _rank_named_vectors(
        self,
        query: str,
        names: List[str],
        vectors_by_name: Dict[str, List[float]],
        fallback_texts: List[str],
        top_k: int,
        use_stored: bool = True,
    ) -> List[tuple[str, float]]:
        if not names:
            return []

        query_vector = await embedding_service.embed_text(query)
        stored_usable = use_stored and all(
            name in vectors_by_name
            and len(vectors_by_name[name]) == len(query_vector)
            for name in names
        )
        if stored_usable:
            vectors = [
                np.asarray(vectors_by_name[name], dtype=np.float32)
                for name in names
            ]
        else:
            vectors = await embedding_service.embed_texts(fallback_texts)

        scored = [
            (
                name,
                embedding_service.cosine_similarity(query_vector, vector),
            )
            for name, vector in zip(names, vectors)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: min(top_k, len(scored))]

    async def _rank_chunks(
        self,
        query: str,
        chunks: List[TextChunk],
        graph_data: GraphData,
        top_k: int,
    ) -> List[TextChunk]:
        if not chunks:
            return []

        query_vector = await embedding_service.embed_text(query)
        stored_usable = self._stored_embeddings_compatible(graph_data) and all(
            chunk.id in graph_data.chunk_embeddings
            and len(graph_data.chunk_embeddings[chunk.id]) == len(query_vector)
            for chunk in chunks
        )
        if stored_usable:
            vectors = [
                np.asarray(graph_data.chunk_embeddings[chunk.id], dtype=np.float32)
                for chunk in chunks
            ]
        else:
            vectors = await embedding_service.embed_texts(
                [chunk.text for chunk in chunks]
            )

        scored = [
            (
                chunk,
                embedding_service.cosine_similarity(query_vector, vector),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return [chunk for chunk, _ in scored[: min(top_k, len(scored))]]

    async def _retrieve_local_context(
        self,
        query: str,
        kg: KnowledgeGraph,
        graph_data: GraphData,
        top_k: int,
        max_tokens: int,
    ) -> Dict[str, Any]:
        entity_names = sorted(kg.entities)
        ranked_entities = await self._rank_named_vectors(
            query,
            entity_names,
            graph_data.entity_embeddings,
            [f"{name}: {kg.entities[name].description}" for name in entity_names],
            min(top_k, settings.MAX_QUERY_TOP_K),
            use_stored=self._stored_embeddings_compatible(graph_data),
        )
        selected = [name for name, _ in ranked_entities]

        relevant_names = set(selected)
        for name in selected:
            relevant_names.update(
                kg.get_neighbors(
                    name,
                    depth=settings.RETRIEVAL_NEIGHBOR_DEPTH,
                )
            )

        entities = [
            kg.entities[name].model_dump()
            for name in sorted(relevant_names)
            if name in kg.entities
        ]
        relations = [
            relation.model_dump()
            for relation in kg.relations
            if relation.source in relevant_names
            and relation.target in relevant_names
        ]

        source_chunk_ids = {
            chunk_id
            for name in relevant_names
            if name in kg.entities
            for chunk_id in kg.entities[name].source_chunk_ids
        }
        chunks_by_id = {chunk.id: chunk for chunk in graph_data.text_chunks}
        source_chunks = [
            chunks_by_id[chunk_id]
            for chunk_id in source_chunk_ids
            if chunk_id in chunks_by_id
        ]
        ranked_chunks = await self._rank_chunks(
            query,
            source_chunks,
            graph_data,
            top_k=settings.MAX_CONTEXT_CHUNKS,
        )
        packed_segments = token_budget.select(
            [(chunk.id, chunk.text) for chunk in ranked_chunks],
            max_tokens,
        )
        packed_text = dict(packed_segments)
        text_units = [
            chunk.model_copy(
                update={"text": packed_text[chunk.id]}
            ).model_dump()
            for chunk in ranked_chunks
            if chunk.id in packed_text
        ]

        return {
            "entities": entities,
            "relations": relations,
            "text_units": text_units,
            "seed_entities": [
                {"name": name, "relevance": score}
                for name, score in ranked_entities
            ],
        }

    async def _retrieve_community_reports(
        self,
        query: str,
        graph_data: GraphData,
        top_k: int,
        community_level: int,
        max_tokens: int,
    ) -> Dict[str, Any]:
        communities = [
            community
            for community in graph_data.communities
            if community.summary
        ]
        chosen_level = community_detector.choose_level(
            communities,
            community_level,
        )
        level_communities = [
            community
            for community in communities
            if community.level == chosen_level
        ]
        ids = [community.id for community in level_communities]
        limit = min(top_k, settings.MAX_QUERY_TOP_K, len(ids))
        ranked = await self._rank_named_vectors(
            query,
            ids,
            graph_data.community_embeddings,
            [community.summary or "" for community in level_communities],
            limit,
            use_stored=self._stored_embeddings_compatible(graph_data),
        )
        by_id = {
            community.id: community
            for community in level_communities
        }
        budgeted = token_budget.select(
            [
                (community_id, by_id[community_id].summary or "")
                for community_id, _ in ranked
            ],
            max_tokens,
        )
        budgeted_text = dict(budgeted)
        reports = [
            {
                "id": community_id,
                "summary": budgeted_text[community_id],
                "size": by_id[community_id].size,
                "level": by_id[community_id].level,
                "parent_id": by_id[community_id].parent_id,
                "relevance": score,
            }
            for community_id, score in ranked
            if community_id in budgeted_text
        ]
        return {
            "community_summaries": reports,
            "community_level": chosen_level,
        }

    async def local_search(self, state: QueryWorkflowState) -> dict:
        try:
            context = await self._retrieve_local_context(
                state["query"],
                state["kg_object"],
                state["graph_data"],
                top_k=state.get("top_k", 5),
                max_tokens=settings.LOCAL_MAX_DATA_TOKENS,
            )
            return {
                "relevant_context": context,
                "current_step": "context_retrieved",
            }
        except Exception as exc:
            return self._failure(exc, "local_search")

    async def global_search(self, state: QueryWorkflowState) -> dict:
        try:
            requested_level = state.get("community_level")
            if requested_level is None:
                requested_level = settings.GLOBAL_COMMUNITY_LEVEL
            context = await self._retrieve_community_reports(
                state["query"],
                state["graph_data"],
                top_k=min(state.get("top_k", 5), settings.GLOBAL_TOP_K),
                community_level=requested_level,
                max_tokens=settings.GLOBAL_MAX_DATA_TOKENS,
            )
            return {
                "relevant_context": context,
                "current_step": "context_retrieved",
            }
        except Exception as exc:
            return self._failure(exc, "global_search")

    @staticmethod
    def _parse_global_map_points(
        raw_response: str,
        source_reports: List[str],
    ) -> List[Dict[str, Any]]:
        data = extract_json(raw_response)
        points: List[Dict[str, Any]] = []
        if isinstance(data, dict) and isinstance(data.get("points"), list):
            for point in data["points"]:
                if not isinstance(point, dict):
                    continue
                description = str(point.get("description", "")).strip()
                if not description:
                    continue
                try:
                    score = float(point.get("score", 0))
                except (TypeError, ValueError):
                    score = 0.0
                points.append(
                    {
                        "description": description,
                        "score": max(0.0, min(100.0, score)),
                        "source_reports": source_reports,
                    }
                )
        if not points and raw_response.strip():
            points.append(
                {
                    "description": raw_response.strip(),
                    "score": 1.0,
                    "source_reports": source_reports,
                }
            )
        return points

    @staticmethod
    def _normalize_question(question: str) -> str:
        return " ".join(question.casefold().split()).rstrip("?.! ")

    @staticmethod
    def _parse_followups(
        items: Any,
        *,
        depth: int,
        parent_id: str | None,
        prefix: str,
    ) -> List[DriftFollowUp]:
        if not isinstance(items, list):
            return []
        followups: List[DriftFollowUp] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            try:
                score = float(item.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            followups.append(
                DriftFollowUp(
                    id=f"{prefix}_{index}",
                    question=question,
                    score=max(0.0, min(100.0, score)),
                    depth=depth,
                    parent_id=parent_id,
                )
            )
        followups.sort(key=lambda item: (-item.score, item.question))
        return followups

    @classmethod
    def _parse_drift_primer(
        cls,
        raw_response: str,
        *,
        prefix: str,
    ) -> tuple[str, List[DriftFollowUp]]:
        data = extract_json(raw_response)
        if not isinstance(data, dict):
            return raw_response.strip(), []
        answer = str(data.get("answer", "")).strip()
        followups = cls._parse_followups(
            data.get("follow_ups"),
            depth=1,
            parent_id=None,
            prefix=prefix,
        )
        return answer, followups

    @classmethod
    def _parse_drift_followup(
        cls,
        raw_response: str,
        *,
        depth: int,
        parent_id: str,
    ) -> tuple[str, float, List[DriftFollowUp]]:
        data = extract_json(raw_response)
        if not isinstance(data, dict):
            return raw_response.strip(), 0.25, []

        answer = str(data.get("answer", "")).strip()
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        followups = cls._parse_followups(
            data.get("follow_ups"),
            depth=depth + 1,
            parent_id=parent_id,
            prefix=f"{parent_id}_q",
        )
        return answer, confidence, followups

    @staticmethod
    def _fold_reports(
        reports: List[Dict[str, Any]],
        folds: int,
    ) -> List[List[Dict[str, Any]]]:
        if not reports:
            return []
        fold_count = min(max(1, folds), len(reports))
        return [
            reports[index::fold_count]
            for index in range(fold_count)
            if reports[index::fold_count]
        ]

    async def drift_search(self, state: QueryWorkflowState) -> dict:
        try:
            requested_level = state.get("community_level")
            if requested_level is None:
                requested_level = settings.DRIFT_COMMUNITY_LEVEL

            primer_context = await self._retrieve_community_reports(
                state["query"],
                state["graph_data"],
                top_k=min(
                    settings.DRIFT_K_FOLLOWUPS * settings.DRIFT_PRIMER_FOLDS,
                    settings.GLOBAL_TOP_K,
                ),
                community_level=requested_level,
                max_tokens=settings.DRIFT_PRIMER_DATA_TOKENS,
            )
            reports = primer_context.get("community_summaries", [])
            folds = self._fold_reports(
                reports,
                settings.DRIFT_PRIMER_FOLDS,
            )

            async def prime_one(
                item: tuple[int, List[Dict[str, Any]]],
            ) -> tuple[str, List[DriftFollowUp]]:
                fold_index, fold_reports = item
                prompt = create_drift_primer_prompt(
                    state["query"],
                    fold_reports,
                )
                raw = await llm_service.generate(
                    prompt,
                    task="drift_primer",
                )
                return self._parse_drift_primer(
                    raw,
                    prefix=f"primer_{fold_index}",
                )

            primer_results = await concurrency_controller.map_async(
                func=prime_one,
                items=list(enumerate(folds)),
                return_exceptions=False,
            ) if folds else []

            primer_answers = [
                answer
                for answer, _ in primer_results
                if answer
            ]
            primer_answer = "\n\n".join(primer_answers)

            generated_followups = [
                followup
                for _, fold_followups in primer_results
                for followup in fold_followups
            ]
            generated_followups.sort(
                key=lambda item: (-item.score, item.question)
            )

            seen_questions = {self._normalize_question(state["query"])}
            initial_queue: List[DriftFollowUp] = []
            for followup in generated_followups:
                normalized = self._normalize_question(followup.question)
                if not normalized or normalized in seen_questions:
                    continue
                seen_questions.add(normalized)
                initial_queue.append(followup)
                if len(initial_queue) >= settings.DRIFT_K_FOLLOWUPS:
                    break

            trace = DriftTrace(
                primer_answer=primer_answer,
                primer_reports=[report["id"] for report in reports],
                follow_ups=list(initial_queue),
                visited_questions=[],
            )
            pending = initial_queue
            evidence: List[DriftEvidence] = []
            actions_executed = 0
            max_depth_reached = 0
            termination_reason = "primer_only" if not pending else ""

            for depth in range(1, settings.DRIFT_N_DEPTH + 1):
                if actions_executed >= settings.DRIFT_MAX_ACTIONS:
                    termination_reason = "max_actions"
                    break

                current = [
                    item
                    for item in pending
                    if item.depth == depth
                ]
                current.sort(key=lambda item: (-item.score, item.question))
                remaining_slots = settings.DRIFT_MAX_ACTIONS - actions_executed
                current = current[: min(settings.DRIFT_K_FOLLOWUPS, remaining_slots)]
                if not current:
                    termination_reason = "no_followups"
                    break

                for item in current:
                    trace.visited_questions.append(item.question)

                async def investigate(
                    followup: DriftFollowUp,
                ) -> tuple[DriftEvidence, List[DriftFollowUp]]:
                    local_context = await self._retrieve_local_context(
                        followup.question,
                        state["kg_object"],
                        state["graph_data"],
                        top_k=settings.DRIFT_LOCAL_TOP_K_ENTITIES,
                        max_tokens=settings.DRIFT_LOCAL_MAX_DATA_TOKENS,
                    )
                    prompt = create_drift_followup_prompt(
                        root_query=state["query"],
                        followup_query=followup.question,
                        entities=local_context.get("entities", []),
                        relations=local_context.get("relations", []),
                        text_units=local_context.get("text_units", []),
                        prior_evidence=[
                            item.model_dump()
                            for item in evidence
                        ],
                    )
                    raw = await llm_service.generate(
                        prompt,
                        task=f"drift_followup_{followup.id}",
                    )
                    answer, confidence, next_followups = self._parse_drift_followup(
                        raw,
                        depth=followup.depth,
                        parent_id=followup.id,
                    )
                    source_ids = [
                        item.get("id")
                        for item in local_context.get("text_units", [])
                        if item.get("id")
                    ]
                    source_ids.extend(
                        f"entity:{item.get('name')}"
                        for item in local_context.get("seed_entities", [])
                        if item.get("name")
                    )
                    return (
                        DriftEvidence(
                            id=followup.id,
                            question=followup.question,
                            answer=answer,
                            confidence=confidence,
                            depth=followup.depth,
                            parent_id=followup.parent_id,
                            source_ids=list(dict.fromkeys(source_ids)),
                        ),
                        next_followups,
                    )

                results = await concurrency_controller.map_async(
                    func=investigate,
                    items=current,
                    return_exceptions=False,
                )

                next_pending: List[DriftFollowUp] = []
                for item_evidence, next_followups in results:
                    evidence.append(item_evidence)
                    actions_executed += 1
                    max_depth_reached = max(
                        max_depth_reached,
                        item_evidence.depth,
                    )
                    if (
                        item_evidence.confidence
                        < settings.DRIFT_EXPANSION_MIN_CONFIDENCE
                        or depth >= settings.DRIFT_N_DEPTH
                    ):
                        continue

                    for candidate in next_followups:
                        normalized = self._normalize_question(candidate.question)
                        if not normalized or normalized in seen_questions:
                            continue
                        seen_questions.add(normalized)
                        next_pending.append(candidate)
                        trace.follow_ups.append(candidate)

                next_pending.sort(
                    key=lambda item: (-item.score, item.question)
                )
                pending = next_pending[: settings.DRIFT_K_FOLLOWUPS]

                if actions_executed >= settings.DRIFT_MAX_ACTIONS:
                    termination_reason = "max_actions"
                    break
                if depth >= settings.DRIFT_N_DEPTH:
                    termination_reason = "max_depth"
                    break
                if not pending:
                    termination_reason = "no_followups"
                    break

            trace.evidence = evidence
            trace.actions_executed = actions_executed
            trace.max_depth_reached = max_depth_reached
            trace.termination_reason = termination_reason or "completed"

            context = {
                **primer_context,
                "drift": trace.model_dump(),
            }
            return {
                "relevant_context": context,
                "drift_trace": trace,
                "current_step": "drift_explored",
            }
        except Exception as exc:
            return self._failure(exc, "drift_search")

    async def basic_search(self, state: QueryWorkflowState) -> dict:
        try:
            graph_data = state["graph_data"]
            chunks = graph_data.text_chunks
            if not chunks:
                return {
                    "relevant_context": {"text_units": []},
                    "current_step": "context_retrieved",
                }
            ranked = await self._rank_chunks(
                state["query"],
                chunks,
                graph_data,
                top_k=min(
                    state.get("top_k", 5),
                    settings.MAX_QUERY_TOP_K,
                ),
            )
            return {
                "relevant_context": {
                    "text_units": [
                        chunk.model_dump()
                        for chunk in ranked
                    ]
                },
                "current_step": "context_retrieved",
            }
        except Exception as exc:
            return self._failure(exc, "basic_search")

    async def generate_local_answer(self, state: QueryWorkflowState) -> dict:
        try:
            context = state["relevant_context"]
            prompt = create_local_search_prompt(
                state["query"],
                context.get("entities", []),
                context.get("relations", []),
                context.get("text_units", []),
            )
            answer = await llm_service.generate(
                prompt,
                task="local_query_answer",
            )
            return {
                "answer": answer.strip(),
                "current_step": "completed",
            }
        except Exception as exc:
            return self._failure(exc, "generate_local_answer")

    async def generate_global_answer(self, state: QueryWorkflowState) -> dict:
        try:
            context = dict(state["relevant_context"])
            reports = context.get("community_summaries", [])
            if not reports:
                return {
                    "answer": (
                        "No community reports were available at the selected "
                        "hierarchy level."
                    ),
                    "current_step": "completed",
                }

            batches = token_budget.batches(
                [
                    (report["id"], report.get("summary", ""))
                    for report in reports
                ],
                settings.GLOBAL_MAP_BATCH_TOKENS,
            )

            async def map_one(item: tuple[int, list[tuple[str, str]]]):
                batch_index, batch = item
                batch_reports = [
                    {"id": report_id, "summary": summary}
                    for report_id, summary in batch
                ]
                prompt = create_global_map_prompt(
                    state["query"],
                    batch_reports,
                )
                raw = await llm_service.generate(
                    prompt,
                    task=f"global_map_{batch_index}",
                )
                return self._parse_global_map_points(
                    raw,
                    [report_id for report_id, _ in batch],
                )

            mapped = await concurrency_controller.map_async(
                func=map_one,
                items=list(enumerate(batches)),
                return_exceptions=False,
            )
            points = [
                point
                for batch_points in mapped
                for point in batch_points
                if point.get("score", 0) > 0
            ]
            points.sort(
                key=lambda point: (
                    -float(point.get("score", 0)),
                    point.get("description", ""),
                )
            )
            if not points:
                return {
                    "answer": (
                        "The selected community reports did not contain enough "
                        "relevant evidence to answer the question."
                    ),
                    "current_step": "completed",
                }

            point_lookup = {
                f"point_{index}": point
                for index, point in enumerate(points)
            }
            selected_point_text = token_budget.select(
                [
                    (point_id, point["description"])
                    for point_id, point in point_lookup.items()
                ],
                settings.GLOBAL_REDUCE_DATA_TOKENS,
            )
            reduce_points = [
                {
                    **point_lookup[point_id],
                    "description": description,
                }
                for point_id, description in selected_point_text
            ]
            prompt = create_global_reduce_prompt(
                state["query"],
                reduce_points,
            )
            answer = await llm_service.generate(
                prompt,
                task="global_reduce",
            )
            context["map_points"] = reduce_points
            return {
                "answer": answer.strip(),
                "relevant_context": context,
                "current_step": "completed",
            }
        except Exception as exc:
            return self._failure(exc, "generate_global_answer")

    async def generate_drift_answer(self, state: QueryWorkflowState) -> dict:
        try:
            trace = state.get("drift_trace")
            if trace is None:
                raise RuntimeError("DRIFT trace missing after exploration")

            primer_budget = max(
                1,
                settings.DRIFT_REDUCE_DATA_TOKENS // 4,
            )
            primer_answer = token_budget.truncate(
                trace.primer_answer,
                primer_budget,
            )
            remaining_budget = max(
                1,
                settings.DRIFT_REDUCE_DATA_TOKENS
                - token_budget.count(primer_answer),
            )

            evidence_by_id = {
                item.id: item
                for item in trace.evidence
            }
            ranked_evidence = sorted(
                trace.evidence,
                key=lambda item: (
                    -item.confidence,
                    item.depth,
                    item.id,
                ),
            )
            selected = token_budget.select(
                [
                    (
                        item.id,
                        (
                            f"Question: {item.question}\n"
                            f"Answer: {item.answer}\n"
                            f"Sources: {', '.join(item.source_ids)}"
                        ),
                    )
                    for item in ranked_evidence
                ],
                remaining_budget,
            )
            reduce_evidence = []
            for evidence_id, clipped in selected:
                original = evidence_by_id[evidence_id]
                answer_marker = "Answer: "
                answer = clipped
                if answer_marker in clipped:
                    answer = clipped.split(answer_marker, 1)[1]
                    if "\nSources:" in answer:
                        answer = answer.split("\nSources:", 1)[0]
                reduce_evidence.append(
                    {
                        **original.model_dump(),
                        "answer": answer.strip(),
                    }
                )

            prompt = create_drift_reduce_prompt(
                state["query"],
                primer_answer,
                reduce_evidence,
            )
            answer = await llm_service.generate(
                prompt,
                task="drift_reduce",
            )
            context = dict(state.get("relevant_context", {}))
            context["drift_reduce_evidence"] = reduce_evidence
            return {
                "answer": answer.strip(),
                "relevant_context": context,
                "current_step": "completed",
            }
        except Exception as exc:
            return self._failure(exc, "generate_drift_answer")

    async def generate_basic_answer(self, state: QueryWorkflowState) -> dict:
        try:
            prompt = create_basic_search_prompt(
                state["query"],
                state["relevant_context"].get("text_units", []),
            )
            answer = await llm_service.generate(
                prompt,
                task="basic_query_answer",
            )
            return {
                "answer": answer.strip(),
                "current_step": "completed",
            }
        except Exception as exc:
            return self._failure(exc, "generate_basic_answer")

    async def run(
        self,
        query: str,
        index_id: str,
        mode: QueryMode = "local",
        top_k: int = 5,
        community_level: int | None = None,
    ) -> dict:
        initial: QueryWorkflowState = {
            "query": query,
            "index_id": index_id,
            "mode": mode,
            "top_k": top_k,
            "community_level": community_level,
            "relevant_context": {},
            "answer": "",
            "current_step": "init",
            "error": None,
        }
        final_state = await self.graph.ainvoke(initial)
        log.info(
            "Query workflow {}:{} finished at {}",
            index_id,
            mode,
            final_state.get("current_step"),
        )
        return final_state


query_workflow = QueryWorkflow()
