"""
实体和关系提取核心算法
使用 LLM 从文本中提取结构化知识
"""
from typing import List
from models.schemas import Entity, Relation, ExtractionResult, TextChunk
from services.llm_service import llm_service
from prompts.extraction_prompts import (
    create_extraction_prompt,
    validate_extraction_result,
    REQUIRED_ENTITY_KEYS,
    REQUIRED_RELATION_KEYS
)
from utils.json_extractor import extract_json
from utils.logger import log, log_exception
from config import settings


class EntityRelationExtractor:
    """实体关系提取器"""
    
    def __init__(self):
        """初始化提取器"""
        self.max_entities = settings.MAX_ENTITIES_PER_CHUNK
        self.max_relations = settings.MAX_RELATIONS_PER_CHUNK
        self.max_retry = settings.MAX_RETRY
        
        log.info("EntityRelationExtractor initialized")
    
    async def extract_from_chunk(
        self, chunk: TextChunk, retry_count: int = 0
    ) -> ExtractionResult:
        """Extract and validate a chunk; never disguise LLM failure as no facts.

        A genuinely empty, correctly structured extraction is valid. Transport,
        JSON, schema and Pydantic failures are retried iteratively and surfaced.
        """
        prompt = create_extraction_prompt(
            text=chunk.text,
            max_entities=self.max_entities,
            max_relations=self.max_relations,
        )
        for attempt in range(retry_count, self.max_retry + 1):
            try:
                response = await llm_service.generate(
                    prompt=prompt,
                    task="entity_extraction",
                    save_response=True,
                )
                payload = extract_json(response)
                if not isinstance(payload, dict):
                    raise ValueError("No valid JSON object in extraction response")
                valid, error_message = validate_extraction_result(payload)
                if not valid:
                    raise ValueError(f"Invalid entity/relation structure: {error_message}")
                if (
                    len(payload["entities"]) > self.max_entities
                    or len(payload["relations"]) > self.max_relations
                ):
                    raise ValueError("Extraction exceeded configured entity/relation limits")

                entities = [
                    Entity(
                        name=item["name"],
                        type=item["type"],
                        description=item["description"],
                        source_chunk_ids=[chunk.id],
                    )
                    for item in payload["entities"]
                ]
                relations = [
                    Relation(
                        source=item["source"],
                        target=item["target"],
                        relation_type=item["relation_type"],
                        description=item["description"],
                        weight=1.0,
                        source_chunk_ids=[chunk.id],
                    )
                    for item in payload["relations"]
                ]
                log.info(
                    "Extracted {} entities and {} relations from chunk {}",
                    len(entities), len(relations), chunk.id,
                )
                return ExtractionResult(
                    entities=entities,
                    relations=relations,
                    chunk_id=chunk.id,
                    raw_response=response,
                )
            except Exception as exc:
                log_exception(exc, f"extract_from_chunk({chunk.id}) attempt={attempt + 1}")
                if attempt >= self.max_retry:
                    raise RuntimeError(
                        f"Extraction failed for chunk {chunk.id} after "
                        f"{self.max_retry + 1} attempts"
                    ) from exc
                log.warning(
                    "Retrying extraction for chunk {} (next attempt {})",
                    chunk.id, attempt + 2,
                )
        raise RuntimeError("Unexpected extraction retry state")

    async def extract_from_chunks(
        self,
        chunks: List[TextChunk]
    ) -> List[ExtractionResult]:
        """
        批量提取多个文本块
        
        Args:
            chunks: 文本块列表
            
        Returns:
            提取结果列表
        """
        log.info(f"Starting extraction from {len(chunks)} chunks")
        
        # 使用并发控制器
        from utils.concurrency import concurrency_controller
        
        results = await concurrency_controller.map_async(
            func=self.extract_from_chunk,
            items=chunks,
            return_exceptions=False
        )
        
        total_entities = sum(len(r.entities) for r in results)
        total_relations = sum(len(r.relations) for r in results)
        
        log.info(
            f"Extraction complete: {total_entities} total entities, "
            f"{total_relations} total relations from {len(chunks)} chunks"
        )
        
        return results


# 全局提取器实例
entity_relation_extractor = EntityRelationExtractor()