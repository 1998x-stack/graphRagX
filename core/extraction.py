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
        self,
        chunk: TextChunk,
        retry_count: int = 0
    ) -> ExtractionResult:
        """
        从单个文本块提取实体和关系
        
        Args:
            chunk: 文本块
            retry_count: 当前重试次数
            
        Returns:
            提取结果
        """
        try:
            # 构建提示词
            prompt = create_extraction_prompt(
                text=chunk.text,
                max_entities=self.max_entities,
                max_relations=self.max_relations
            )
            
            # 调用 LLM
            log.info(f"Extracting from chunk: {chunk.id}")
            response = await llm_service.generate(
                prompt=prompt,
                task="entity_extraction",
                save_response=True
            )
            
            # 提取 JSON
            result_dict = extract_json(response)
            
            if result_dict is None:
                log.error(f"Failed to extract JSON from LLM response for chunk {chunk.id}")
                if retry_count < self.max_retry:
                    log.info(f"Retrying extraction for chunk {chunk.id} (attempt {retry_count + 1})")
                    return await self.extract_from_chunk(chunk, retry_count + 1)
                else:
                    # 返回空结果
                    return ExtractionResult(
                        entities=[],
                        relations=[],
                        chunk_id=chunk.id,
                        raw_response=response
                    )
            
            # 验证格式
            is_valid, error_msg = validate_extraction_result(result_dict)
            if not is_valid:
                log.error(f"Invalid extraction result for chunk {chunk.id}: {error_msg}")
                if retry_count < self.max_retry:
                    log.info(f"Retrying extraction for chunk {chunk.id} (attempt {retry_count + 1})")
                    return await self.extract_from_chunk(chunk, retry_count + 1)
                else:
                    result_dict = {"entities": [], "relations": []}
            
            # 转换为 Pydantic 模型
            entities = []
            for entity_dict in result_dict.get("entities", []):
                try:
                    entity = Entity(
                        name=entity_dict["name"],
                        type=entity_dict["type"],
                        description=entity_dict["description"],
                        source_chunk_ids=[chunk.id]
                    )
                    entities.append(entity)
                except Exception as e:
                    log.warning(f"Failed to parse entity: {entity_dict}, error: {e}")
            
            relations = []
            for relation_dict in result_dict.get("relations", []):
                try:
                    relation = Relation(
                        source=relation_dict["source"],
                        target=relation_dict["target"],
                        relation_type=relation_dict["relation_type"],
                        description=relation_dict["description"],
                        weight=1.0,
                        source_chunk_ids=[chunk.id]
                    )
                    relations.append(relation)
                except Exception as e:
                    log.warning(f"Failed to parse relation: {relation_dict}, error: {e}")
            
            log.info(
                f"Extracted from chunk {chunk.id}: "
                f"{len(entities)} entities, {len(relations)} relations"
            )
            
            return ExtractionResult(
                entities=entities,
                relations=relations,
                chunk_id=chunk.id,
                raw_response=response
            )
            
        except Exception as e:
            log_exception(e, f"extract_from_chunk({chunk.id})")
            
            # 重试
            if retry_count < self.max_retry:
                log.info(f"Retrying extraction for chunk {chunk.id} (attempt {retry_count + 1})")
                return await self.extract_from_chunk(chunk, retry_count + 1)
            else:
                # 返回空结果
                return ExtractionResult(
                    entities=[],
                    relations=[],
                    chunk_id=chunk.id,
                    raw_response=""
                )
    
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