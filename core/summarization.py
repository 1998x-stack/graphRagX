"""
社区摘要生成
使用 LLM 为每个社区生成自然语言摘要
"""
from typing import List
from models.schemas import Community
from models.graph import KnowledgeGraph
from services.llm_service import llm_service
from prompts.summary_prompts import create_community_summary_prompt
from utils.logger import log, log_exception


class CommunitySummarizer:
    """社区摘要生成器"""
    
    def __init__(self):
        """初始化摘要生成器"""
        log.info("CommunitySummarizer initialized")
    
    async def summarize_community(
        self,
        community: Community,
        kg: KnowledgeGraph
    ) -> str:
        """
        为单个社区生成摘要
        
        Args:
            community: 社区对象
            kg: 知识图谱（用于获取实体和关系详情）
            
        Returns:
            社区摘要文本
        """
        try:
            log.info(f"Generating summary for community: {community.id}")
            
            # 获取社区中的实体详情
            entities = []
            for entity_name in community.entities:
                entity = kg.get_entity(entity_name)
                if entity:
                    entities.append(entity.dict())
            
            # 获取社区内部的关系
            relations = []
            for relation in kg.relations:
                if (relation.source in community.entities and 
                    relation.target in community.entities):
                    relations.append(relation.dict())
            
            # 构建提示词
            prompt = create_community_summary_prompt(
                community_id=community.id,
                entities=entities,
                relations=relations,
                level=community.level
            )
            
            # 调用 LLM 生成摘要
            summary = await llm_service.generate(
                prompt=prompt,
                task=f"community_summary_{community.id}",
                save_response=True
            )
            
            log.info(f"Generated summary for {community.id} (length: {len(summary)})")
            return summary.strip()
            
        except Exception as e:
            log_exception(e, f"summarize_community({community.id})")
            return f"Failed to generate summary for community {community.id}"
    
    async def summarize_communities(
        self,
        communities: List[Community],
        kg: KnowledgeGraph
    ) -> List[Community]:
        """
        批量生成社区摘要
        
        Args:
            communities: 社区列表
            kg: 知识图谱
            
        Returns:
            包含摘要的社区列表
        """
        log.info(f"Generating summaries for {len(communities)} communities")
        
        # 使用并发控制
        from utils.concurrency import concurrency_controller
        
        # 创建任务列表
        async def summarize_single(comm):
            summary = await self.summarize_community(comm, kg)
            comm.summary = summary
            return comm
        
        # 并发执行
        summarized_communities = await concurrency_controller.map_async(
            func=summarize_single,
            items=communities,
            return_exceptions=False
        )
        
        log.info(f"Generated {len(summarized_communities)} community summaries")
        return summarized_communities


# 全局摘要生成器实例
community_summarizer = CommunitySummarizer()