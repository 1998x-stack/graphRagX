"""
索引构建工作流
使用 LangGraph 实现完整的索引流程
"""
from typing import TypedDict, List
from langgraph.graph import Graph, END
from models.schemas import IndexingState, TextChunk, ExtractionResult, GraphData
from core.chunking import text_chunker
from core.extraction import entity_relation_extractor
from core.graph_builder import graph_builder
from core.community import community_detector
from core.summarization import community_summarizer
from services.storage_service import storage_service
from utils.logger import log, log_exception


class IndexingWorkflow:
    """索引工作流"""
    
    def __init__(self):
        """初始化工作流"""
        self.graph = self._build_graph()
        log.info("IndexingWorkflow initialized")
    
    def _build_graph(self) -> Graph:
        """
        构建 LangGraph 工作流图
        
        节点流程:
        start → chunk_documents → extract_entities → build_graph 
        → detect_communities → generate_summaries → save_graph → end
        """
        workflow = Graph()
        
        # 添加节点
        workflow.add_node("chunk_documents", self.chunk_documents)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("build_graph", self.build_graph_node)
        workflow.add_node("detect_communities", self.detect_communities)
        workflow.add_node("generate_summaries", self.generate_summaries)
        workflow.add_node("save_graph", self.save_graph)
        
        # 添加边（定义流程）
        workflow.set_entry_point("chunk_documents")
        workflow.add_edge("chunk_documents", "extract_entities")
        workflow.add_edge("extract_entities", "build_graph")
        workflow.add_edge("build_graph", "detect_communities")
        workflow.add_edge("detect_communities", "generate_summaries")
        workflow.add_edge("generate_summaries", "save_graph")
        workflow.add_edge("save_graph", END)
        
        return workflow.compile()
    
    async def chunk_documents(self, state: dict) -> dict:
        """
        节点1: 文本分块
        
        Args:
            state: 工作流状态
            
        Returns:
            更新后的状态
        """
        try:
            log.info("=== Step 1: Chunking Documents ===")
            
            documents = state["documents"]
            index_id = state["index_id"]
            
            # 生成文档ID
            doc_ids = [f"{index_id}_doc_{i}" for i in range(len(documents))]
            
            # 分块
            chunks = text_chunker.chunk_documents(documents, doc_ids)
            
            state["chunks"] = chunks
            state["current_step"] = "chunked"
            
            log.info(f"Chunking complete: {len(chunks)} chunks")
            return state
            
        except Exception as e:
            log_exception(e, "chunk_documents")
            state["error"] = str(e)
            state["current_step"] = "error"
            return state
    
    async def extract_entities(self, state: dict) -> dict:
        """
        节点2: 实体关系提取
        
        Args:
            state: 工作流状态
            
        Returns:
            更新后的状态
        """
        try:
            log.info("=== Step 2: Extracting Entities and Relations ===")
            
            chunks = state["chunks"]
            
            # 提取实体和关系
            extraction_results = await entity_relation_extractor.extract_from_chunks(chunks)
            
            state["extraction_results"] = extraction_results
            state["current_step"] = "extracted"
            
            total_entities = sum(len(r.entities) for r in extraction_results)
            total_relations = sum(len(r.relations) for r in extraction_results)
            log.info(f"Extraction complete: {total_entities} entities, {total_relations} relations")
            
            return state
            
        except Exception as e:
            log_exception(e, "extract_entities")
            state["error"] = str(e)
            state["current_step"] = "error"
            return state
    
    async def build_graph_node(self, state: dict) -> dict:
        """
        节点3: 构建知识图谱
        
        Args:
            state: 工作流状态
            
        Returns:
            更新后的状态
        """
        try:
            log.info("=== Step 3: Building Knowledge Graph ===")
            
            extraction_results = state["extraction_results"]
            
            # 构建图谱
            kg = graph_builder.build_graph(extraction_results)
            
            # 转换为 GraphData（可序列化）
            graph_data = kg.to_graph_data()
            state["graph_data"] = graph_data
            state["kg_object"] = kg  # 临时存储（不序列化）
            state["current_step"] = "graph_built"
            
            log.info(f"Graph built: {len(kg.entities)} entities, {len(kg.relations)} relations")
            return state
            
        except Exception as e:
            log_exception(e, "build_graph_node")
            state["error"] = str(e)
            state["current_step"] = "error"
            return state
    
    async def detect_communities(self, state: dict) -> dict:
        """
        节点4: 社区检测
        
        Args:
            state: 工作流状态
            
        Returns:
            更新后的状态
        """
        try:
            log.info("=== Step 4: Detecting Communities ===")
            
            kg = state["kg_object"]
            
            # 社区检测
            communities = community_detector.detect_communities(kg)
            
            # 更新图谱数据
            kg.set_communities(communities)
            state["graph_data"] = kg.to_graph_data()
            state["current_step"] = "communities_detected"
            
            log.info(f"Community detection complete: {len(communities)} communities")
            return state
            
        except Exception as e:
            log_exception(e, "detect_communities")
            state["error"] = str(e)
            state["current_step"] = "error"
            return state
    
    async def generate_summaries(self, state: dict) -> dict:
        """
        节点5: 生成社区摘要
        
        Args:
            state: 工作流状态
            
        Returns:
            更新后的状态
        """
        try:
            log.info("=== Step 5: Generating Community Summaries ===")
            
            kg = state["kg_object"]
            
            # 生成摘要
            communities_with_summaries = await community_summarizer.summarize_communities(
                kg.communities,
                kg
            )
            
            # 更新图谱
            kg.set_communities(communities_with_summaries)
            state["graph_data"] = kg.to_graph_data()
            state["current_step"] = "summaries_generated"
            
            log.info(f"Summary generation complete: {len(communities_with_summaries)} summaries")
            return state
            
        except Exception as e:
            log_exception(e, "generate_summaries")
            state["error"] = str(e)
            state["current_step"] = "error"
            return state
    
    async def save_graph(self, state: dict) -> dict:
        """
        节点6: 保存图谱
        
        Args:
            state: 工作流状态
            
        Returns:
            更新后的状态
        """
        try:
            log.info("=== Step 6: Saving Graph ===")
            
            index_id = state["index_id"]
            graph_data = state["graph_data"]
            
            # 保存图谱
            storage_service.save_graph(index_id, graph_data)
            
            state["current_step"] = "completed"
            
            log.info(f"Graph saved successfully: {index_id}")
            return state
            
        except Exception as e:
            log_exception(e, "save_graph")
            state["error"] = str(e)
            state["current_step"] = "error"
            return state
    
    async def run(
        self,
        index_id: str,
        documents: List[str]
    ) -> dict:
        """
        运行完整的索引工作流
        
        Args:
            index_id: 索引ID
            documents: 文档列表
            
        Returns:
            最终状态
        """
        log.info(f"Starting indexing workflow for index_id={index_id}")
        
        # 初始化状态
        initial_state = {
            "index_id": index_id,
            "documents": documents,
            "chunks": [],
            "extraction_results": [],
            "graph_data": None,
            "current_step": "init",
            "error": None
        }
        
        # 运行工作流
        final_state = await self.graph.ainvoke(initial_state)
        
        log.info(f"Indexing workflow completed: {final_state['current_step']}")
        return final_state


# 全局工作流实例
indexing_workflow = IndexingWorkflow()