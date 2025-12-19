"""
查询工作流
支持 Local Search 和 Global Search
"""
from typing import List, Dict, Any
from langgraph.graph import Graph, END
from models.schemas import QueryState, GraphData
from models.graph import KnowledgeGraph
from services.storage_service import storage_service
from services.llm_service import llm_service
from services.embedding_service import embedding_service
from prompts.summary_prompts import create_local_search_prompt, create_global_search_prompt
from utils.logger import log, log_exception


class QueryWorkflow:
    """查询工作流"""
    
    def __init__(self):
        """初始化工作流"""
        self.local_graph = self._build_local_graph()
        self.global_graph = self._build_global_graph()
        log.info("QueryWorkflow initialized")
    
    def _build_local_graph(self) -> Graph:
        """
        构建 Local Search 工作流图
        
        流程: start → load_graph → local_search → generate_answer → end
        """
        workflow = Graph()
        
        workflow.add_node("load_graph", self.load_graph)
        workflow.add_node("local_search", self.local_search)
        workflow.add_node("generate_answer", self.generate_local_answer)
        
        workflow.set_entry_point("load_graph")
        workflow.add_edge("load_graph", "local_search")
        workflow.add_edge("local_search", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def _build_global_graph(self) -> Graph:
        """
        构建 Global Search 工作流图
        
        流程: start → load_graph → global_search → generate_answer → end
        """
        workflow = Graph()
        
        workflow.add_node("load_graph", self.load_graph)
        workflow.add_node("global_search", self.global_search)
        workflow.add_node("generate_answer", self.generate_global_answer)
        
        workflow.set_entry_point("load_graph")
        workflow.add_edge("load_graph", "global_search")
        workflow.add_edge("global_search", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    async def load_graph(self, state: dict) -> dict:
        """
        节点: 加载图谱
        
        Args:
            state: 工作流状态
            
        Returns:
            更新后的状态
        """
        try:
            log.info(f"=== Loading Graph: {state['index_id']} ===")
            
            index_id = state["index_id"]
            
            # 从存储加载图谱
            graph_data = storage_service.load_graph(index_id)
            
            # 重建 KnowledgeGraph 对象
            kg = KnowledgeGraph.from_graph_data(graph_data)
            
            state["graph_data"] = graph_data
            state["kg_object"] = kg
            state["current_step"] = "graph_loaded"
            
            log.info(f"Graph loaded: {len(kg.entities)} entities, {len(kg.communities)} communities")
            return state
            
        except Exception as e:
            log_exception(e, "load_graph")
            state["error"] = str(e)
            state["current_step"] = "error"
            return state
    
    async def local_search(self, state: dict) -> dict:
        """
        节点: Local Search（局部搜索）
        
        策略:
        1. 对查询生成 embedding
        2. 找到最相似的实体
        3. 扩展到邻居节点
        4. 获取相关关系
        
        Args:
            state: 工作流状态
            
        Returns:
            更新后的状态
        """
        try:
            log.info("=== Step: Local Search ===")
            
            query = state["query"]
            kg = state["kg_object"]
            top_k = state.get("top_k", 5)
            
            # 生成查询 embedding
            query_embedding = await embedding_service.embed_text(query)
            
            # 生成所有实体的 embeddings（实际应预先计算并存储）
            entity_texts = [
                f"{entity.name}: {entity.description}"
                for entity in kg.entities.values()
            ]
            entity_embeddings = await embedding_service.embed_texts(entity_texts)
            
            # 找到最相似的实体
            similar_indices = embedding_service.find_most_similar(
                query_embedding,
                entity_embeddings,
                top_k=top_k
            )
            
            entity_list = list(kg.entities.values())
            relevant_entities = [entity_list[i] for i in similar_indices]
            
            # 扩展到邻居（1-hop）
            relevant_entity_names = {e.name for e in relevant_entities}
            for entity_name in list(relevant_entity_names):
                neighbors = kg.get_neighbors(entity_name, depth=1)
                relevant_entity_names.update(neighbors)
            
            # 获取相关关系
            relevant_relations = []
            for relation in kg.relations:
                if (relation.source in relevant_entity_names and 
                    relation.target in relevant_entity_names):
                    relevant_relations.append(relation)
            
            # 构建上下文
            state["relevant_context"] = {
                "entities": [kg.entities[name].dict() for name in relevant_entity_names if name in kg.entities],
                "relations": [r.dict() for r in relevant_relations]
            }
            state["current_step"] = "context_retrieved"
            
            log.info(
                f"Local search complete: {len(relevant_entity_names)} entities, "
                f"{len(relevant_relations)} relations"
            )
            return state
            
        except Exception as e:
            log_exception(e, "local_search")
            state["error"] = str(e)
            state["current_step"] = "error"
            return state
    
    async def global_search(self, state: dict) -> dict:
        """
        节点: Global Search（全局搜索）
        
        策略:
        1. 获取所有社区摘要
        2. 使用社区摘要回答查询
        
        Args:
            state: 工作流状态
            
        Returns:
            更新后的状态
        """
        try:
            log.info("=== Step: Global Search ===")
            
            kg = state["kg_object"]
            
            # 获取所有社区摘要
            community_summaries = []
            for community in kg.communities:
                if community.summary:
                    community_summaries.append({
                        "id": community.id,
                        "summary": community.summary,
                        "size": community.size
                    })
            
            state["relevant_context"] = {
                "community_summaries": community_summaries
            }
            state["current_step"] = "context_retrieved"
            
            log.info(f"Global search complete: {len(community_summaries)} community summaries")
            return state
            
        except Exception as e:
            log_exception(e, "global_search")
            state["error"] = str(e)
            state["current_step"] = "error"
            return state
    
    async def generate_local_answer(self, state: dict) -> dict:
        """
        节点: 生成 Local Search 答案
        
        Args:
            state: 工作流状态
            
        Returns:
            更新后的状态
        """
        try:
            log.info("=== Step: Generating Local Answer ===")
            
            query = state["query"]
            context = state["relevant_context"]
            
            # 构建提示词
            prompt = create_local_search_prompt(
                query=query,
                entities=context["entities"],
                relations=context["relations"]
            )
            
            # 生成答案
            answer = await llm_service.generate(
                prompt=prompt,
                task=f"local_query_answer",
                save_response=True
            )
            
            state["answer"] = answer.strip()
            state["current_step"] = "completed"
            
            log.info(f"Local answer generated (length: {len(answer)})")
            return state
            
        except Exception as e:
            log_exception(e, "generate_local_answer")
            state["error"] = str(e)
            state["current_step"] = "error"
            return state
    
    async def generate_global_answer(self, state: dict) -> dict:
        """
        节点: 生成 Global Search 答案
        
        Args:
            state: 工作流状态
            
        Returns:
            更新后的状态
        """
        try:
            log.info("=== Step: Generating Global Answer ===")
            
            query = state["query"]
            context = state["relevant_context"]
            
            # 构建提示词
            prompt = create_global_search_prompt(
                query=query,
                community_summaries=context["community_summaries"]
            )
            
            # 生成答案
            answer = await llm_service.generate(
                prompt=prompt,
                task=f"global_query_answer",
                save_response=True
            )
            
            state["answer"] = answer.strip()
            state["current_step"] = "completed"
            
            log.info(f"Global answer generated (length: {len(answer)})")
            return state
            
        except Exception as e:
            log_exception(e, "generate_global_answer")
            state["error"] = str(e)
            state["current_step"] = "error"
            return state
    
    async def run(
        self,
        query: str,
        index_id: str,
        mode: str = "local",
        top_k: int = 5
    ) -> dict:
        """
        运行查询工作流
        
        Args:
            query: 查询问题
            index_id: 索引ID
            mode: 查询模式 ('local' or 'global')
            top_k: 返回结果数量
            
        Returns:
            最终状态
        """
        log.info(f"Starting query workflow: mode={mode}, index_id={index_id}")
        
        # 初始化状态
        initial_state = {
            "query": query,
            "index_id": index_id,
            "mode": mode,
            "top_k": top_k,
            "graph_data": None,
            "relevant_context": [],
            "answer": "",
            "current_step": "init",
            "error": None
        }
        
        # 选择工作流
        if mode == "local":
            graph = self.local_graph
        elif mode == "global":
            graph = self.global_graph
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'local' or 'global'")
        
        # 运行工作流
        final_state = await graph.ainvoke(initial_state)
        
        log.info(f"Query workflow completed: {final_state['current_step']}")
        return final_state


# 全局工作流实例
query_workflow = QueryWorkflow()