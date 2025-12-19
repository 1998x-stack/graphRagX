"""
数据模型定义
使用 Pydantic 进行数据验证
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ==================== 基础数据模型 ====================

class Entity(BaseModel):
    """实体模型"""
    name: str = Field(..., description="实体名称（首字母大写）")
    type: str = Field(..., description="实体类型")
    description: str = Field(..., description="实体描述")
    source_chunk_ids: List[str] = Field(default_factory=list, description="来源文本块ID列表")
    
    def __hash__(self):
        return hash((self.name, self.type))
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.name == other.name and self.type == other.type


class Relation(BaseModel):
    """关系模型"""
    source: str = Field(..., description="源实体名称")
    target: str = Field(..., description="目标实体名称")
    relation_type: str = Field(..., description="关系类型")
    description: str = Field(..., description="关系描述")
    weight: float = Field(default=1.0, description="关系权重")
    source_chunk_ids: List[str] = Field(default_factory=list, description="来源文本块ID列表")
    
    def __hash__(self):
        return hash((self.source, self.target, self.relation_type))
    
    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        return (self.source == other.source and 
                self.target == other.target and 
                self.relation_type == other.relation_type)


class Community(BaseModel):
    """社区模型"""
    id: str = Field(..., description="社区ID")
    level: int = Field(..., description="社区层级")
    entities: List[str] = Field(default_factory=list, description="包含的实体名称列表")
    summary: Optional[str] = Field(None, description="社区摘要")
    parent_id: Optional[str] = Field(None, description="父社区ID")
    size: int = Field(default=0, description="社区大小")


class TextChunk(BaseModel):
    """文本块模型"""
    id: str = Field(..., description="文本块唯一ID")
    text: str = Field(..., description="文本内容")
    doc_id: str = Field(..., description="所属文档ID")
    chunk_index: int = Field(..., description="在文档中的索引")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


# ==================== 提取结果模型 ====================

class ExtractionResult(BaseModel):
    """实体/关系提取结果"""
    entities: List[Entity] = Field(default_factory=list, description="提取的实体列表")
    relations: List[Relation] = Field(default_factory=list, description="提取的关系列表")
    chunk_id: str = Field(..., description="来源文本块ID")
    raw_response: str = Field(..., description="LLM原始响应")


class GraphData(BaseModel):
    """图谱数据"""
    entities: Dict[str, Entity] = Field(default_factory=dict, description="实体字典 {name: Entity}")
    relations: List[Relation] = Field(default_factory=list, description="关系列表")
    communities: List[Community] = Field(default_factory=list, description="社区列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


# ==================== API 请求/响应模型 ====================

class IndexRequest(BaseModel):
    """索引请求"""
    documents: List[str] = Field(..., description="文档内容列表", min_length=1)
    index_id: str = Field(..., description="索引ID", min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class IndexResponse(BaseModel):
    """索引响应"""
    status: str = Field(..., description="状态: success/failed")
    index_id: str = Field(..., description="索引ID")
    num_documents: int = Field(..., description="文档数量")
    num_chunks: int = Field(..., description="文本块数量")
    num_entities: int = Field(..., description="实体数量")
    num_relations: int = Field(..., description="关系数量")
    num_communities: int = Field(..., description="社区数量")
    processing_time: float = Field(..., description="处理时间（秒）")
    error: Optional[str] = Field(None, description="错误信息")


class QueryRequest(BaseModel):
    """查询请求"""
    query: str = Field(..., description="查询问题", min_length=1)
    index_id: str = Field(..., description="索引ID", min_length=1)
    mode: str = Field(default="local", description="查询模式: local/global")
    top_k: int = Field(default=5, description="返回结果数量")


class QueryResponse(BaseModel):
    """查询响应"""
    status: str = Field(..., description="状态: success/failed")
    answer: str = Field(..., description="答案")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="来源信息")
    mode: str = Field(..., description="查询模式")
    processing_time: float = Field(..., description="处理时间（秒）")
    error: Optional[str] = Field(None, description="错误信息")


# ==================== LangGraph 状态模型 ====================

class IndexingState(BaseModel):
    """索引工作流状态"""
    index_id: str
    documents: List[str]
    chunks: List[TextChunk] = Field(default_factory=list)
    extraction_results: List[ExtractionResult] = Field(default_factory=list)
    graph_data: Optional[GraphData] = None
    current_step: str = "init"
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class QueryState(BaseModel):
    """查询工作流状态"""
    query: str
    index_id: str
    mode: str
    graph_data: Optional[GraphData] = None
    relevant_context: List[Dict[str, Any]] = Field(default_factory=list)
    answer: str = ""
    current_step: str = "init"
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True