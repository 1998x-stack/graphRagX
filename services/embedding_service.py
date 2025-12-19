"""
Embedding 服务
用于文本向量化（实体、文本块、社区摘要）
"""
from typing import List
import numpy as np
from config import settings
from utils.logger import log, log_exception


class EmbeddingService:
    """Embedding 服务类"""
    
    def __init__(
        self,
        model: str = None,
        dimension: int = None
    ):
        """
        初始化 Embedding 服务
        
        Args:
            model: 模型名称
            dimension: 向量维度
        """
        self.model = model or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        
        # TODO: 初始化 Embedding 客户端
        # from openai import AsyncOpenAI
        # self.client = AsyncOpenAI(api_key=settings.EMBEDDING_API_KEY)
        
        log.info(f"EmbeddingService initialized: model={self.model}, dim={self.dimension}")
        log.warning("EmbeddingService client NOT implemented - using mock embeddings!")
    
    async def embed_text(self, text: str) -> np.ndarray:
        """
        对单个文本生成嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量（numpy 数组）
        """
        try:
            # TODO: 实现真实的 Embedding API 调用
            # response = await self.client.embeddings.create(
            #     model=self.model,
            #     input=text
            # )
            # embedding = np.array(response.data[0].embedding)
            
            # 临时 Mock 实现：返回随机向量
            embedding = np.random.randn(self.dimension).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # 归一化
            
            log.debug(f"Generated embedding for text (length={len(text)})")
            return embedding
            
        except Exception as e:
            log_exception(e, "embed_text")
            raise
    
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        批量生成嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        try:
            # TODO: 实现批量 API 调用（更高效）
            # response = await self.client.embeddings.create(
            #     model=self.model,
            #     input=texts
            # )
            # embeddings = [np.array(item.embedding) for item in response.data]
            
            # 临时 Mock 实现
            embeddings = []
            for text in texts:
                embedding = await self.embed_text(text)
                embeddings.append(embedding)
            
            log.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            log_exception(e, "embed_texts")
            raise
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度 [-1, 1]
        """
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray],
        top_k: int = 5
    ) -> List[int]:
        """
        查找最相似的向量
        
        Args:
            query_embedding: 查询向量
            embeddings: 候选向量列表
            top_k: 返回前 k 个
            
        Returns:
            最相似向量的索引列表
        """
        similarities = [
            self.cosine_similarity(query_embedding, emb)
            for emb in embeddings
        ]
        
        # 排序并返回 top-k 索引
        top_indices = np.argsort(similarities)[::-1][:top_k].tolist()
        log.debug(f"Found top-{top_k} similar items")
        return top_indices


# 全局 Embedding 服务实例
embedding_service = EmbeddingService()