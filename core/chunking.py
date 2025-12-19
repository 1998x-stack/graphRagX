"""
文本分块算法
将长文档切分为适合 LLM 处理的块
"""
import uuid
from typing import List
from models.schemas import TextChunk
from config import settings
from utils.logger import log


class TextChunker:
    """文本分块器"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        初始化分块器
        
        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 块重叠大小
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        log.info(f"TextChunker initialized: size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict = None
    ) -> List[TextChunk]:
        """
        将文本分块
        
        Args:
            text: 输入文本
            doc_id: 文档ID
            metadata: 元数据
            
        Returns:
            文本块列表
        """
        if not text or not text.strip():
            log.warning(f"Empty text for doc_id={doc_id}")
            return []
        
        chunks = []
        text_length = len(text)
        start = 0
        chunk_index = 0
        
        while start < text_length:
            # 计算当前块的结束位置
            end = min(start + self.chunk_size, text_length)
            
            # 提取当前块
            chunk_text = text[start:end]
            
            # 如果不是最后一块，尝试在句子边界处切分（避免切断句子）
            if end < text_length:
                # 查找最后一个句号、问号或感叹号
                last_sentence_end = max(
                    chunk_text.rfind('.'),
                    chunk_text.rfind('?'),
                    chunk_text.rfind('!'),
                    chunk_text.rfind('。'),  # 中文句号
                    chunk_text.rfind('？'),
                    chunk_text.rfind('！')
                )
                
                if last_sentence_end > self.chunk_size * 0.5:  # 至少保留一半内容
                    end = start + last_sentence_end + 1
                    chunk_text = text[start:end]
            
            # 创建 TextChunk 对象
            chunk = TextChunk(
                id=f"{doc_id}_chunk_{chunk_index}",
                text=chunk_text.strip(),
                doc_id=doc_id,
                chunk_index=chunk_index,
                metadata=metadata or {}
            )
            
            chunks.append(chunk)
            
            # 计算下一块的起始位置（考虑重叠）
            start = end - self.chunk_overlap
            chunk_index += 1
            
            # 避免无限循环
            if start >= text_length:
                break
        
        log.info(f"Chunked doc_id={doc_id} into {len(chunks)} chunks")
        return chunks
    
    def chunk_documents(
        self,
        documents: List[str],
        doc_ids: List[str] = None
    ) -> List[TextChunk]:
        """
        批量分块多个文档
        
        Args:
            documents: 文档列表
            doc_ids: 文档ID列表（可选，自动生成）
            
        Returns:
            所有文本块列表
        """
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        if len(documents) != len(doc_ids):
            raise ValueError("documents and doc_ids must have same length")
        
        all_chunks = []
        for doc, doc_id in zip(documents, doc_ids):
            chunks = self.chunk_text(doc, doc_id)
            all_chunks.extend(chunks)
        
        log.info(f"Chunked {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks


# 全局分块器实例
text_chunker = TextChunker()