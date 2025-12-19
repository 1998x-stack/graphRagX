"""
LLM 服务抽象层
支持流式输出和自动保存
"""
from typing import Optional, AsyncIterator
from abc import ABC, abstractmethod
from config import settings
from utils.logger import log, log_llm_call, log_stream_chunk, log_exception
from services.storage_service import storage_service


class LLMService(ABC):
    """LLM 服务抽象基类"""
    
    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        初始化 LLM 服务
        
        Args:
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
        """
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        log.info(f"LLMService initialized: model={self.model}, temp={self.temperature}")
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        task: str = "general",
        stream: bool = None,
        save_response: bool = True,
        **kwargs
    ) -> str:
        """
        生成文本（抽象方法）
        
        Args:
            prompt: 输入提示词
            task: 任务类型（用于日志和文件命名）
            stream: 是否使用流式输出（None 则使用配置）
            save_response: 是否保存响应到文件
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        task: str = "general",
        **kwargs
    ) -> AsyncIterator[str]:
        """
        流式生成文本（抽象方法）
        
        Args:
            prompt: 输入提示词
            task: 任务类型
            **kwargs: 其他参数
            
        Yields:
            文本块
        """
        pass


class OpenAILLMService(LLMService):
    """OpenAI LLM 服务实现（TODO: 需要实现）"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # TODO: 初始化 OpenAI 客户端
        # from openai import AsyncOpenAI
        # self.client = AsyncOpenAI(
        #     api_key=settings.LLM_API_KEY,
        #     base_url=settings.LLM_BASE_URL or None
        # )
        
        log.warning("OpenAILLMService initialized but client NOT implemented - TODO!")
    
    async def generate(
        self,
        prompt: str,
        task: str = "general",
        stream: bool = None,
        save_response: bool = True,
        **kwargs
    ) -> str:
        """生成文本"""
        use_stream = stream if stream is not None else settings.ENABLE_STREAM
        
        try:
            if use_stream:
                # 流式生成
                full_response = ""
                async for chunk in self.generate_stream(prompt, task, **kwargs):
                    full_response += chunk
                    if settings.STREAM_LOG_ENABLED:
                        log_stream_chunk(chunk, task)
                response = full_response
            else:
                # 非流式生成
                # TODO: 实现 OpenAI API 调用
                # response_obj = await self.client.chat.completions.create(
                #     model=self.model,
                #     messages=[{"role": "user", "content": prompt}],
                #     temperature=self.temperature,
                #     max_tokens=self.max_tokens,
                #     **kwargs
                # )
                # response = response_obj.choices[0].message.content
                
                # 临时占位符
                response = "[TODO: Implement OpenAI API call]"
                log.warning("OpenAI API call not implemented - returning placeholder")
            
            # 记录日志
            log_llm_call(prompt, response, self.model, task)
            
            # 保存响应
            if save_response:
                storage_service.save_llm_response(
                    task=task,
                    prompt=prompt,
                    response=response,
                    metadata={"model": self.model, "temperature": self.temperature}
                )
            
            return response
            
        except Exception as e:
            log_exception(e, f"LLMService.generate(task={task})")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        task: str = "general",
        **kwargs
    ) -> AsyncIterator[str]:
        """流式生成文本"""
        try:
            # TODO: 实现流式 API 调用
            # stream = await self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=self.temperature,
            #     max_tokens=self.max_tokens,
            #     stream=True,
            #     **kwargs
            # )
            # 
            # async for chunk in stream:
            #     if chunk.choices[0].delta.content:
            #         yield chunk.choices[0].delta.content
            
            # 临时占位符
            yield "[TODO: Implement streaming]"
            log.warning("Streaming not implemented")
            
        except Exception as e:
            log_exception(e, f"LLMService.generate_stream(task={task})")
            raise


class MockLLMService(LLMService):
    """Mock LLM 服务（用于测试）"""
    
    async def generate(
        self,
        prompt: str,
        task: str = "general",
        stream: bool = None,
        save_response: bool = True,
        **kwargs
    ) -> str:
        """生成 Mock 响应"""
        log.info(f"MockLLMService.generate(task={task})")
        
        # 根据任务类型返回不同的 Mock 数据
        if task == "entity_extraction":
            response = '''```json
{
  "entities": [
    {"name": "Alice", "type": "PERSON", "description": "A test person"},
    {"name": "Company X", "type": "ORGANIZATION", "description": "A test company"}
  ],
  "relations": [
    {"source": "Alice", "target": "Company X", "relation_type": "WORKS_FOR", "description": "Alice works for Company X"}
  ]
}
```'''
        elif task == "community_summary":
            response = "This is a mock community summary. It describes the main entities and relationships in the community."
        else:
            response = f"Mock response for task: {task}"
        
        # 记录和保存
        log_llm_call(prompt, response, self.model, task)
        if save_response:
            storage_service.save_llm_response(task, prompt, response)
        
        return response
    
    async def generate_stream(
        self,
        prompt: str,
        task: str = "general",
        **kwargs
    ) -> AsyncIterator[str]:
        """流式生成 Mock 响应"""
        response = f"Mock streaming response for task: {task}"
        for char in response:
            yield char


# 工厂函数：创建 LLM 服务实例
def create_llm_service(service_type: str = "openai") -> LLMService:
    """
    创建 LLM 服务实例
    
    Args:
        service_type: 服务类型 ('openai', 'mock')
        
    Returns:
        LLM 服务实例
    """
    if service_type == "openai":
        return OpenAILLMService()
    elif service_type == "mock":
        return MockLLMService()
    else:
        raise ValueError(f"Unknown service type: {service_type}")


# 全局 LLM 服务实例（默认使用 Mock，生产环境改为 'openai'）
llm_service = create_llm_service("mock")  # TODO: 改为 "openai"