"""
配置管理模块
所有全局配置集中管理
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """全局配置"""
    
    # ==================== 服务配置 ====================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # ==================== LLM 配置 ====================
    LLM_MODEL: str = "gpt-4"  # TODO: 替换为你的模型
    LLM_API_KEY: str = ""  # TODO: 填写 API Key
    LLM_BASE_URL: str = ""  # TODO: 如果需要自定义 base_url
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 4000
    
    # ==================== Embedding 配置 ====================
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_API_KEY: str = ""  # TODO: 填写 API Key
    EMBEDDING_DIMENSION: int = 1536
    
    # ==================== 并发配置 ====================
    MAX_CONCURRENCY: int = 2  # 默认并发数
    MAX_RETRY: int = 3  # 最大重试次数
    RETRY_DELAY: float = 1.0  # 重试延迟（秒）
    
    # ==================== 分块配置 ====================
    CHUNK_SIZE: int = 1000  # 分块大小（字符）
    CHUNK_OVERLAP: int = 200  # 分块重叠
    
    # ==================== 提取配置 ====================
    MAX_ENTITIES_PER_CHUNK: int = 20  # 每个块最多提取实体数
    MAX_RELATIONS_PER_CHUNK: int = 20  # 每个块最多提取关系数
    
    # ==================== 社区检测配置 ====================
    LEIDEN_RESOLUTION: float = 1.0  # 分辨率参数
    LEIDEN_MAX_ITERATIONS: int = 100
    LEIDEN_SEED: int = 42
    
    # ==================== 流式输出配置 ====================
    ENABLE_STREAM: bool = False  # 是否启用流式输出
    STREAM_LOG_ENABLED: bool = True  # 流式时是否记录日志
    
    # ==================== 存储配置 ====================
    OUTPUT_DIR: Path = Path("./outputs")
    LLM_LOGS_DIR: Path = OUTPUT_DIR / "llm_logs"
    GRAPHS_DIR: Path = OUTPUT_DIR / "graphs"
    COMMUNITIES_DIR: Path = OUTPUT_DIR / "communities"
    
    # ==================== 日志配置 ====================
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = Path("./logs/graphrag.log")
    LOG_ROTATION: str = "500 MB"
    LOG_RETENTION: str = "10 days"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 自动创建必要目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        for dir_path in [
            self.OUTPUT_DIR,
            self.LLM_LOGS_DIR,
            self.GRAPHS_DIR,
            self.COMMUNITIES_DIR,
            self.LOG_FILE.parent
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# 全局配置实例
settings = Settings()