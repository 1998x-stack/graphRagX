"""
Loguru 日志配置模块
所有异常自动打印 traceback
"""
import sys
from loguru import logger
from config import settings


def setup_logger():
    """配置 Loguru 日志系统"""
    
    # 移除默认 handler
    logger.remove()
    
    # 控制台输出（带颜色）
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
        backtrace=True,  # 启用回溯
        diagnose=True,   # 启用详细诊断
    )
    
    # 文件输出（JSON 格式，便于解析）
    logger.add(
        settings.LOG_FILE,
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True,  # 异步写入
    )
    
    logger.info("Logger initialized successfully")
    return logger


# 全局 logger 实例
log = setup_logger()


def log_exception(exc: Exception, context: str = ""):
    """
    统一的异常日志记录
    
    Args:
        exc: 异常对象
        context: 上下文信息
    """
    log.exception(f"Exception in {context}: {str(exc)}")


def log_llm_call(prompt: str, response: str, model: str, task: str):
    """
    记录 LLM 调用
    
    Args:
        prompt: 输入提示词
        response: LLM 响应
        model: 模型名称
        task: 任务类型
    """
    log.info(
        f"LLM Call - Task: {task} | Model: {model} | "
        f"Prompt Length: {len(prompt)} | Response Length: {len(response)}"
    )


def log_stream_chunk(chunk: str, task: str):
    """
    记录流式输出块
    
    Args:
        chunk: 流式输出块
        task: 任务类型
    """
    if settings.STREAM_LOG_ENABLED:
        log.info(f"Stream [{task}]: {chunk[:100]}...")  # 只记录前100字符