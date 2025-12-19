"""
并发控制模块
使用 asyncio.Semaphore 控制并发数
"""
import asyncio
from typing import List, Callable, Any, TypeVar
from config import settings
from utils.logger import log, log_exception

T = TypeVar('T')


class ConcurrencyController:
    """并发控制器"""
    
    def __init__(self, max_concurrency: int = None):
        """
        初始化并发控制器
        
        Args:
            max_concurrency: 最大并发数，默认使用配置值
        """
        self.max_concurrency = max_concurrency or settings.MAX_CONCURRENCY
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        log.info(f"ConcurrencyController initialized with max_concurrency={self.max_concurrency}")
    
    async def run_with_semaphore(self, coro: Callable, *args, **kwargs) -> Any:
        """
        在 Semaphore 控制下运行协程
        
        Args:
            coro: 协程函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            协程执行结果
        """
        async with self.semaphore:
            try:
                result = await coro(*args, **kwargs)
                return result
            except Exception as e:
                log_exception(e, f"run_with_semaphore({coro.__name__})")
                raise
    
    async def gather_with_concurrency(
        self,
        tasks: List[Callable],
        *args_list,
        return_exceptions: bool = False
    ) -> List[Any]:
        """
        并发执行多个任务（带并发控制）
        
        Args:
            tasks: 任务列表（协程函数）
            args_list: 每个任务的参数列表
            return_exceptions: 是否返回异常而非抛出
            
        Returns:
            结果列表
        """
        log.info(f"Starting {len(tasks)} concurrent tasks (max_concurrency={self.max_concurrency})")
        
        # 创建受控的协程列表
        controlled_tasks = [
            self.run_with_semaphore(task, *args)
            for task, args in zip(tasks, args_list)
        ]
        
        # 并发执行
        results = await asyncio.gather(*controlled_tasks, return_exceptions=return_exceptions)
        
        log.info(f"Completed {len(tasks)} concurrent tasks")
        return results
    
    async def map_async(
        self,
        func: Callable,
        items: List[Any],
        return_exceptions: bool = False
    ) -> List[Any]:
        """
        对列表中的每个元素异步应用函数（类似 map）
        
        Args:
            func: 异步函数
            items: 输入列表
            return_exceptions: 是否返回异常
            
        Returns:
            结果列表
        """
        tasks = [func for _ in items]
        args_list = [(item,) for item in items]
        return await self.gather_with_concurrency(
            tasks,
            *args_list,
            return_exceptions=return_exceptions
        )


# 全局并发控制器实例
concurrency_controller = ConcurrencyController()