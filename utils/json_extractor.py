"""
鲁棒的 JSON 提取器
使用多种策略从 LLM 输出中提取 JSON
"""
import re
import json
from typing import Any, Optional
from utils.logger import log, log_exception


class JSONExtractor:
    """JSON 提取器"""
    
    # 正则表达式模式（按优先级排序）
    PATTERNS = [
        # 1. 标准 JSON 代码块: ```json ... ```
        r'```json\s*(\{.*?\})\s*```',
        # 2. 普通代码块: ``` ... ```
        r'```\s*(\{.*?\})\s*```',
        # 3. 直接 JSON 对象（贪婪匹配最外层大括号）
        r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})',
        # 4. 宽松模式：查找任何 { ... } 结构
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
    ]
    
    @classmethod
    def extract(cls, text: str) -> Optional[dict]:
        """
        从文本中提取 JSON 对象
        
        Args:
            text: 包含 JSON 的文本
            
        Returns:
            提取的 JSON 对象，失败返回 None
        """
        if not text:
            log.warning("Empty text provided for JSON extraction")
            return None
        
        # 策略 1: 直接解析（文本就是纯 JSON）
        try:
            result = json.loads(text.strip())
            log.debug("JSON extracted using direct parsing")
            return result
        except json.JSONDecodeError:
            pass
        
        # 策略 2: 使用正则表达式提取
        for i, pattern in enumerate(cls.PATTERNS, 1):
            try:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    json_str = match.group(1) if match.lastindex else match.group(0)
                    result = json.loads(json_str)
                    log.debug(f"JSON extracted using pattern #{i}")
                    return result
            except (json.JSONDecodeError, AttributeError, IndexError) as e:
                log.debug(f"Pattern #{i} failed: {str(e)}")
                continue
        
        # 策略 3: 清理后重试（移除常见噪音）
        try:
            # 移除可能的前缀/后缀文本
            cleaned = text.strip()
            # 查找第一个 { 和最后一个 }
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = cleaned[start:end+1]
                result = json.loads(json_str)
                log.debug("JSON extracted using cleanup strategy")
                return result
        except json.JSONDecodeError:
            pass
        
        log.error(f"Failed to extract JSON from text. First 200 chars: {text[:200]}...")
        return None
    
    @classmethod
    def extract_list(cls, text: str) -> Optional[list]:
        """
        从文本中提取 JSON 数组
        
        Args:
            text: 包含 JSON 数组的文本
            
        Returns:
            提取的 JSON 数组，失败返回 None
        """
        # 修改模式以匹配数组
        array_patterns = [
            r'```json\s*(\[.*?\])\s*```',
            r'```\s*(\[.*?\])\s*```',
            r'(\[(?:[^\[\]]|(?:\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]))*\])',
        ]
        
        for i, pattern in enumerate(array_patterns, 1):
            try:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    json_str = match.group(1) if match.lastindex else match.group(0)
                    result = json.loads(json_str)
                    if isinstance(result, list):
                        log.debug(f"JSON array extracted using pattern #{i}")
                        return result
            except (json.JSONDecodeError, AttributeError) as e:
                log.debug(f"Array pattern #{i} failed: {str(e)}")
                continue
        
        log.error("Failed to extract JSON array from text")
        return None
    
    @classmethod
    def validate_schema(cls, data: dict, required_keys: list) -> bool:
        """
        验证 JSON 数据是否包含必需的键
        
        Args:
            data: JSON 数据
            required_keys: 必需的键列表
            
        Returns:
            是否有效
        """
        if not isinstance(data, dict):
            return False
        
        for key in required_keys:
            if key not in data:
                log.warning(f"Missing required key: {key}")
                return False
        
        return True
    
    @classmethod
    def safe_extract(cls, text: str, default: Any = None) -> Any:
        """
        安全提取 JSON（失败返回默认值）
        
        Args:
            text: 文本
            default: 默认值
            
        Returns:
            提取的 JSON 或默认值
        """
        try:
            result = cls.extract(text)
            return result if result is not None else default
        except Exception as e:
            log_exception(e, "safe_extract")
            return default


# 便捷函数
def extract_json(text: str) -> Optional[dict]:
    """提取 JSON 对象"""
    return JSONExtractor.extract(text)


def extract_json_list(text: str) -> Optional[list]:
    """提取 JSON 数组"""
    return JSONExtractor.extract_list(text)