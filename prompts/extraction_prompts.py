"""
实体和关系提取提示词模板
详细的提示词设计，确保 LLM 输出符合预期格式
"""


ENTITY_EXTRACTION_SYSTEM_PROMPT = """你是一个专业的知识图谱构建专家。你的任务是从给定文本中精确提取实体和关系，构建结构化的知识图谱。

# 核心任务
从文本中识别并提取：
1. **实体（Entities）**: 文本中的关键概念、人物、组织、地点、事件等
2. **关系（Relations）**: 实体之间的明确关联

# 输出要求
必须严格按照 JSON 格式输出，包含两个顶级键：
- "entities": 实体列表
- "relations": 关系列表

# 实体提取规则
每个实体必须包含：
- "name": 实体名称（首字母大写，使用规范形式）
- "type": 实体类型（如：PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, TECHNOLOGY 等）
- "description": 实体的详细描述（综合文本中关于该实体的所有信息）

实体命名规范：
- 人名：完整姓名，首字母大写
- 组织：官方全称或常用简称
- 概念：使用标准术语
- 避免代词（他、她、它等），使用完整实体名

# 关系提取规则
每个关系必须包含：
- "source": 源实体名称（必须与 entities 中的 name 完全一致）
- "target": 目标实体名称（必须与 entities 中的 name 完全一致）
- "relation_type": 关系类型（如：WORKS_FOR, LOCATED_IN, PART_OF, USES, CREATES 等）
- "description": 关系的详细描述（说明两个实体如何关联）

关系提取注意事项：
- 只提取文本中**明确存在**的关系，不要推断
- source 和 target 必须都在 entities 列表中
- 关系必须是有向的（source → target）
- 关系类型使用大写下划线格式（如：WORKS_FOR）

# 质量要求
- 实体名称必须一致（同一实体在整个文本中使用相同名称）
- 描述必须准确、简洁、信息丰富
- 避免冗余（不要重复提取相同的实体或关系）
- 确保关系的 source 和 target 都存在于实体列表中

# 输出格式示例
```json
{
  "entities": [
    {
      "name": "Alice Johnson",
      "type": "PERSON",
      "description": "Software engineer at Tech Corp, specializes in machine learning and has 5 years of experience"
    },
    {
      "name": "Tech Corp",
      "type": "ORGANIZATION",
      "description": "A technology company focused on AI and cloud computing solutions"
    },
    {
      "name": "Machine Learning",
      "type": "TECHNOLOGY",
      "description": "A branch of artificial intelligence that enables systems to learn from data"
    }
  ],
  "relations": [
    {
      "source": "Alice Johnson",
      "target": "Tech Corp",
      "relation_type": "WORKS_FOR",
      "description": "Alice Johnson is employed as a software engineer at Tech Corp"
    },
    {
      "source": "Alice Johnson",
      "target": "Machine Learning",
      "relation_type": "SPECIALIZES_IN",
      "description": "Alice Johnson has expertise in machine learning technologies"
    }
  ]
}
```

# 重要提醒
- 输出**必须是有效的 JSON**，不要包含任何其他文本
- 可以用 ```json ... ``` 代码块包裹 JSON
- 如果文本中没有明确的实体或关系，返回空列表
"""


def create_extraction_prompt(
    text: str,
    max_entities: int = 20,
    max_relations: int = 20
) -> str:
    """
    创建实体关系提取提示词
    
    Args:
        text: 待提取的文本
        max_entities: 最大实体数量
        max_relations: 最大关系数量
        
    Returns:
        完整的提示词
    """
    prompt = f"""{ENTITY_EXTRACTION_SYSTEM_PROMPT}

# 当前任务
请从以下文本中提取实体和关系：

## 约束条件
- 最多提取 {max_entities} 个实体
- 最多提取 {max_relations} 个关系

## 待分析文本
\"\"\"
{text}
\"\"\"

## 你的输出
请严格按照上述 JSON 格式输出提取结果：
"""
    return prompt


# 提取结果验证的必需字段
REQUIRED_ENTITY_KEYS = ["name", "type", "description"]
REQUIRED_RELATION_KEYS = ["source", "target", "relation_type", "description"]


def validate_extraction_result(result: dict) -> tuple[bool, str]:
    """
    验证提取结果的格式
    
    Args:
        result: 提取结果字典
        
    Returns:
        (是否有效, 错误信息)
    """
    # 检查顶级键
    if "entities" not in result:
        return False, "Missing 'entities' key"
    if "relations" not in result:
        return False, "Missing 'relations' key"
    
    # 检查类型
    if not isinstance(result["entities"], list):
        return False, "'entities' must be a list"
    if not isinstance(result["relations"], list):
        return False, "'relations' must be a list"
    
    # 检查实体格式
    for i, entity in enumerate(result["entities"]):
        if not isinstance(entity, dict):
            return False, f"Entity {i} is not a dict"
        for key in REQUIRED_ENTITY_KEYS:
            if key not in entity:
                return False, f"Entity {i} missing key: {key}"
    
    # 检查关系格式
    entity_names = {e["name"] for e in result["entities"]}
    for i, relation in enumerate(result["relations"]):
        if not isinstance(relation, dict):
            return False, f"Relation {i} is not a dict"
        for key in REQUIRED_RELATION_KEYS:
            if key not in relation:
                return False, f"Relation {i} missing key: {key}"
        
        # 验证 source 和 target 存在于实体列表中
        if relation["source"] not in entity_names:
            return False, f"Relation {i} source '{relation['source']}' not in entities"
        if relation["target"] not in entity_names:
            return False, f"Relation {i} target '{relation['target']}' not in entities"
    
    return True, ""