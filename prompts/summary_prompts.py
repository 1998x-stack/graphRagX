"""
社区摘要生成提示词模板
用于生成社区的自然语言描述
"""


COMMUNITY_SUMMARY_SYSTEM_PROMPT = """你是一个知识图谱分析专家。你的任务是对社区进行深入分析并生成结构化的摘要报告。

# 任务目标
为给定的实体社区生成一份全面的摘要报告，帮助用户快速理解社区的核心内容和结构。

# 社区信息
社区由一组紧密相关的实体和它们之间的关系组成。你将得到：
1. 社区中的所有实体及其描述
2. 社区内部的关系网络
3. 社区的规模和层级信息

# 摘要内容要求
你的摘要应该包含以下部分：

## 1. 社区概述（2-3句话）
- 社区的主题或焦点是什么？
- 社区的核心内容简述

## 2. 关键实体（列出3-5个最重要的实体）
- 实体名称和类型
- 实体在社区中的角色

## 3. 主要关系
- 描述社区中最重要的关系模式
- 实体之间如何相互关联

## 4. 见解和模式
- 社区揭示了什么样的知识模式？
- 有什么值得注意的发现？

# 输出风格
- 使用清晰、专业的语言
- 避免冗长，突出重点
- 使用自然流畅的段落（不要使用 markdown 标题）
- 信息密度高，避免废话

# 输出长度
- 目标长度：150-300 词
- 根据社区规模适当调整

# 输出格式
直接输出文本摘要，不要使用 JSON 或其他结构化格式。
"""


def create_community_summary_prompt(
    community_id: str,
    entities: list,
    relations: list,
    level: int = 0
) -> str:
    """
    创建社区摘要生成提示词
    
    Args:
        community_id: 社区ID
        entities: 社区中的实体列表
        relations: 社区内的关系列表
        level: 社区层级
        
    Returns:
        完整的提示词
    """
    # 格式化实体信息
    entity_info = []
    for entity in entities[:20]:  # 最多显示20个实体
        name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "Unknown")
        desc = entity.get("description", "No description")
        entity_info.append(f"- {name} ({entity_type}): {desc}")
    
    entity_text = "\n".join(entity_info)
    if len(entities) > 20:
        entity_text += f"\n... and {len(entities) - 20} more entities"
    
    # 格式化关系信息
    relation_info = []
    for relation in relations[:20]:  # 最多显示20个关系
        source = relation.get("source", "Unknown")
        target = relation.get("target", "Unknown")
        rel_type = relation.get("relation_type", "RELATED_TO")
        desc = relation.get("description", "")
        relation_info.append(f"- {source} --[{rel_type}]--> {target}: {desc}")
    
    relation_text = "\n".join(relation_info)
    if len(relations) > 20:
        relation_text += f"\n... and {len(relations) - 20} more relations"
    
    prompt = f"""{COMMUNITY_SUMMARY_SYSTEM_PROMPT}

# 当前社区信息

## 基本信息
- 社区 ID: {community_id}
- 层级: Level {level}
- 实体数量: {len(entities)}
- 关系数量: {len(relations)}

## 社区实体
{entity_text}

## 社区关系
{relation_text}

# 你的任务
请基于上述信息，生成一份全面而简洁的社区摘要报告。直接输出摘要文本即可。
"""
    return prompt


# 全局搜索的社区摘要提示词
GLOBAL_SEARCH_PROMPT_TEMPLATE = """# 任务
请基于社区摘要信息回答用户的问题。

# 社区摘要信息
{community_summaries}

# 用户问题
{query}

# 要求
- 综合多个社区的信息
- 提供全面、准确的答案
- 如果信息不足，明确说明
- 引用相关社区（可选）

# 你的回答
"""


def create_global_search_prompt(query: str, community_summaries: list) -> str:
    """
    创建全局搜索提示词
    
    Args:
        query: 用户查询
        community_summaries: 社区摘要列表
        
    Returns:
        完整的提示词
    """
    # 格式化社区摘要
    summaries_text = []
    for i, summary in enumerate(community_summaries, 1):
        comm_id = summary.get("id", f"community_{i}")
        text = summary.get("summary", "No summary available")
        summaries_text.append(f"## Community {comm_id}\n{text}\n")
    
    summaries_str = "\n".join(summaries_text)
    
    return GLOBAL_SEARCH_PROMPT_TEMPLATE.format(
        community_summaries=summaries_str,
        query=query
    )


# 局部搜索的提示词模板
LOCAL_SEARCH_PROMPT_TEMPLATE = """# 任务
请基于给定的实体和关系信息回答用户的问题。

# 相关实体
{entities}

# 相关关系
{relations}

# 用户问题
{query}

# 要求
- 优先使用给定的实体和关系信息
- 答案要准确、具体
- 如果信息不足以回答，明确说明
- 可以适当推理，但要基于事实

# 你的回答
"""


def create_local_search_prompt(query: str, entities: list, relations: list) -> str:
    """
    创建局部搜索提示词
    
    Args:
        query: 用户查询
        entities: 相关实体列表
        relations: 相关关系列表
        
    Returns:
        完整的提示词
    """
    # 格式化实体
    entities_text = []
    for entity in entities:
        name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "Unknown")
        desc = entity.get("description", "")
        entities_text.append(f"- {name} ({entity_type}): {desc}")
    
    entities_str = "\n".join(entities_text) if entities_text else "No entities found"
    
    # 格式化关系
    relations_text = []
    for relation in relations:
        source = relation.get("source", "Unknown")
        target = relation.get("target", "Unknown")
        rel_type = relation.get("relation_type", "RELATED_TO")
        desc = relation.get("description", "")
        relations_text.append(f"- {source} --[{rel_type}]--> {target}: {desc}")
    
    relations_str = "\n".join(relations_text) if relations_text else "No relations found"
    
    return LOCAL_SEARCH_PROMPT_TEMPLATE.format(
        entities=entities_str,
        relations=relations_str,
        query=query
    )