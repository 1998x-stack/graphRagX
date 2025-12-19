# GraphRAG 项目结构设计

## 核心流程
```
文档 → 文本分块 → 实体/关系提取 → 图谱构建 → 社区检测 → 社区摘要 → 查询响应
```

## 项目目录结构
```
graphrag/
├── main.py                      # FastAPI 主入口 + Uvicorn 启动
├── config.py                    # 配置管理（并发数、模型配置等）
│
├── models/                      # 数据模型层
│   ├── __init__.py
│   ├── schemas.py              # Pydantic 数据模型
│   └── graph.py                # 图数据结构（Entity、Relation、Community）
│
├── services/                    # 服务层（外部依赖抽象）
│   ├── __init__.py
│   ├── llm_service.py          # LLM 服务（stream 支持）
│   ├── embedding_service.py    # Embedding 服务
│   └── storage_service.py      # 文件存储服务（保存所有 LLM 输出）
│
├── core/                        # 核心算法层
│   ├── __init__.py
│   ├── chunking.py             # 文本分块算法
│   ├── extraction.py           # 实体/关系提取（LLM + JSON 解析）
│   ├── graph_builder.py        # 图谱构建（合并实体/关系）
│   ├── community.py            # Leiden 社区检测算法
│   └── summarization.py        # 社区摘要生成
│
├── workflows/                   # LangGraph 工作流
│   ├── __init__.py
│   ├── indexing_workflow.py    # 索引构建工作流（完整流程）
│   └── query_workflow.py       # 查询工作流（Local/Global Search）
│
├── api/                         # FastAPI 路由层
│   ├── __init__.py
│   ├── index.py                # 索引 API（POST /index）
│   └── query.py                # 查询 API（POST /query）
│
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── logger.py               # Loguru 日志配置
│   ├── concurrency.py          # asyncio 并发控制（Semaphore）
│   └── json_extractor.py       # 鲁棒的 JSON 提取器（正则 + 重试）
│
├── prompts/                     # 提示词模板
│   ├── __init__.py
│   ├── extraction_prompts.py   # 实体/关系提取提示词
│   └── summary_prompts.py      # 社区摘要提示词
│
└── outputs/                     # 输出目录（自动创建）
    ├── llm_logs/               # LLM 响应日志
    │   └── {timestamp}_{task}.json
    ├── graphs/                 # 图谱数据
    │   └── {index_id}_graph.json
    └── communities/            # 社区数据
        └── {index_id}_communities.json
```

## 技术栈
- **Web 框架**: FastAPI + Uvicorn
- **日志**: Loguru（所有异常打 traceback）
- **并发**: asyncio + Semaphore（默认并发数 2）
- **工作流**: LangGraph（状态图）
- **图算法**: NetworkX + igraph（Leiden）
- **LLM**: 抽象接口（支持 streaming）

## 核心设计原则
1. **简洁优先**: 只实现核心功能，避免过度设计
2. **异步优先**: 所有 I/O 操作使用 async/await
3. **日志完备**: 每个步骤都有详细日志，异常必打 traceback
4. **文件保存**: 所有 LLM 输出自动保存到 outputs/ 目录
5. **鲁棒解析**: JSON 提取使用正则 + 多种策略

## 数据流
```
1. 索引阶段:
   Document → Chunks → [Entity/Relation Extraction] 
   → Graph Building → Community Detection → [Community Summary]
   → Save to Storage

2. 查询阶段:
   Query → Load Graph → Search Algorithm (Local/Global)
   → [Generate Answer] → Response
```

## API 设计
```
POST /api/v1/index
- 输入: { "documents": [...], "index_id": "..." }
- 输出: { "status": "success", "entities": N, "communities": M }

POST /api/v1/query
- 输入: { "query": "...", "index_id": "...", "mode": "local|global" }
- 输出: { "answer": "...", "sources": [...] }

GET /api/v1/health
- 健康检查
```

## 配置项（config.py）
```python
- LLM_MODEL: 模型名称
- EMBEDDING_MODEL: 嵌入模型
- MAX_CONCURRENCY: 最大并发数（默认 2）
- CHUNK_SIZE: 分块大小（默认 1000）
- CHUNK_OVERLAP: 分块重叠（默认 200）
- ENABLE_STREAM: 是否启用流式输出（默认 False）
- OUTPUT_DIR: 输出目录（默认 ./outputs）
```