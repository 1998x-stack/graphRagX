# GraphRAG 项目

基于 LangGraph 的知识图谱检索增强生成（Graph RAG）系统。

## 核心特性

- ✅ **完整的 GraphRAG 流程**: 文档 → 分块 → 提取 → 图谱 → 社区 → 摘要 → 查询
- ✅ **FastAPI + Uvicorn**: 高性能异步 API
- ✅ **LangGraph 工作流**: 清晰的状态图管理
- ✅ **Loguru 日志**: 完整的异常追踪
- ✅ **并发控制**: asyncio + Semaphore
- ✅ **LLM 流式输出**: 可选的 streaming 支持
- ✅ **文件自动保存**: 所有 LLM 输出自动归档
- ✅ **鲁棒 JSON 提取**: 多策略正则表达式解析

## 技术栈

- **Web 框架**: FastAPI + Uvicorn
- **工作流**: LangGraph
- **图算法**: NetworkX + python-louvain
- **日志**: Loguru
- **并发**: asyncio
- **LLM**: 抽象接口（支持 OpenAI/Claude 等）

## 项目结构

```
graphrag/
├── main.py                  # 主应用入口
├── config.py               # 配置管理
├── models/                 # 数据模型
├── services/               # 服务层（LLM、Embedding、Storage）
├── core/                   # 核心算法
├── workflows/              # LangGraph 工作流
├── api/                    # API 路由
├── utils/                  # 工具函数
├── prompts/                # 提示词模板
└── outputs/                # 输出目录
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.template .env
# 编辑 .env 文件，填入你的 API Key
```

### 3. 运行服务

```bash
python main.py
```

服务将在 `http://localhost:8000` 启动。

### 4. 访问文档

- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/api/v1/health

## API 使用示例

### 1. 创建索引

```bash
curl -X POST "http://localhost:8000/api/v1/index" \
  -H "Content-Type: application/json" \
  -d '{
    "index_id": "my_knowledge_base",
    "documents": [
      "Alice works for TechCorp as a software engineer...",
      "TechCorp is a technology company specializing in AI..."
    ]
  }'
```

响应:
```json
{
  "status": "success",
  "index_id": "my_knowledge_base",
  "num_documents": 2,
  "num_chunks": 5,
  "num_entities": 15,
  "num_relations": 8,
  "num_communities": 3,
  "processing_time": 45.6
}
```

### 2. 局部搜索（Local Search）

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does Alice do?",
    "index_id": "my_knowledge_base",
    "mode": "local",
    "top_k": 5
  }'
```

### 3. 全局搜索（Global Search）

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main themes in this knowledge base?",
    "index_id": "my_knowledge_base",
    "mode": "global"
  }'
```

### 4. 列出所有索引

```bash
curl "http://localhost:8000/api/v1/indexes"
```

### 5. 获取索引统计

```bash
curl "http://localhost:8000/api/v1/query/stats/my_knowledge_base"
```

## 核心算法流程

### 索引构建流程

```
1. 文本分块 (Chunking)
   ├─ 按 chunk_size 切分
   ├─ 支持句子边界对齐
   └─ 块之间有 overlap

2. 实体关系提取 (Extraction)
   ├─ LLM 提取实体（名称、类型、描述）
   ├─ LLM 提取关系（源、目标、类型、描述）
   └─ 鲁棒的 JSON 解析

3. 图谱构建 (Graph Building)
   ├─ 实体消歧（同名同类型合并）
   ├─ 关系合并（权重累加）
   └─ 构建 NetworkX 图

4. 社区检测 (Community Detection)
   ├─ Louvain 算法（或 Leiden）
   ├─ 基于模块度优化
   └─ 生成层次化社区

5. 社区摘要 (Summarization)
   ├─ LLM 生成自然语言摘要
   ├─ 包含主要实体和关系
   └─ 150-300 词的简洁描述

6. 保存存储 (Storage)
   └─ JSON 格式持久化
```

### 查询流程

**Local Search（局部搜索）:**
```
Query → Embedding → 相似实体检索 → 邻居扩展 → 关系提取 → LLM 生成答案
```

**Global Search（全局搜索）:**
```
Query → 加载社区摘要 → LLM 综合回答
```

## 配置说明

主要配置项（在 `config.py` 或 `.env`）:

- `MAX_CONCURRENCY`: 并发数（默认 2）
- `CHUNK_SIZE`: 分块大小（默认 1000）
- `ENABLE_STREAM`: 启用流式输出
- `LLM_MODEL`: LLM 模型名称
- `LEIDEN_RESOLUTION`: 社区检测分辨率

## TODO 列表

### 必须实现

1. **LLM 服务** (`services/llm_service.py`)
   - [ ] 实现 OpenAI API 调用
   - [ ] 实现流式输出
   - [ ] 添加重试逻辑

2. **Embedding 服务** (`services/embedding_service.py`)
   - [ ] 实现真实的 Embedding API
   - [ ] 批量处理优化

### 可选增强

3. **社区检测**
   - [ ] 安装 igraph 和 leidenalg
   - [ ] 实现真正的 Leiden 算法
   - [ ] 支持层次化社区

4. **向量存储**
   - [ ] 集成 LanceDB 或 Milvus
   - [ ] 持久化 embeddings

5. **增量索引**
   - [ ] 支持增量更新
   - [ ] 避免重新索引整个文档

## 日志和调试

### 日志位置

- 控制台: 彩色输出，实时查看
- 文件: `./logs/graphrag.log`（自动轮转）
- LLM 响应: `./outputs/llm_logs/`

### 调试技巧

1. 设置 `LOG_LEVEL=DEBUG` 查看详细日志
2. 检查 `./outputs/llm_logs/` 查看 LLM 原始响应
3. 使用 Mock LLM 服务快速测试流程

## 性能优化

1. **并发控制**: 调整 `MAX_CONCURRENCY` 平衡速度和成本
2. **分块大小**: 较小的块提高精度但增加成本
3. **缓存**: 缓存 embedding 结果
4. **批处理**: LLM 批量调用

## 常见问题

### Q: 如何切换 LLM 服务?

A: 修改 `services/llm_service.py` 中的 `create_llm_service("openai")` 参数。

### Q: JSON 解析失败怎么办?

A: 系统有多重解析策略和自动重试。检查 `utils/json_extractor.py` 的日志。

### Q: 如何使用真正的 Leiden 算法?

A: 安装 `igraph` 和 `leidenalg`，修改 `core/community.py`。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## License

MIT License

---

**注意**: 这是一个教学和研究项目，生产使用前请充分测试。