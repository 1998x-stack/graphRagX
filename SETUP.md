# GraphRAG 项目安装和设置指南

## 📋 前置要求

- Python 3.9+
- pip 或 conda

## 🚀 完整安装步骤

### 步骤 1: 创建项目目录结构

```bash
mkdir -p graphrag
cd graphrag

# 创建所有必要的目录
mkdir -p models services core workflows api utils prompts
mkdir -p outputs/llm_logs outputs/graphs outputs/communities
mkdir -p logs

# 创建所有 __init__.py 文件
touch models/__init__.py
touch services/__init__.py
touch core/__init__.py
touch workflows/__init__.py
touch api/__init__.py
touch utils/__init__.py
touch prompts/__init__.py
```

### 步骤 2: 复制所有代码文件

将所有 artifact 中的代码文件放置到对应的位置：

```
graphrag/
├── main.py
├── config.py
├── requirements.txt
├── .env.template
├── README.md
├── SETUP.md
├── example_usage.py
│
├── models/
│   ├── __init__.py
│   ├── schemas.py
│   └── graph.py
│
├── services/
│   ├── __init__.py
│   ├── llm_service.py
│   ├── embedding_service.py
│   └── storage_service.py
│
├── core/
│   ├── __init__.py
│   ├── chunking.py
│   ├── extraction.py
│   ├── graph_builder.py
│   ├── community.py
│   └── summarization.py
│
├── workflows/
│   ├── __init__.py
│   ├── indexing_workflow.py
│   └── query_workflow.py
│
├── api/
│   ├── __init__.py
│   ├── index.py
│   └── query.py
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── concurrency.py
│   └── json_extractor.py
│
└── prompts/
    ├── __init__.py
    ├── extraction_prompts.py
    └── summary_prompts.py
```

### 步骤 3: 创建 __init__.py 文件

在每个包目录下创建 `__init__.py`（可以为空或包含导入）:

**models/__init__.py**:
```python
from .schemas import *
from .graph import *
```

**services/__init__.py**:
```python
from .llm_service import llm_service
from .embedding_service import embedding_service
from .storage_service import storage_service
```

**core/__init__.py**:
```python
from .chunking import text_chunker
from .extraction import entity_relation_extractor
from .graph_builder import graph_builder
from .community import community_detector
from .summarization import community_summarizer
```

**workflows/__init__.py**:
```python
from .indexing_workflow import indexing_workflow
from .query_workflow import query_workflow
```

**api/__init__.py**:
```python
# Empty or import routers if needed
```

**utils/__init__.py**:
```python
from .logger import log
from .concurrency import concurrency_controller
from .json_extractor import extract_json, extract_json_list
```

**prompts/__init__.py**:
```python
from .extraction_prompts import *
from .summary_prompts import *
```

### 步骤 4: 安装依赖

```bash
pip install -r requirements.txt
```

如果某些包安装失败，可以逐个安装：

```bash
# 核心框架
pip install fastapi uvicorn pydantic pydantic-settings

# LangGraph
pip install langgraph langchain langchain-core

# 图和网络
pip install networkx python-louvain

# 日志
pip install loguru

# 数据处理
pip install numpy

# 工具
pip install python-dotenv aiofiles

# 可选：真正的 Leiden 算法
# pip install igraph leidenalg
```

### 步骤 5: 配置环境变量

```bash
# 复制模板
cp .env.template .env

# 编辑 .env 文件
# 填入你的 API Key 和配置
vim .env  # 或使用其他编辑器
```

**必须配置的项**:
```
LLM_API_KEY=your-openai-api-key-here
EMBEDDING_API_KEY=your-embedding-api-key-here
```

### 步骤 6: 实现 LLM 服务（重要！）

编辑 `services/llm_service.py`，实现真实的 LLM API 调用：

```python
# 取消注释并填充 OpenAILLMService 中的 TODO 部分
# 示例：
from openai import AsyncOpenAI

class OpenAILLMService(LLMService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL or None
        )
    
    async def generate(self, prompt, task="general", stream=None, save_response=True, **kwargs):
        # 实现 API 调用
        response_obj = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        response = response_obj.choices[0].message.content
        # ... 其余代码
```

同样实现 `services/embedding_service.py` 中的 embedding API 调用。

### 步骤 7: 测试运行

```bash
# 启动服务
python main.py
```

服务应该在 `http://localhost:8000` 启动。

访问 API 文档：http://localhost:8000/docs

### 步骤 8: 运行示例

```bash
# 在另一个终端运行示例脚本
python example_usage.py
```

## 🔧 常见问题排查

### 问题1: 导入错误

```
ModuleNotFoundError: No module named 'xxx'
```

**解决**:
- 确保所有 `__init__.py` 文件已创建
- 从项目根目录运行 `python main.py`
- 检查依赖是否安装完整

### 问题2: LLM API 失败

```
TODO: Implement OpenAI API call
```

**解决**:
- 实现 `services/llm_service.py` 中的 TODO 部分
- 或暂时使用 Mock 服务测试流程：
  ```python
  # 在 services/llm_service.py 底部
  llm_service = create_llm_service("mock")  # 使用 Mock
  ```

### 问题3: JSON 解析失败

**解决**:
- 检查 `utils/json_extractor.py` 的日志
- 查看 `outputs/llm_logs/` 中的原始 LLM 响应
- 调整提示词使 LLM 输出更规范

### 问题4: 社区检测失败

```
ModuleNotFoundError: No module named 'community'
```

**解决**:
```bash
pip install python-louvain
```

### 问题5: 端口占用

```
ERROR: [Errno 48] Address already in use
```

**解决**:
- 修改 `.env` 中的 `API_PORT`
- 或杀死占用端口的进程

## 📊 验证安装

运行以下命令验证各组件：

```bash
# 1. 测试导入
python -c "from models.schemas import Entity; print('✅ Models OK')"
python -c "from services.llm_service import llm_service; print('✅ Services OK')"
python -c "from core.chunking import text_chunker; print('✅ Core OK')"
python -c "from workflows.indexing_workflow import indexing_workflow; print('✅ Workflows OK')"

# 2. 测试 API
curl http://localhost:8000/api/v1/health

# 3. 运行示例
python example_usage.py
```

## 🎯 下一步

1. **实现 LLM 服务**: 完成 `services/llm_service.py` 的 TODO
2. **实现 Embedding 服务**: 完成 `services/embedding_service.py` 的 TODO
3. **测试完整流程**: 运行 `example_usage.py`
4. **调优提示词**: 根据实际效果调整 `prompts/` 中的提示词
5. **优化性能**: 调整并发数、分块大小等参数

## 📚 参考资源

- FastAPI 文档: https://fastapi.tiangolo.com/
- LangGraph 文档: https://langchain-ai.github.io/langgraph/
- NetworkX 文档: https://networkx.org/
- Loguru 文档: https://loguru.readthedocs.io/

## 🐛 遇到问题？

1. 检查日志: `./logs/graphrag.log`
2. 查看 LLM 响应: `./outputs/llm_logs/`
3. 启用 DEBUG 日志: 设置 `.env` 中 `LOG_LEVEL=DEBUG`
4. 提交 Issue（如果是代码问题）

## ✅ 安装完成检查清单

- [ ] Python 3.9+ 已安装
- [ ] 所有依赖已安装 (`pip install -r requirements.txt`)
- [ ] 目录结构已创建
- [ ] 所有代码文件已放置
- [ ] 所有 `__init__.py` 已创建
- [ ] `.env` 文件已配置
- [ ] LLM 服务已实现（或使用 Mock）
- [ ] 服务可以启动 (`python main.py`)
- [ ] 健康检查通过 (`curl /health`)
- [ ] 示例脚本运行成功

祝你使用愉快！🎉