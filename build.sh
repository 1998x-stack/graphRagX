#!/bin/bash

# Create directories
mkdir -p models
mkdir -p services
mkdir -p core
mkdir -p workflows
mkdir -p api
mkdir -p utils
mkdir -p prompts
mkdir -p outputs/llm_logs outputs/graphs outputs/communities

# Create root files
touch main.py
touch config.py

# Create models/ files
touch models/__init__.py
touch models/schemas.py
touch models/graph.py

# Create services/ files
touch services/__init__.py
touch services/llm_service.py
touch services/embedding_service.py
touch services/storage_service.py

# Create core/ files
touch core/__init__.py
touch core/chunking.py
touch core/extraction.py
touch core/graph_builder.py
touch core/community.py
touch core/summarization.py

# Create workflows/ files
touch workflows/__init__.py
touch workflows/indexing_workflow.py
touch workflows/query_workflow.py

# Create api/ files
touch api/__init__.py
touch api/index.py
touch api/query.py

# Create utils/ files
touch utils/__init__.py
touch utils/logger.py
touch utils/concurrency.py
touch utils/json_extractor.py

# Create prompts/ files
touch prompts/__init__.py
touch prompts/extraction_prompts.py
touch prompts/summary_prompts.py

echo "Directory structure and files created successfully!"
echo "Total files created: $(find . -name "*.py" -o -name "*.json" | wc -l)"