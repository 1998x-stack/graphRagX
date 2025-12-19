"""
GraphRAG 使用示例
演示如何使用 API 创建索引和查询
"""
import asyncio
import httpx
import json


BASE_URL = "http://localhost:8000"


async def create_index_example():
    """示例1: 创建索引"""
    print("\n" + "="*50)
    print("示例 1: 创建索引")
    print("="*50)
    
    documents = [
        """
        Alice Johnson is a senior software engineer at TechCorp, 
        specializing in machine learning and artificial intelligence. 
        She has been with the company for 5 years and leads the AI research team.
        """,
        """
        TechCorp is a technology company founded in 2015, 
        focusing on developing cutting-edge AI solutions for enterprise clients. 
        The company is headquartered in San Francisco and has over 500 employees.
        """,
        """
        Bob Smith works as a product manager at TechCorp. 
        He collaborates closely with Alice Johnson on AI product development. 
        Bob previously worked at Google for 3 years.
        """,
        """
        Machine Learning is a subset of artificial intelligence that 
        enables systems to learn and improve from experience without being 
        explicitly programmed. It has applications in various domains including 
        natural language processing, computer vision, and recommendation systems.
        """
    ]
    
    request_data = {
        "index_id": "tech_knowledge_base",
        "documents": documents,
        "metadata": {
            "domain": "technology",
            "created_by": "example_script"
        }
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        print(f"\n发送索引请求到: {BASE_URL}/api/v1/index")
        print(f"文档数量: {len(documents)}")
        
        response = await client.post(
            f"{BASE_URL}/api/v1/index",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ 索引创建成功!")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\n❌ 索引创建失败: {response.status_code}")
            print(response.text)


async def local_search_example():
    """示例2: 局部搜索"""
    print("\n" + "="*50)
    print("示例 2: 局部搜索 (Local Search)")
    print("="*50)
    
    queries = [
        "What does Alice Johnson do?",
        "Tell me about TechCorp",
        "How are Alice and Bob related?"
    ]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for query in queries:
            print(f"\n查询: {query}")
            
            request_data = {
                "query": query,
                "index_id": "tech_knowledge_base",
                "mode": "local",
                "top_k": 5
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/query",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n答案:\n{result['answer']}")
                print(f"\n来源数量: {len(result['sources'])}")
                print(f"处理时间: {result['processing_time']:.2f}s")
            else:
                print(f"❌ 查询失败: {response.status_code}")


async def global_search_example():
    """示例3: 全局搜索"""
    print("\n" + "="*50)
    print("示例 3: 全局搜索 (Global Search)")
    print("="*50)
    
    queries = [
        "What are the main themes in this knowledge base?",
        "Summarize the key entities and their relationships",
        "What is the focus of this dataset?"
    ]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for query in queries:
            print(f"\n查询: {query}")
            
            request_data = {
                "query": query,
                "index_id": "tech_knowledge_base",
                "mode": "global"
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/query",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n答案:\n{result['answer']}")
                print(f"\n社区数量: {len(result['sources'])}")
                print(f"处理时间: {result['processing_time']:.2f}s")
            else:
                print(f"❌ 查询失败: {response.status_code}")


async def list_indexes_example():
    """示例4: 列出所有索引"""
    print("\n" + "="*50)
    print("示例 4: 列出所有索引")
    print("="*50)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/v1/indexes")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n找到 {result['count']} 个索引:")
            for index_id in result['indexes']:
                print(f"  - {index_id}")
        else:
            print(f"❌ 获取索引列表失败: {response.status_code}")


async def get_stats_example():
    """示例5: 获取索引统计"""
    print("\n" + "="*50)
    print("示例 5: 获取索引统计")
    print("="*50)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/v1/query/stats/tech_knowledge_base"
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n统计信息:")
            print(json.dumps(result['stats'], indent=2, ensure_ascii=False))
        else:
            print(f"❌ 获取统计失败: {response.status_code}")


async def health_check():
    """健康检查"""
    print("\n" + "="*50)
    print("健康检查")
    print("="*50)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/api/v1/health")
            if response.status_code == 200:
                result = response.json()
                print("\n✅ 服务正常运行")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return True
            else:
                print(f"\n❌ 服务异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"\n❌ 无法连接到服务: {e}")
            print(f"请确保服务正在运行: python main.py")
            return False


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("  GraphRAG API 使用示例")
    print("="*60)
    
    # 健康检查
    if not await health_check():
        return
    
    # 示例1: 创建索引
    await create_index_example()
    
    # 等待索引完成
    await asyncio.sleep(2)
    
    # 示例2: 局部搜索
    await local_search_example()
    
    # 示例3: 全局搜索
    await global_search_example()
    
    # 示例4: 列出索引
    await list_indexes_example()
    
    # 示例5: 获取统计
    await get_stats_example()
    
    print("\n" + "="*60)
    print("  所有示例执行完成!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())