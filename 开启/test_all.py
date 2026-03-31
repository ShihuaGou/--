#!/usr/bin/env python3
"""
ULM-MARv2 系统快速测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_memory():
    """测试记忆模块"""
    print("=== 测试记忆模块 ===")
    try:
        from agent.memory import UnifiedLowLevelSemanticMemory, KnowledgeTriple
        ulsm = UnifiedLowLevelSemanticMemory()
        print("✅ 双模态记忆模块初始化成功")

        # 测试写入
        triple = KnowledgeTriple(head="买3斤送1斤", relation="等价于", tail="3斤的钱可购买4斤苹果")
        mem_id = ulsm.add_memory(
            content="苹果单价5元/斤，买3斤送1斤，每4斤的成本为15元，单斤实际成本3.75元",
            task_type="math",
            is_experience=True,
            knowledge_triples=[triple]
        )
        print(f"✅ 记忆写入成功，记忆ID：{mem_id}")

        # 测试检索
        results, triples = ulsm.search_memory("买苹果优惠计算", task_type="math")
        print(f"📌 检索到 {len(results)} 条记忆，{len(triples)} 个知识三元组")

        # 测试统计
        stats = ulsm.get_memory_stats()
        print(f"📊 记忆统计：{stats}")

        print("🎉 记忆模块测试通过！\n")
        return True
    except Exception as e:
        print(f"❌ 记忆模块测试失败: {e}\n")
        return False

def test_model():
    """测试模型加载"""
    print("=== 测试模型加载 ===")
    try:
        from agent.model_loader import ModelLoader
        loader = ModelLoader()
        loader.load_model()
        print("✅ 模型加载成功")

        # 测试推理
        response = loader.generate("1+1等于几？请给出完整的计算过程。", max_new_tokens=100)
        print(f"📌 推理结果：{response[:200]}...")

        print("🎉 模型测试通过！\n")
        return True
    except Exception as e:
        print(f"❌ 模型测试失败: {e}\n")
        return False

def test_agent():
    """测试代理"""
    print("=== 测试代理 ===")
    try:
        from agent.agent_core import ULMMarAgent
        from agent.model_loader import ModelLoader
        loader = ModelLoader()
        loader.load_model()
        agent = ULMMarAgent(model_loader=loader)
        print("✅ 代理初始化成功")

        # 测试查询
        output, summary = agent.process_query("计算：2+2等于几？")
        print(f"📌 代理回答：{output}")
        print(f"📊 记忆摘要：{summary}")

        print("🎉 代理测试通过！\n")
        return True
    except Exception as e:
        print(f"❌ 代理测试失败: {e}\n")
        return False

def main():
    """主测试函数"""
    print("🚀 ULM-MARv2 系统测试开始\n")

    results = []
    results.append(("记忆模块", test_memory()))
    results.append(("模型加载", test_model()))
    results.append(("代理系统", test_agent()))

    print("=== 测试结果汇总 ===")
    passed = 0
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name}: {status}")
        if success:
            passed += 1

    print(f"\n总计：{passed}/{len(results)} 个测试通过")

    if passed == len(results):
        print("🎉 所有测试通过！系统准备就绪。")
        print("\n启动 FastAPI 服务：")
        print("uvicorn app:app --reload --host 0.0.0.0 --port 8000")
    else:
        print("⚠️ 部分测试失败，请检查依赖和配置。")

if __name__ == "__main__":
    main()