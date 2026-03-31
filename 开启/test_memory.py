import sys
import os
sys.path.append(os.path.dirname(__file__))

from agent.memory import UnifiedLowLevelSemanticMemory, KnowledgeTriple

# 初始化记忆模块
ulsm = UnifiedLowLevelSemanticMemory()
print("✅ 双模态记忆模块初始化成功")

# 测试写入结构化知识与记忆
triple = KnowledgeTriple(head="买3斤送1斤", relation="等价于", tail="3斤的钱可购买4斤苹果")
mem_id = ulsm.add_memory(
    content="苹果单价5元/斤，买3斤送1斤，每4斤的成本为15元，单斤实际成本3.75元",
    task_type="math",
    is_experience=True,
    knowledge_triples=[triple]
)
print(f"✅ 记忆写入成功，记忆ID：{mem_id}")

# 测试双模态检索
results, triples = ulsm.search_memory("买苹果优惠计算", task_type="math")
print("\n📌 检索到的记忆内容：")
for item in results:
    print(f"- {item.content[:100]}...")
print("\n📌 检索到的结构化知识：")
for t in triples:
    print(f"- {t.head} {t.relation} {t.tail}")

# 测试系统统计
stats = ulsm.get_memory_stats()
print("\n📊 系统统计指标：")
for k, v in stats.items():
    print(f"{k}: {v}")

print("\n🎉 记忆模块所有测试全部通过！")