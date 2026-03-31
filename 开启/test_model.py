import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from agent.config import BNB_CONFIG, GLOBAL_DEVICE
from agent.model_loader import ModelLoader

# 测试模型加载
model_loader = ModelLoader()
model_loader.load_model()
print("✅ 模型加载成功！")

# 测试推理
inputs = model_loader.encode_text("1+1等于几？请给出完整的计算过程。")
input_ids = inputs["input_ids"].to(model_loader.model.device)
with torch.no_grad():
    outputs = model_loader.model.generate(
        input_ids=input_ids,
        max_new_tokens=200,
        temperature=0.3,
        pad_token_id=model_loader.tokenizer.eos_token_id
    )
response = model_loader.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"📌 推理结果：{response}")

# 显存监控
try:
    gpu_used = torch.cuda.memory_allocated(GLOBAL_DEVICE) / 1024 / 1024
except AttributeError:
    gpu_used = torch.cuda.memory_reserved(GLOBAL_DEVICE) / 1024 / 1024
gpu_total = torch.cuda.get_device_properties(GLOBAL_DEVICE).total_memory / 1024 / 1024
print(f"\n📊 显存使用情况：已用{gpu_used:.2f}MB / 总计{gpu_total:.2f}MB")

print("\n🎉 模型加载与推理测试全部通过！")