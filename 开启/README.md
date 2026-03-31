# ULM-MARv2 视觉推理代理

这是一个基于视觉与文本混合推理的代理项目骨架，目标是构建一个可运行的 `ULM-MARv2` 风格系统。

## 目标

- 提供多模态任务调度与回答
- 支持 A100 + RTX3060 的混合部署策略
- 包括记忆管理、模型加载与安全控制模块
- 提供简单的 FastAPI 服务

## 项目结构

- `agent/`
  - `config.py`：模型与系统配置
  - `model_loader.py`：模型与 tokenizer 加载器
  - `memory.py`：记忆存储与检索模块（已实现UnifiedLowLevelSemanticMemory）
  - `agent_core.py`：代理主逻辑
- `app.py`：FastAPI 服务入口
- `test_memory.py`：记忆模块测试
- `test_model.py`：模型加载测试
- `requirements.txt`：依赖列表
- `setup_env.ps1`：Windows/WSL 环境安装脚本

## 快速开始

1. 安装 Miniconda 或 Anaconda
2. 运行 `setup_env.ps1`
3. 启动服务：

```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

4. 访问 API：

```bash
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"input_text": "请解释这张图像的关键要点。"}'
```

## 依赖

参见 `requirements.txt`

## 当前实现状态

✅ **已完成**：
- 双模态统一底层语义记忆 (ULSM) 模块
- 结构化知识图谱与三元组存储
- 分层记忆管理（短期/热长期/冷长期）
- LZ4 压缩与 FAISS 向量检索
- 记忆重要性评分与保护机制
- 经验蒸馏与结构化推理轨迹存储

🔄 **进行中**：
- 多智能体调度与推理分布对齐
- MIRA (记忆引导推理对齐) 机制
- 强/弱智能体联合推理

## 测试

运行记忆模块测试：
```bash
python test_memory.py
```

## 注意

本项目当前为骨架实现，主要用于快速搭建模型推理、记忆模块和 API 接口。后续可根据需求补充多模态输入、视觉理解和任务调度逻辑。