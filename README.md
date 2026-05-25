# RAG-Memory-Agent

FastAPI + RAG + 记忆（PostgreSQL）。

## 接口

- `GET /`：Web UI
- `POST /api/chat`：对话
- `GET /api/models?provider=...`：模型列表
- `POST /api/upload`：上传并索引文档

## v2.0 更新说明

本次更新将项目从基础 RAG + 记忆助手升级为可量化评估的工程化 Agent 系统，新增 RAGAS 评测、LLM-as-Judge、检索优化和模型路由 benchmark。

### 核心更新

- 新增 RAGAS 评测脚本，支持检索质量与回答忠实度评估
- 新增 LLM-as-Judge 评测脚本，补充回答相关性与事实正确性指标
- 新增 10 万字企业制度模拟知识库，支持无真实业务文档时进行完整评测
- 新增多套测试集，覆盖常规问答、跨分块问题、模糊口语问题和无答案问题
- 优化 RAG 检索链路：标题感知分块、章节标题注入、多 Query 检索、中文关键词召回、RRF 融合、多轮上下文检索
- 新增 benchmark 脚本，对比固定大模型方案与三档模型路由方案的成本和延迟
- 输出完整实验报告，便于复现实验结果和量化项目收益

### 量化结果

- `Context Precision`：`0.8072`
- `Context Recall`：`0.9000`
- `Faithfulness`：`0.8534`
- `Answer Relevancy`：`0.8567`
- `Answer Correctness`：`0.8867`
- 模型调用成本下降：`89.15%`
- 平均响应延迟下降：`70.15%`
- P95 延迟下降：`56.30%`

### 主要文件

- `scripts/ragas_eval.py`
- `scripts/llm_judge_eval.py`
- `scripts/benchmark_routing.py`
- `outputs/routing_benchmark_report.md`
- `data/simulated_enterprise_kb_100k.md`
- `data/ragas_enterprise_kb_testset.csv`
- `data/ragas_enterprise_edge_testset.csv`

## 启动

先按 `.env.example` 准备 `.env`（包含 LLM/Embedding Key 与数据库配置）。

### Docker

```bash
docker compose up -d --build
```

### 本地

```bash
pip install -r requirements.txt
python main.py
```
