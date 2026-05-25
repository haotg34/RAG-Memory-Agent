# RAG-Memory-Agent

企业知识库 RAG + 三层记忆 AI 助理。项目基于 FastAPI、LangChain、PostgreSQL 和 OpenAI-compatible API 构建，支持文档知识库问答、多轮记忆、用户偏好、三档模型路由、级联升级、Agent 状态机与自动化评测。

## 核心能力

- 企业知识库 RAG：文档上传、解析、标题感知分块、Embedding 入库、混合检索、RRF 融合、LLM 重排序、基于上下文生成回答
- 三层记忆：短期会话窗口、长期记忆向量召回、用户偏好抽取与更新
- 多轮上下文检索：原始问题、改写问题、近期用户问题共同参与召回，缓解指代不明和上下文遗忘
- 大模型路由：规则路由 + 小模型难度评分，自动分配 `small / medium / large` 三档模型
- 级联升级：小模型优先回答，自检不通过时自动升级高配模型重生成
- Agent 状态机：显式记录 `route / retrieve / generate / validate / persist / done` 等阶段，降低复杂链路中的步骤错乱和关键信息遗漏
- 结构化输出约束：路由评分与回答自检使用 JSON 结构化输出，提升解析稳定性
- 自动化评测：RAGAS、LLM-as-Judge、成本/延迟 benchmark、状态机稳定性 benchmark

## 架构流程

```text
用户请求
→ Agent 状态机初始化
→ 混合路由器（规则 + 难度评分）
→ 多 Query 检索（原问题 + 改写问题 + 最近用户问题）
→ 语义检索 + 中文关键词检索
→ RRF 融合 + LLM 重排序
→ 三层记忆注入
→ 小模型优先生成
→ JSON 自检
→ 必要时升级大模型
→ 写入记忆、用量日志、路由日志和状态机事件
→ 返回答案
```

## 量化结果

### RAG 检索与回答质量

基于 10 万字企业制度模拟知识库、184 个优化后知识分块、30 条企业制度测试样本评测：

| 指标 | 得分 |
|---|---:|
| Context Precision | 0.8072 |
| Context Recall | 0.9000 |
| Faithfulness | 0.8534 |
| Answer Relevancy | 0.8567 |
| Answer Correctness | 0.8867 |

检索优化前后对比：

| 指标 | 优化前 | 优化后 | 提升 |
|---|---:|---:|---:|
| Context Precision | 0.7500 | 0.8072 | +7.63% |
| Context Recall | 0.7833 | 0.9000 | +14.89% |

### 路由成本与延迟

对比固定使用 `qwen-max` 的 baseline 与三档模型路由方案：

| 指标 | Baseline | Optimized | 改善 |
|---|---:|---:|---:|
| 总成本估算 | 0.007257 | 0.0007871 | 下降 89.15% |
| 平均延迟 | 21620.69 ms | 6454.71 ms | 下降 70.15% |
| P95 延迟 | 41866.45 ms | 18294.86 ms | 下降 56.30% |

### Agent 状态机稳定性

基于 50 条混合测试样本，覆盖闲聊、普通问答、敏感信息、复杂问题、模糊问法、跨分块、无答案和上下文指代：

| 指标 | 得分 |
|---|---:|
| 请求成功率 | 100.00% |
| 状态链路完整率 | 100.00% |
| 路由结构化解析成功率 | 100.00% |
| 自检结构化解析成功率 | 100.00% |

完整报告见：

```text
outputs/routing_benchmark_report.md
```

## 接口

- `GET /`：Web UI
- `POST /api/chat`：对话
- `GET /api/models?provider=...`：模型列表
- `POST /api/upload`：上传并索引文档

## 快速启动

先按 `.env.example` 准备 `.env`，配置 LLM、Embedding 和数据库连接。

### Docker

```bash
docker compose up -d --build
```

### 本地

```bash
pip install -r requirements.txt
python main.py
```

## 三档模型配置

示例（DashScope / Qwen）：

```env
LLM_PROVIDER=dashscope
LLM_SMALL_MODEL=qwen-turbo
LLM_MEDIUM_MODEL=qwen-plus
LLM_LARGE_MODEL=qwen-max
```

不传 `llm_model` 时，`/api/chat` 会自动启用路由与级联升级；传入 `llm_model` 时会绕过路由，便于调试指定模型。

## 评测与实验

### RAGAS 检索质量

```bash
docker compose exec -T app python scripts/ragas_eval.py \
  --input data/ragas_enterprise_kb_testset.csv \
  --metric-set retrieval \
  --output outputs/ragas_enterprise_retrieval_results_optimized.csv
```

### 回答端 LLM-as-Judge

```bash
docker compose exec -T app python scripts/llm_judge_eval.py \
  --input outputs/ragas_enterprise_faithfulness_collected.csv \
  --output outputs/llm_judge_answer_metrics.csv
```

### 路由成本与延迟

```bash
docker compose exec -T app python scripts/benchmark_routing.py \
  --input data/ragas_employee_handbook_testset.csv
```

### Agent 状态机稳定性

```bash
docker compose exec -T app python scripts/benchmark_agent_state.py \
  --input data/agent_state_benchmark_questions.csv
```

## 主要文件

- `services/chat_service.py`：对话主链路，整合 RAG、记忆、路由、级联和状态机
- `services/router_service.py`：混合路由、结构化难度评分、回答自检
- `services/agent_state.py`：Agent 状态机
- `modules/rag/loader.py`：标题感知分块
- `modules/rag/retriever.py`：语义检索、关键词检索、RRF 融合、重排序
- `scripts/ragas_eval.py`：RAGAS 评测
- `scripts/llm_judge_eval.py`：回答端 LLM-as-Judge
- `scripts/benchmark_routing.py`：成本与延迟 benchmark
- `scripts/benchmark_agent_state.py`：状态机稳定性 benchmark
- `outputs/routing_benchmark_report.md`：完整实验报告

## 技术栈

Python / FastAPI / LangChain / SQLAlchemy / PostgreSQL / Docker / RAG / Embedding / RAGAS / LLM-as-Judge / OpenAI-compatible API / Qwen（DashScope）
