# 大模型路由与级联优化实验报告

## 实验目标

对比固定大模型调用方案与混合路由+级联方案在成本、延迟、模型命中分布上的差异，验证多模型调度是否能在保证可用性的前提下降低调用成本和响应时间。

## 实验设置

- 测试集：`data/ragas_employee_handbook_testset.csv`
- 样本数：20
- Baseline：所有请求强制使用 `qwen-max`
- Optimized：启用规则路由 + 小模型难度评分 + 级联自检
- 三档模型：`qwen-turbo` / `qwen-plus` / `qwen-max`
- 成本口径：按默认相对单价估算，small=0.1、medium=0.4、large=1.0（每百万 token 输入/输出同价）

## 核心结果

| 指标 | Baseline | Optimized | 改善 |
|---|---:|---:|---:|
| 总成本估算 | 0.007257 | 0.0007871 | 下降 89.15% |
| 平均延迟 | 21620.69 ms | 6454.71 ms | 下降 70.15% |
| P95 延迟 | 41866.45 ms | 18294.86 ms | 下降 56.30% |
| 样本数 | 20 | 20 | - |

## 路由分布

| 模型档位 | 命中数 | 命中率 |
|---|---:|---:|
| small | 19 | 95.00% |
| large | 1 | 5.00% |

- 升级率：0.00%
- 降级/失败：本轮未出现服务失败

## 结论

本轮实验表明，在员工手册类高频问答场景中，多数请求可以被小模型稳定处理。引入混合路由和级联机制后，系统在 20 条测试样本上实现了约 89.15% 的成本下降和 70.15% 的平均延迟下降，同时保留了对敏感/复杂问题升级到高配模型的能力。

## RAG检索质量补充实验

- 知识库：`data/simulated_enterprise_kb_100k.md`
- 优化前入库分块：367
- 优化后入库分块：184
- 测试集：`data/ragas_enterprise_kb_testset.csv`
- 样本数：30
- 评测方式：RAGAS retrieval-only（`context_precision` / `context_recall`）

| 指标 | 优化前 | 优化后 | 提升 |
|---|---:|---:|---:|
| Context Precision | 0.7500 | 0.8072 | +7.63% |
| Context Recall | 0.7833 | 0.9000 | +14.89% |

本轮优化包括 Markdown 标题感知分块、章节标题注入、chunk overlap 提升、多查询融合、中文关键词召回、RRF 融合和多轮上下文检索。优化后在 10 万字级企业制度知识库上，Context Precision 从 0.7500 提升到 0.8072，Context Recall 从 0.7833 提升到 0.9000。

## 极端场景检索实验

- 测试集：`data/ragas_enterprise_edge_testset.csv`
- 样本数：12
- 覆盖场景：跨分块问题、模糊口语问题、无答案问题、冲突处理问题

| 指标 | 得分 |
|---|---:|
| Context Precision | 0.7880 |
| Context Recall | 0.8194 |

结果表明，优化后的检索链路在模糊问法和跨分块问题下仍能保持较好的上下文召回能力。

## 回答端质量补充实验

- 测试集：`data/ragas_enterprise_kb_testset.csv`
- 样本数：30
- Faithfulness：RAGAS
- Answer Relevancy / Answer Correctness：LLM-as-Judge（绕过 DashScope embedding 兼容导致的 RAGAS `nan`）

| 指标 | 得分 |
|---|---:|
| Faithfulness | 0.8534 |
| Answer Relevancy | 0.8567 |
| Answer Correctness | 0.8867 |

结果表明，系统回答在忠实度、问题相关性和事实正确性上均达到 0.85 以上，适合企业制度问答这类要求低幻觉和高一致性的场景。

## Agent状态机与结构化输出实验

- 测试集：`data/agent_state_benchmark_questions.csv`
- 样本数：50
- 覆盖场景：闲聊、普通制度问答、敏感信息、复杂问题、模糊问法、跨分块、无答案、上下文指代
- 评测脚本：`scripts/benchmark_agent_state.py`

| 指标 | 得分 |
|---|---:|
| 请求成功率 | 100.00% |
| 状态链路完整率 | 100.00% |
| 路由结构化解析成功率 | 100.00% |
| 自检结构化解析成功率 | 100.00% |

实验结果表明，引入显式 Agent 状态机和 JSON 结构化输出约束后，系统能够稳定记录 `route / retrieve / generate / validate / persist / done` 等关键阶段，并保证路由评分、自检判断等中间结果可解析、可观测、可复盘。

## 口径说明

成本与延迟实验使用员工手册类高频问答样本，重点验证模型路由与级联策略的收益；检索质量实验使用 10 万字企业制度知识库与匹配测试集，重点验证 RAG 检索链路的上下文精度和召回率。RAGAS 的 Answer Correctness 在当前 DashScope embedding 组合下会出现 `nan`，因此回答正确性与相关性采用独立 LLM-as-Judge 复核。

