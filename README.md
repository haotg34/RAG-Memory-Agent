# RAG-Memory-Agent

FastAPI + RAG + 记忆（PostgreSQL）。

## 接口

- `GET /`：Web UI
- `POST /api/chat`：对话
- `GET /api/models?provider=...`：模型列表
- `POST /api/upload`：上传并索引文档

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
