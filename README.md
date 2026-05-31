# 萌宠之家 — 宠物客服 AI 助手

基于 LangGraph + DeepSeek 的智能宠物客服系统，支持图片品种识别、知识库检索和联网搜索。

## 架构

```
Vue3 :5173 (Vite)  ←SSE→  FastAPI :8000  →  LangGraph Agent
                                  ↓              ├─ CNN (ResNet50)
                             直接调用           ├─ RAG (Chroma)
                                                └─ Web Search (Tavily)
```

## 快速开始

### 环境要求

- Python ≥ 3.13
- Node.js ≥ 18
- CUDA GPU (可选，CPU 也可运行)

### 安装

```bash
# 后端
cd D:/LangChain/pet
uv sync

# 前端
cd frontend && npm install
```

### 配置

复制 `.env.example` 为 `.env` 并填入 API Key：

```env
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=sk-xxx
DASHSCOPE_API_KEY=sk-xxx   # 嵌入模型（可选，自动降级）
TAVILY_API_KEY=tvly-xxx     # 联网搜索（可选）
```

### 初始化知识库

```bash
uv run python -c "from app.agents.rag_tools import init_vector_store; init_vector_store()"
```

### 训练图片识别模型

```bash
uv run python -m app.cnn.train /path/to/oxford-iiit-pet
```

### 启动服务

```bash
# 终端 1 — 后端
uv run python -m app.main serve

# 终端 2 — 前端
cd frontend && npm run dev
```

浏览器访问 `http://localhost:5173`

### 命令行模式

```bash
uv run python -m app.main chat              # 文本对话
uv run python -m app.main chat -i cat.jpg   # 图片识别 + 对话
```

## 项目结构

```
app/
├── agents/         # LangGraph Agent (RAG 工具、系统提示)
├── cnn/            # ResNet50 品种识别 (训练、推理、评估)
├── api/            # FastAPI 服务 (SSE 流式)
├── models/         # Pydantic 数据模型
└── common/         # 日志配置
frontend/src/       # Vue3 聊天界面
resources/          # 模型权重、向量库、SQLite
data/               # 知识库 txt 文件
```

## 技术栈

| 模块 | 技术 |
|------|------|
| Agent 框架 | LangGraph + DeepSeek-Reasoner |
| 图片识别 | PyTorch + ResNet50 (87.5%+ on Oxford Pets) |
| 向量检索 | Chroma + DashScope Embeddings |
| 后端 | FastAPI + SSE 流式 |
| 前端 | Vue3 + Vite |
| 数据库 | SQLite (会话持久化) |

## License

MIT
