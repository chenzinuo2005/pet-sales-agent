# 🐾 萌宠之家 · 宠物销售智能体

基于 LangGraph + DeepSeek 的宠物店智能客服助手，集成 CNN 品种识别、RAG 知识库和 SSE 流式对话，覆盖 37 种猫狗品种的咨询、饲养、健康、价格全场景。

## 架构

```
Vue3 :5173 (Vite)  ←SSE→  FastAPI :8000  →  LangGraph Agent
                                  ↓              ├─ CNN (EfficientNet-B0)
                             直接调用           ├─ RAG (Chroma)
                                                └─ Web Search (Tavily)
```

## 快速开始

### 环境要求

- Python ≥ 3.13
- Node.js ≥ 18
- CUDA GPU（可选，CPU 也可运行）

### 安装

```bash
# 后端
git clone https://github.com/chenzinuo2005/pet-sales-agent.git
cd pet-sales-agent
uv sync

# 前端
cd frontend && npm install
```

### 配置

复制 `.env.example` 为 `.env` 并填入 API Key：

```env
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=sk-xxx           # DeepSeek Chat 大模型（必填）
DASHSCOPE_API_KEY=sk-xxx          # 阿里云嵌入模型（可选，自动降级 HuggingFace）
TAVILY_API_KEY=tvly-xxx           # Tavily 联网搜索（可选）
```

### 启动服务

```bash
# 终端 1 — 初始化知识库并启动后端
uv run python main.py init-rag
uv run python main.py serve

# 终端 2 — 前端
cd frontend && npm run dev
```

浏览器访问 `http://localhost:5173`

### CLI 命令

```bash
uv run python main.py chat              # 交互对话（支持 image: 前缀发送图片）
uv run python main.py init-rag          # 初始化 RAG 向量库
uv run python main.py serve             # 启动 FastAPI 服务
uv run python main.py train-cnn --data-root /path/to/oxford-pets  # 训练 CNN
uv run python main.py predict-cnn --image cat.jpg                  # 单图品种识别
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/v1/health` | 深度健康检查（数据库、CNN、RAG、LLM） |
| `POST` | `/api/v1/session` | 创建新会话，返回 `thread_id` |
| `POST` | `/api/v1/chat` | 文本对话（SSE 流式） |
| `POST` | `/api/v1/chat/upload` | 图片 + 文本对话（SSE 流式） |
| `GET` | `/api/v1/history/{thread_id}` | 获取会话历史 |
| `DELETE` | `/api/v1/history/{thread_id}` | 清除会话历史 |

## 项目结构

```
pet-sales-agent/
├── app/
│   ├── agents/                # LangGraph Agent
│   │   ├── pet_agent.py       # 对话编排（流式、CNN 集成）
│   │   ├── custom_tools.py    # RAG / Tavily 工具工厂
│   │   ├── rag_tools.py       # ChromaDB 加载、检索
│   │   └── system_prompt.py   # 系统提示词（只答所问、反模板、反照搬）
│   ├── cnn/                   # CNN 品种识别
│   │   ├── model.py           # EfficientNet-B0（37 类，87%+ 准确率）
│   │   ├── dataset.py         # Oxford-IIIT Pet 数据加载
│   │   ├── train.py           # 训练脚本
│   │   ├── inference.py       # 单图推理
│   │   └── evaluate.py        # 评估 + 混淆矩阵
│   ├── api/                   # FastAPI
│   │   ├── server.py          # SSE 流式端点 + 异常处理
│   │   ├── middleware.py      # RequestID / Auth / RateLimit 中间件栈
│   │   └── deps.py            # 依赖注入
│   ├── common/                # 基础设施
│   │   ├── config.py          # Pydantic Settings（环境变量 + 路径解析）
│   │   ├── container.py       # DI 容器（LLM/Embeddings/向量库/CNN 懒加载）
│   │   ├── exceptions.py      # 统一异常体系
│   │   └── logger.py          # JSON/文本双格式日志
│   ├── models/
│   │   └── schemas.py         # Pydantic 请求/响应模型
│   └── main.py                # CLI 命令路由
├── frontend/src/              # Vue3 聊天界面
├── data/                      # RAG 知识库文件
│   └── breeds_encyclopedia.txt # 37 品种完整百科（按品种整合，4 章节/品种）
├── resources/                 # 运行时数据（向量库、模型权重、SQLite）
│   ├── models/pet_cnn.pth     # CNN 模型权重
│   └── chroma_db/             # Chroma 持久化向量库
├── scripts/
│   └── build_rag_data.py      # 知识库数据构建脚本
├── tests/                     # pytest 测试（57 用例）
├── pyproject.toml             # uv 项目配置
└── main.py                    # 入口
```

## 技术栈

| 模块 | 技术 |
|------|------|
| Agent 框架 | LangGraph + DeepSeek-Chat |
| 图片识别 | PyTorch + EfficientNet-B0（87%+ on Oxford Pets） |
| 向量检索 | ChromaDB + DashScope text-embedding-v4 |
| 后端 | FastAPI + SSE 流式 + 中间件栈 |
| 前端 | Vue3 + Vite（SSE 代理） |
| 会话持久化 | SQLite（LangGraph SqliteSaver） |
| 依赖管理 | uv |

## 核心特性

**品种识别 + 对话融合：** 用户上传宠物图片后，CNN 自动识别品种（37 类），将识别结果注入对话上下文，Agent 基于品种信息给出针对性回答。

**RAG 知识库（按品种整合）：** 每品种为一个完整检索单元，包含基本特征、饲养指南、健康提示、价格参考四部分。检索命中某品种即可获得该品种的全貌信息，避免碎片化。

**提示词优化：** 系统提示词经过多轮迭代，核心规则包括"只答所问"（用户问喂食只讲喂食）、反模板句式、emoji 控制、连带健康提醒等，使对话更自然聚焦。

**SSE 流式对话：** 前后端通过 SSE 逐 token 推送，Vite 代理已配置禁用缓冲，确保实时体验。

**企业级中间件栈：** RequestID 全链路追踪 + API Key 认证 + IP 限流（每分钟 30 次），错误响应统一格式。

## 作者闲言

这一切始于一个简单的发现：**DeepSeek 是纯文本模型，没有视觉能力。**

用户发来一张猫照片问"这是什么品种"，DeepSeek 只能干瞪眼。而市面上支持视觉的大模型，要么贵，要么慢，要么中文能力不行。

于是就有了这个项目：**给 DeepSeek 装一双眼睛。**

我们用 PyTorch 训练了一个 EfficientNet-B0，在牛津宠物数据集上做到 87%+ 的识别准确率，覆盖 37 种猫狗品种。当用户上传图片时，CNN 先认出品种，然后把识别结果像这样塞进对话：

> [系统：图片识别结果 — 波斯猫，置信度 92.3%]
> 用户：这是什么猫？

DeepSeek 拿到这条上下文后，就能像亲眼见过这只猫一样，结合 RAG 知识库里波斯猫的完整百科——性格、饲养、健康、价格——给出靠谱的回答。

这个架构的妙处在于：**CNN 负责"看"，DeepSeek 负责"聊"，两者通过一行字符串优雅地缝合，完全不改模型本身。** 对 DeepSeek 来说，品种信息只是消息里多出来的一段文字，它甚至不知道背后有一张图片存在。

如果你也在用 DeepSeek 做需要图片理解的应用，希望这个思路对你有启发。

## License

MIT
