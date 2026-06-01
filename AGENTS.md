# Pet Sales Customer Service Agent — Codex Instructions

## Project Overview
- **Stack**: LangGraph Agent + PyTorch CNN + Chroma RAG
- **Python**: 3.13, managed via `uv`
- **LLM**: DeepSeek-Reasoner (via OpenAI-compatible endpoint)
- **Scope**: 37 Oxford cat/dog breeds, pre-sale + post-sale scenarios
- **Interface**: CLI only (argparse), streaming token-by-token output
- **Reference project**: `D:/LangChain/traffy` — follow its patterns for agent/RAG

## Key Design Decisions (DO NOT CHANGE)
1. No standalone NLP pipeline — DeepSeek handles Chinese understanding natively
2. CNN is preprocessing, NOT an agent tool — breed recognition result injected into message context before Agent sees it
3. RAG is an `@tool`, NOT automatic pre-fetch — Agent decides when to retrieve
4. `deepseek-reasoner` model — accepts 5-15s thinking latency for quality
5. Model lazy loading — CNN loaded once on first inference, reused thereafter
6. Embedding fallback: DashScope → sentence-transformers (text2vec-base-chinese)

## Directory Structure
```
D:/LangChain/pet/
├── app/
│   ├── agents/          # pet_agent.py, custom_tools.py, rag_tools.py, system_prompt.py
│   ├── cnn/             # dataset.py, model.py, train.py, evaluate.py, inference.py
│   ├── common/          # logger.py
│   └── models/          # schemas.py (Pydantic models)
├── data/                # breed_info.txt, care_guide.txt, pricing.txt, health.txt, after_sales.txt
├── resources/           # chroma_db/, models/, outputs/
└── tests/               # test_agent.py, test_cnn.py
```

## Phase Implementation Order
- **Phase 1**: Scaffold (pyproject.toml, .env, logger, schemas)
- **Phase 2-5**: CNN pipeline (dataset → model → train → eval → inference)
- **Phase 6**: RAG vector store (run in parallel with CNN)
- **Phase 7**: Agent tools (search_pet_knowledge, tavily_web_search) — depends on 5+6
- **Phase 8**: Agent core (DeepSeek + SqliteSaver + streaming) — depends on 7
- **Phase 9**: CLI integration — depends on 8
- **Phase 10**: Tests — depends on 9

## Conventions
- Use `/` as path separator (Python auto-converts on Windows)
- All files UTF-8 encoding
- SQLite with `check_same_thread=False`
- Streaming: iterate `agent.stream()` with `stream_mode="messages"`, yield `AIMessageChunk.content`
- Error handling: `@tool` functions catch internally, return error strings to Agent
- Exit codes: 0=OK, 1=runtime error, 2=config error, 3=path not found

## Reference Patterns (from traffy)
- Agent creation: `create_agent(model, tools, checkpointer, system_prompt)` from `langchain.agents`
- Model init: `init_chat_model(model="deepseek-reasoner", model_provider="openai", ...)` from `langchain.chat_models`
- RAG: `DashScopeEmbeddings(model="text-embedding-v4")` + `Chroma` + `RecursiveCharacterTextSplitter`
- Memory: `SqliteSaver(sqlite3.connect(db, check_same_thread=False))`
- Streaming: `for chunk, metadata in agent.stream(..., stream_mode="messages")`
