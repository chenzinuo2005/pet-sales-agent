from collections.abc import Generator

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from app.common.logger import logger
from app.agents.custom_tools import search_pet_knowledge, tavily_web_search
from app.agents.system_prompt import SYSTEM_PROMPT
from app.models.schemas import CNNPredictResult
import os
import sqlite3
import uuid

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="deepseek-reasoner",
    model_provider="openai",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

db_path = os.path.join(os.path.dirname(__file__), "../../resources/pet_agent.db")
connection = sqlite3.connect(db_path, check_same_thread=False)
checkpointer = SqliteSaver(connection)
checkpointer.setup()

agent = create_agent(
    model=model,
    tools=[search_pet_knowledge, tavily_web_search],
    checkpointer=checkpointer,
    system_prompt=SYSTEM_PROMPT,
)


def _build_breed_hint(result: CNNPredictResult) -> str:
    """根据 CNN 识别结果构造品种提示文本。"""
    conf_pct = result.confidence * 100
    if result.status == "success":
        return (
            f"[系统: 图片识别结果 - {result.breed_cn}"
            f" (置信度 {conf_pct:.1f}%)]"
        )
    elif result.status == "low_confidence":
        if result.confidence >= 0.60:
            return (
                f"[系统: 图片识别结果 - 可能是 {result.breed_cn}"
                f" ({conf_pct:.1f}%)，不太确定]"
            )
        else:
            top3_str = "/".join(t["breed_cn"] for t in result.top3[:3])
            return f"[系统: 图片识别结果 - 不太确定，可能是 {top3_str}]"
    else:
        top3_str = "/".join(t["breed_cn"] for t in result.top3[:3])
        return f"[系统: 图片识别结果 - 置信度较低，仅供参考: {top3_str}]"


def chat_with_agent(message: str, image_path: str, thread_id: str) -> Generator[str, None, None]:
    """流式调用 Agent 聊天，逐 token 返回回复。

    Args:
        message: 用户输入文本
        image_path: 图片本地路径（可空）
        thread_id: 会话 ID（空时自动生成 UUID）
    """
    logger.info(f"[用户]: {message}, image: {image_path}, thread_id: {thread_id}")
    try:
        if not thread_id:
            thread_id = str(uuid.uuid4())

        augmented_message = message
        if image_path and image_path.strip():
            try:
                from app.cnn.inference import predict_breed

                result = predict_breed(image_path)
                breed_hint = _build_breed_hint(result)
                augmented_message = (
                    f"{breed_hint}\n用户问题: {message or '这是什么品种？'}"
                )
            except (ImportError, FileNotFoundError) as e:
                logger.warning(f"CNN 模型不可用: {e}，直接传递用户问题")
            except Exception as e:
                logger.warning(f"图片识别失败: {e}，直接传递用户问题")

        msg = HumanMessage(content=augmented_message)

        for chunk, metadata in agent.stream(
            {"messages": [msg]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content

    except Exception as e:
        logger.error(f"Agent 错误: {e}")
        yield "小宠正在休息，请稍后再试"


def clear_messages(thread_id: str) -> None:
    """清空会话历史。"""
    logger.info(f"清空历史消息，thread_id: {thread_id}")
    checkpointer.delete_thread(thread_id)


def get_messages(thread_id: str) -> list[dict[str, str]]:
    """获取会话历史，返回 [{\"role\": str, \"content\": str}] 格式。"""
    logger.info(f"获取历史消息，thread_id: {thread_id}")

    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    if not checkpoint:
        return []

    channel_values = checkpoint.get("channel_values")
    if not channel_values:
        return []

    messages = channel_values.get("messages", [])
    if not messages:
        return []

    result = []
    for msg in messages:
        if not msg.content:
            continue
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})

    return result
