"""Core LangGraph agent — chat logic (DI-ready)."""
import uuid
from collections.abc import Generator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.common.logger import get_logger
from app.models.schemas import CNNPredictResult

logger = get_logger(__name__)


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


def chat_with_agent(
    agent,
    checkpointer,
    message: str,
    image_path: str | None = None,
    thread_id: str = "",
) -> Generator[str]:
    """流式调用 Agent 聊天。

    Args:
        agent: LangGraph agent instance (injected)
        checkpointer: SqliteSaver instance (injected)
        message: 用户输入文本
        image_path: 图片本地路径（可空）
        thread_id: 会话 ID（空时自动生成 UUID）
    """
    logger.info("user_message", extra={"user_message": message, "image_path": image_path, "thread_id": thread_id})
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
            except Exception as e:
                logger.warning("cnn_inference_failed", extra={"error": str(e)})

        for chunk, _metadata in agent.stream(
            {"messages": [HumanMessage(content=augmented_message)]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content  # type: ignore[misc]
    except Exception as e:
        logger.error("agent_error", extra={"error": str(e)})
        yield "小宠正在休息，请稍后再试"


def clear_messages(checkpointer, thread_id: str) -> None:
    """清空会话历史。"""
    checkpointer.delete_thread(thread_id)


def get_messages(checkpointer, thread_id: str) -> list[dict[str, str]]:
    """获取会话历史。"""
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


# ---- Container-based convenience wrappers (for CLI / simpler callers) ----

def chat_with_container(container, message: str, image_path: str | None = None, thread_id: str = "") -> Generator[str]:
    """Convenience: chat using the DI container to resolve agent/checkpointer."""
    return chat_with_agent(
        agent=container.get_agent(),
        checkpointer=container.get_checkpointer(),
        message=message,
        image_path=image_path,
        thread_id=thread_id,
    )


def clear_messages_with_container(container, thread_id: str) -> None:
    """Convenience: clear history using the DI container."""
    clear_messages(container.get_checkpointer(), thread_id)


def get_messages_with_container(container, thread_id: str) -> list[dict[str, str]]:
    """Convenience: get history using the DI container."""
    return get_messages(container.get_checkpointer(), thread_id)
