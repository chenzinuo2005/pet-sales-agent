from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from app.common.logger import logger
from app.agents import rag_tools

# 初始化 TavilySearch（API Key 未配置时置为 None）
try:
    tavily = TavilySearch(max_results=5, topic="general")
except Exception as e:
    logger.warning(f"TavilySearch 初始化失败: {e}")
    tavily = None


@tool
def search_pet_knowledge(query: str) -> str:
    """搜索宠物知识库，获取品种信息、饲养指南、价格参考、健康问题、售后政策。

    Args:
        query: 自然语言查询，如 "金毛犬价格" 或 "折耳猫常见健康问题"

    Returns:
        拼接后的检索结果字符串，格式为 [来源:xxx.txt] chunk1 [来源:xxx.txt] chunk2 ...
        未检索到时返回 "知识库中暂无相关信息"
    """
    try:
        result = rag_tools.retrieve(query, k=3)
        if not result or not result.strip():
            return "知识库中暂无相关信息"
        return result
    except FileNotFoundError:
        return "知识库暂未初始化，请先运行 python -m app.main init-rag"
    except Exception as e:
        logger.error(f"知识库检索失败: {e}")
        return "知识库暂未初始化，请先运行 python -m app.main init-rag"


@tool
def tavily_web_search(query: str) -> str:
    """搜索互联网获取宠物相关信息。

    Args:
        query: 搜索查询词

    Returns:
        搜索结果摘要 (最多5条)
    """
    if tavily is None:
        return "网络搜索功能未启用，请配置 TAVILY_API_KEY"
    try:
        result = tavily.invoke(query)
        if not result:
            return "未找到相关信息"
        return str(result)
    except Exception as e:
        error_msg = str(e).lower()
        if any(kw in error_msg for kw in ("429", "rate limit", "too many requests")):
            return "搜索服务暂时不可用"
        logger.error(f"网络搜索失败: {e}")
        return "搜索服务暂时不可用"
