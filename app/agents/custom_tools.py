"""LangChain @tool functions — created via factory for DI."""
from langchain_core.tools import tool

from app.common.logger import get_logger

logger = get_logger(__name__)


def create_search_pet_knowledge(container):
    """Create the search_pet_knowledge tool bound to the container's vector store."""
    @tool
    def search_pet_knowledge(query: str) -> str:
        """搜索宠物知识库，获取品种信息、饲养指南、价格参考、健康问题、售后政策。

        Args:
            query: 自然语言查询

        Returns:
            拼接后的检索结果字符串
        """
        try:
            from app.agents import rag_tools
            result = rag_tools.retrieve_with_store(container.get_vector_store(), query, k=3)
            if not result or not result.strip():
                return "知识库中暂无相关信息"
            return result
        except Exception as e:
            logger.warning("rag_search_failed", extra={"error": str(e)})
            return "知识库检索暂时不可用，请稍后重试"
    return search_pet_knowledge


def create_tavily_web_search(container):
    """Create the tavily_web_search tool bound to the container's Tavily client."""
    @tool
    def tavily_web_search(query: str) -> str:
        """搜索互联网获取宠物相关信息。

        Args:
            query: 搜索查询词

        Returns:
            搜索结果摘要 (最多5条)
        """
        try:
            tavily = container.get_tavily()
            result = tavily.invoke(query)
            if not result:
                return "未找到相关信息"
            return str(result)
        except Exception as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ("429", "rate limit", "too many requests")):
                return "搜索请求太频繁，请稍后重试"
            if any(kw in error_msg for kw in ("401", "unauthorized", "403", "forbidden")):
                return "搜索服务未正确配置"
            logger.error("web_search_failed", extra={"error": str(e)})
            return "网络搜索暂时不可用，请稍后重试"
    return tavily_web_search
