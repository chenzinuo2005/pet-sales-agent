import os
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

from app.common.logger import logger

# 路径配置
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../../resources/chroma_db")

# 嵌入模型 (DashScope 优先, sentence-transformers 备选)
try:
    from langchain_community.embeddings import DashScopeEmbeddings

    embeddings = DashScopeEmbeddings(model="text-embedding-v4")
    logger.info("使用 DashScope 嵌入模型 (text-embedding-v4)")
except Exception as e:
    logger.warning(
        f"DashScope 嵌入不可用 ({e})，降级使用本地 sentence-transformers"
    )
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

# 文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    length_function=len,
)


def load_txt_files(directory: str) -> list:
    """加载目录下所有 txt 文件，文件名作为 source metadata"""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            documents.extend(docs)
    return documents


def init_vector_store():
    """初始化向量数据库（离线流程）"""
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"知识库目录不存在: {DATA_DIR}")

    documents = load_txt_files(DATA_DIR)
    if not documents:
        raise ValueError(f"目录中没有找到 txt 文件: {DATA_DIR}")

    logger.info(f"加载 {len(documents)} 个文档，开始分割...")
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"分割完成，共 {len(split_docs)} 个片段")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    vector_store = Chroma.from_documents(
        client=client,
        documents=split_docs,
        embedding=embeddings,
    )
    logger.info(f"向量库已保存至 {CHROMA_DIR}")
    return vector_store, len(split_docs)


def get_vector_store():
    """获取已存在的向量数据库（在线流程）"""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"向量数据库不存在，请先运行初始化：\n"
            f'python -c "from app.agents.rag_tools import init_vector_store; init_vector_store()"'
        )

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return Chroma(
        client=client,
        embedding_function=embeddings,
    )


def retrieve(query: str, k: int = 3) -> str:
    """检索最相关的文档片段

    Args:
        query: 用户提问
        k: 返回最相似的 k 个片段

    Returns:
        拼接的参考文本，格式：[来源:xxx.txt] chunk内容
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=k)

    parts = []
    for doc in results:
        source = doc.metadata.get("source", "unknown.txt")
        parts.append(f"[来源:{source}] {doc.page_content}")

    return "\n".join(parts)


if __name__ == "__main__":
    print("初始化向量数据库...")
    vector_store, doc_count = init_vector_store()
    print(f"完成！共处理 {doc_count} 个文档片段")
    print(f"向量数据库路径: {CHROMA_DIR}")
