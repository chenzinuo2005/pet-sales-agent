import os
import re

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.common.logger import logger

# 路径配置
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../../resources/chroma_db")

# 嵌入模型 (DashScope 优先, sentence-transformers 备选)
try:
    from langchain_community.embeddings import DashScopeEmbeddings

    embeddings = DashScopeEmbeddings(model="text-embedding-v4")
    logger.info("embeddings_provider", extra={"provider": "dashscope", "model": "text-embedding-v4"})
except Exception as e:
    logger.warning("embeddings_fallback", extra={"error": str(e), "fallback": "huggingface"})
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")  # type: ignore[assignment]

# 文本分割器 — for non-encyclopedia files (after_sales, care_guide, etc.)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    length_function=len,
)

# Breed-centric splitter — each ## header starts a new chunk (keeps breed intact)
BREED_PATTERN = re.compile(r"(?=## \S)")


def _parse_breed_metadata(header_line: str) -> dict:
    """Extract breed name and type from header like '## 波斯猫 | 类型: 猫'"""
    meta = {"breed": "", "type": ""}
    # Remove ## prefix
    content = header_line.lstrip("#").strip()
    if "|" in content:
        parts = content.split("|")
        meta["breed"] = parts[0].strip()
        type_part = parts[1].strip()
        if "猫" in type_part:
            meta["type"] = "cat"
        elif "犬" in type_part:
            meta["type"] = "dog"
    return meta


def load_txt_files(directory: str) -> list:
    """加载目录下所有 txt 文件，文件名作为 source metadata"""
    documents = []
    for filename in sorted(os.listdir(directory)):  # sorted for deterministic ordering
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            documents.extend(docs)
    return documents


def _split_breed_encyclopedia(docs: list[Document]) -> list[Document]:
    """Split breeds_encyclopedia.txt by ## headers — each breed = one retrieval unit."""
    result = []
    for doc in docs:
        text = doc.page_content
        # Split on ## headers (keep the delimiter with the following text)
        parts = BREED_PATTERN.split(text)
        for part in parts:
            part = part.strip()
            if not part or part.startswith("# "):  # skip file-level header lines
                continue
            # Extract first line as breed header
            first_line = part.split("\n")[0]
            meta = _parse_breed_metadata(first_line)
            if not meta["breed"]:
                continue
            result.append(Document(
                page_content=part,
                metadata={
                    "source": doc.metadata.get("source", ""),
                    "breed": meta["breed"],
                    "type": meta["type"],
                },
            ))
    return result


def init_vector_store():
    """初始化向量数据库（离线流程）

    Uses breed-aware chunking: each breed stays as a complete retrieval unit
    with breed name and type (dog/cat) in metadata.
    """
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"知识库目录不存在: {DATA_DIR}")

    documents = load_txt_files(DATA_DIR)
    if not documents:
        raise ValueError(f"目录中没有找到 txt 文件: {DATA_DIR}")

    logger.info("documents_loaded", extra={"count": len(documents)})

    # Separate breed encyclopedia (breed-aware splitting) from other files
    encyclopedia_docs = [d for d in documents if d.metadata["source"] == "breeds_encyclopedia.txt"]
    other_docs = [d for d in documents if d.metadata["source"] != "breeds_encyclopedia.txt"]

    # Breed encyclopedia: split by breed header
    breed_chunks = _split_breed_encyclopedia(encyclopedia_docs)
    logger.info("breed_chunks", extra={"count": len(breed_chunks)})

    # Other files: use standard recursive splitter
    other_chunks = text_splitter.split_documents(other_docs) if other_docs else []
    logger.info("other_chunks", extra={"count": len(other_chunks)})

    all_chunks = breed_chunks + other_chunks
    logger.info("total_chunks", extra={"count": len(all_chunks)})

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    vector_store = Chroma.from_documents(
        client=client,
        documents=all_chunks,
        embedding=embeddings,
    )
    logger.info("vector_store_saved", extra={"path": CHROMA_DIR})
    return vector_store, len(all_chunks)


def get_vector_store():
    """获取已存在的向量数据库（在线流程）"""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            "向量数据库不存在，请先运行初始化：\n"
            'python -c "from app.agents.rag_tools import init_vector_store; init_vector_store()"'
        )

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return Chroma(
        client=client,
        embedding_function=embeddings,
    )


def retrieve_with_store(vector_store, query: str, k: int = 3) -> str:
    """检索最相关的文档片段 (接受外部 vector_store 参数，用于 DI)"""
    results = vector_store.similarity_search(query, k=k)
    parts = []
    for doc in results:
        source = doc.metadata.get("source", "unknown.txt")
        parts.append(f"[来源:{source}] {doc.page_content}")
    return "\n".join(parts)


def retrieve(query: str, k: int = 3) -> str:
    """检索最相关的文档片段

    Args:
        query: 用户提问
        k: 返回最相似的 k 个片段

    Returns:
        拼接的参考文本，格式：[来源:xxx.txt] chunk内容
    """
    vector_store = get_vector_store()
    return retrieve_with_store(vector_store, query, k)


if __name__ == "__main__":
    print("初始化向量数据库...")
    vector_store, doc_count = init_vector_store()
    print(f"完成！共处理 {doc_count} 个文档片段")
    print(f"向量数据库路径: {CHROMA_DIR}")
