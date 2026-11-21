"""
数据层：文档加载 -> 切块 -> 向量化 -> 检索器
RAG 的前半段全在这里，面试时按 build_retriever() 的顺序讲即可。
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "docs"
DIM = 512
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class LocalHashEmbeddings(Embeddings):
    """
    零依赖的本地降级 Embedding：中文字符 bigram + 英文单词哈希 + L2 归一化。

    它不是语义向量，但能让关键词重合的检索稳定命中；
    没有外部 Embedding Key 时，整条 RAG 链路依然可以完整演示。
    用 md5 而不是内置 hash()：内置 hash 每进程加盐，重启后向量对不上。
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def _vectorize(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for seg in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text.lower()):
            if "\u4e00" <= seg[0] <= "\u9fff":
                tokens = [seg[i : i + 2] for i in range(len(seg) - 1)] + [seg]
            else:
                tokens = [seg]
            for token in tokens:
                idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


class QueryPrefixedEmbeddings(Embeddings):
    """BGE 系列查询侧要加指令前缀、文档侧不加，这是训练方式决定的。"""

    def __init__(self, base: Embeddings, prefix: str = BGE_QUERY_PREFIX):
        self.base = base
        self.prefix = prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self.prefix + text)


def get_embeddings() -> Embeddings:
    """配了 Key 就用 BGE 语义向量，否则用本地哈希兜底。Key 留空时自动复用 LLM Key。"""
    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1").strip()
    # 硅基流动一个 Key 打通问答和向量化：EMBEDDING_API_KEY 留空则复用 LLM_API_KEY
    if not api_key and "siliconflow" in os.getenv("LLM_BASE_URL", ""):
        api_key = os.getenv("LLM_API_KEY", "").strip()
    if api_key and not api_key.startswith("sk-在这里"):
        from langchain_openai import OpenAIEmbeddings

        base = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            api_key=api_key,
            base_url=base_url,
        )
        print("[ingest] Embedding: BAAI/bge-m3（硅基流动语义检索）")
        return QueryPrefixedEmbeddings(base)
    print("[ingest] Embedding: 本地哈希（降级模式，零成本跑通链路）")
    return LocalHashEmbeddings()


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """加载目录下所有 .md / .txt，按空行分段，每段记录来源文件名。"""
    docs: list[Document] = []
    if not docs_dir.exists():
        return docs
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        content = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        # Markdown 标题常与正文之间有空行，单独成段会污染检索，合并到下一段
        merged: list[str] = []
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i]
            if para.startswith("#") and i + 1 < len(paragraphs):
                para = para + "\n" + paragraphs[i + 1]
                i += 1
            merged.append(para)
            i += 1
        for i, para in enumerate(merged):
            docs.append(
                Document(
                    page_content=para.strip(),
                    metadata={"source": path.name, "paragraph": i},
                )
            )
    return docs


def build_retriever():
    """完整数据链路：加载 -> 切块 -> 向量化 -> 存库 -> 返回检索器。"""
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"知识库为空，请在 {DOCS_DIR} 下放置 .md 或 .txt 文档")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "300")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] 文档 {len(docs)} 段 -> 切块 {len(chunks)} 个")

    vectorstore = InMemoryVectorStore.from_documents(chunks, get_embeddings())
    return vectorstore.as_retriever(search_kwargs={"k": int(os.getenv("TOP_K", "3"))})


if __name__ == "__main__":
    retriever = build_retriever()
    for question in ["标准版价格是多少？", "支持哪些部署方式？", "如何申请售后服务？"]:
        print(f"\n>>> {question}")
        for doc in retriever.invoke(question):
            print(f"  [{doc.metadata['source']}] {doc.page_content[:100]}...")
