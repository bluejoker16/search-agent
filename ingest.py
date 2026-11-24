"""
数据层：文档加载 -> 切块 -> 向量化 -> 混合检索器
赛博算命项目：两个知识库（心理学 / 易经）各建一个检索器。
每个检索器 = BM25（字面精确）+ 向量（语义泛化）的 RRF 融合。
面试时按 build_retrievers() 的顺序讲即可。
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
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
    """按 EMBEDDING_MODE 选向量：local 本地哈希 / bge 硅基流动语义向量。"""
    if os.getenv("EMBEDDING_MODE", "bge").strip().lower() == "local":
        print("[ingest] Embedding: 本地哈希（关键词模式，短条目命中更稳）")
        return LocalHashEmbeddings()

    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1").strip()
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


def _tokenize(text: str) -> list[str]:
    """中文按字符 bigram 切（免分词依赖），英文/数字按整词。"""
    tokens: list[str] = []
    for seg in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text.lower()):
        if "\u4e00" <= seg[0] <= "\u9fff":
            tokens += [seg[i : i + 2] for i in range(len(seg) - 1)] + ([seg] if len(seg) == 1 else [])
        else:
            tokens.append(seg)
    return tokens


class BM25Retriever:
    """
    BM25 字面检索（Okapi BM25 公式，零新依赖）。

    为什么比纯向量好：向量靠 bigram 重合，"焦虑/压力"这类模板词每条都有，
    会把弱相关条目一起拉高；BM25 的 IDF 会压低高频词的权重，不相关的直接 0 分。
    中文用 bigram 当 token，省掉 jieba 分词依赖。
    """

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        texts = [d.page_content for d in docs]
        self.tfs = [Counter(_tokenize(t)) for t in texts]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / len(self.lens) if self.lens else 1.0
        self.df: Counter = Counter()
        for tf in self.tfs:
            self.df.update(tf.keys())
        self.n = len(docs)

    def scores(self, query: str) -> np.ndarray:
        q_tokens = _tokenize(query)
        out = np.zeros(self.n)
        for i, tf in enumerate(self.tfs):
            for token in q_tokens:
                freq = tf.get(token, 0)
                if not freq:
                    continue
                idf = math.log(1 + (self.n - self.df[token] + 0.5) / (self.df[token] + 0.5))
                norm = freq + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                out[i] += idf * freq * (self.k1 + 1) / norm
        return out


class HybridRetriever:
    """
    混合检索 = BM25（字面精确）+ 向量（语义泛化），用 RRF 融合排序。

    RRF（Reciprocal Rank Fusion）：score(d) = Σ 1/(c + rank_i(d))，c 常取 60。
    它只用排名、不用分数，天然规避 BM25 与 cosine 分数尺度不一致的问题，
    且两路互补：BM25 抓精确关键词，向量抓语义泛化（如"感情不顺"->"亲密关系焦虑"）。

    无命中兜底：BM25 全 0 且向量分数低于阈值时返回空列表，
    交给上层工具让模型基于通用知识作答，而不是硬塞弱相关片段制造幻觉。
    """

    def __init__(self, docs: list[Document], embeddings: Embeddings, k: int):
        self.docs = docs
        self.k = k
        self.bm25 = BM25Retriever(docs)
        self.vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
        self.rrf_c = int(os.getenv("RRF_C", "60"))
        self.min_vec_score = float(os.getenv("MIN_VEC_SCORE", "0.18"))

    def retrieve_with_diagnostics(self, query: str) -> tuple[list[Document], bool]:
        """返回 (文档列表, 是否命中)。工具层据此决定要不要给模型兜底提示。"""
        bm25_scores = self.bm25.scores(query)
        vec_pairs = self.vectorstore.similarity_search_with_score(query, k=self.k)

        # 无命中判定：BM25 全 0 + 最高向量分也低于阈值 -> 认为知识库没有相关内容
        top_bm25 = float(bm25_scores.max()) if len(bm25_scores) else 0.0
        top_vec = float(vec_pairs[0][1]) if vec_pairs else 0.0
        if top_bm25 <= 0 and top_vec < self.min_vec_score:
            return [], False

        # RRF 融合：按各自排名累加倒数分
        rrf: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        bm25_order = np.argsort(-bm25_scores)[: self.k]
        for rank, idx in enumerate(bm25_order, 1):
            doc = self.docs[int(idx)]
            doc_map[doc.page_content] = doc
            rrf[doc.page_content] = rrf.get(doc.page_content, 0.0) + 1.0 / (self.rrf_c + rank)
        for rank, (doc, _score) in enumerate(vec_pairs, 1):
            doc_map[doc.page_content] = doc
            rrf[doc.page_content] = rrf.get(doc.page_content, 0.0) + 1.0 / (self.rrf_c + rank)

        ranked = sorted(rrf, key=rrf.get, reverse=True)[: self.k]
        return [doc_map[key] for key in ranked], True

    def invoke(self, query: str) -> list[Document]:
        """兼容标准 retriever 接口（app.py 的 /api/retrieve 会用到）。"""
        docs, _ = self.retrieve_with_diagnostics(query)
        return docs


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """加载目录下所有 .md / .txt，按空行分段，每段记录来源文件名。"""
    docs: list[Document] = []
    if not docs_dir.exists():
        return docs
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        content = path.read_text(encoding="utf-8")
        # 跳过一级标题（如 "# 心理学 · 情绪与睡眠科普"），它只是文件说明，不是知识条目
        content = "\n".join(line for line in content.splitlines() if not line.startswith("# "))
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
        category = path.stem
        for i, para in enumerate(merged):
            docs.append(
                Document(
                    page_content=para.strip(),
                    metadata={"source": path.name, "category": category, "paragraph": i},
                )
            )
    return docs


def build_retrievers() -> dict:
    """按类别分组建混合检索器，返回 {类别名: HybridRetriever}。
    每库独立 Top-K：心理学少而精（默认 2），易经多留候选（默认 3）。"""
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

    by_category: dict[str, list] = {}
    for chunk in chunks:
        by_category.setdefault(chunk.metadata.get("category", "未分类"), []).append(chunk)

    embeddings = get_embeddings()
    use_hybrid = os.getenv("ENABLE_BM25", "1").strip() != "0"
    retrievers: dict = {}
    for category, cat_chunks in by_category.items():
        # 每库独立 Top-K：默认 心理学 2 / 易经 3，可用环境变量覆盖
        default_k = 2 if category == "心理学" else 3
        k = int(os.getenv(f"TOP_K_{category}", os.getenv("TOP_K", str(default_k))))
        retriever = HybridRetriever(cat_chunks, embeddings, k) if use_hybrid else _VectorOnlyRetriever(cat_chunks, embeddings, k)
        retrievers[category] = retriever
        mode = "BM25+向量 RRF 融合" if use_hybrid else "纯向量"
        print(f"[ingest] 知识库「{category}」: {len(cat_chunks)} 块, Top-{k}, {mode}")
    return retrievers


class _VectorOnlyRetriever:
    """ENABLE_BM25=0 时的对照实现：纯向量，保留 diagnose 接口形状。"""

    def __init__(self, docs, embeddings, k):
        self.vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
        self.k = k

    def retrieve_with_diagnostics(self, query: str) -> tuple[list[Document], bool]:
        pairs = self.vectorstore.similarity_search_with_score(query, k=self.k)
        return [d for d, _ in pairs], bool(pairs)

    def invoke(self, query: str) -> list[Document]:
        docs, _ = self.retrieve_with_diagnostics(query)
        return docs


if __name__ == "__main__":
    retrievers = build_retrievers()
    for question in ["考试前特别紧张手心出汗", "我最近失眠，心里很焦虑，压力大", "我最近总梦见牙齿掉了"]:
        print(f"\n>>> 测试问题: {question}")
        for category, retriever in retrievers.items():
            docs, hit = retriever.retrieve_with_diagnostics(question)
            flag = "命中" if hit else "未命中(兜底)"
            print(f"  【{category}】{flag}:")
            for doc in docs:
                print(f"    - {doc.page_content.splitlines()[0][:40]}")
