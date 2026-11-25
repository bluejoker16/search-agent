"""
数据层：文档加载 -> 切块 -> 向量化 -> 混合检索器（Chroma 持久化）
赛博算命项目：两个知识库（心理学 / 易经）各建一个检索器。
每个检索器 = BM25（字面精确）+ 向量（语义泛化）的 RRF 融合。
向量存在本地 Chroma（sqlite），重启不重新 embed；文档变更自动重建。
面试时按 build_retrievers() 的顺序讲即可。
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "docs"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "data" / "chroma_db")))
DIM = 512
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class LocalHashEmbeddings(Embeddings):
    """
    零依赖的本地降级 Embedding：中文字符 bigram + 英文单词哈希 + L2 归一化。
    用 md5 而不是内置 hash()：内置 hash 每进程加盐，重启后向量对不上（持久化时尤其致命）。
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
    """按 EMBEDDING_MODE 选向量：local 本地哈希 / bge 硅基流动语义向量。
    切换 mode 会改变文档签名 -> 自动重建对应 collection，不会混用两种向量。"""
    if os.getenv("EMBEDDING_MODE", "bge").strip().lower() == "local":
        print("[ingest] Embedding: 本地哈希（关键词模式）")
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
    print("[ingest] Embedding: 本地哈希（降级模式）")
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
    """BM25 字面检索（Okapi BM25，零新依赖）。IDF 压低每条都有的模板词，不相关直接 0 分。"""

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


def _distance_to_similarity(distance: float) -> float:
    """Chroma 默认平方 L2 距离（越小越近）；归一化向量下 dist = 2 - 2cos，故 cos = 1 - dist/2。
    统一转成 0~1 的相似度，让 MIN_VEC_SCORE 阈值语义与内存库保持一致。"""
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


class HybridRetriever:
    """
    混合检索 = BM25（字面精确）+ Chroma 向量（语义泛化），RRF 融合排序。

    RRF：score(d) = Σ 1/(c + rank_i(d))，c 常取 60。只用排名，规避两路分数尺度不一致。
    无命中兜底：BM25 全 0 且向量相似度 < min_vec_score 时返回空，交给上层让模型用通用知识作答。
    """

    def __init__(self, docs: list[Document], vectorstore, k: int, use_hybrid: bool = True):
        self.docs = docs
        self.k = k
        self.use_hybrid = use_hybrid
        self.vectorstore = vectorstore
        self.bm25 = BM25Retriever(docs) if use_hybrid else None
        self.rrf_c = int(os.getenv("RRF_C", "60"))
        self.min_vec_score = float(os.getenv("MIN_VEC_SCORE", "0.18"))

    def retrieve_with_diagnostics(self, query: str) -> tuple[list[Document], bool]:
        """返回 (文档列表, 是否命中)。工具层据此决定要不要给模型兜底提示。"""
        # Chroma 返回 (doc, 距离)，转成相似度并按相似度降序
        raw_pairs = self.vectorstore.similarity_search_with_score(query, k=self.k)
        vec_pairs = sorted(
            [(doc, _distance_to_similarity(dist)) for doc, dist in raw_pairs],
            key=lambda x: -x[1],
        )

        bm25_scores = self.bm25.scores(query) if self.bm25 else np.zeros(len(self.docs))
        top_bm25 = float(bm25_scores.max()) if len(bm25_scores) else 0.0
        top_vec = vec_pairs[0][1] if vec_pairs else 0.0

        # 无命中：BM25 全 0 且向量相似度也低于阈值 -> 知识库没有相关内容
        if top_bm25 <= 0 and top_vec < self.min_vec_score:
            return [], False

        if not self.use_hybrid:
            return [doc for doc, _ in vec_pairs], True

        # RRF 融合：按各自排名累加倒数分
        rrf: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        for rank, idx in enumerate(np.argsort(-bm25_scores)[: self.k], 1):
            doc = self.docs[int(idx)]
            doc_map[doc.page_content] = doc
            rrf[doc.page_content] = rrf.get(doc.page_content, 0.0) + 1.0 / (self.rrf_c + rank)
        for rank, (doc, _sim) in enumerate(vec_pairs, 1):
            doc_map[doc.page_content] = doc
            rrf[doc.page_content] = rrf.get(doc.page_content, 0.0) + 1.0 / (self.rrf_c + rank)

        ranked = sorted(rrf, key=rrf.get, reverse=True)[: self.k]
        return [doc_map[key] for key in ranked], True

    def invoke(self, query: str) -> list[Document]:
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
        content = "\n".join(line for line in content.splitlines() if not line.startswith("# "))
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
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


def _doc_signature(chunks: list[Document], embedding_sig: str) -> str:
    """文档内容签名：任一 md 变了、或 embedding 模式变了，签名就变，触发 collection 重建。"""
    h = hashlib.md5()
    h.update(embedding_sig.encode("utf-8"))
    for chunk in chunks:
        h.update(chunk.page_content.encode("utf-8"))
    return h.hexdigest()


def _get_chroma_store(chunks: list[Document], embeddings: Embeddings, category: str, signature: str, force: bool):
    """
    拿到该类别的 Chroma 向量库：签名一致则直接加载（不重新 embed），否则重建。
    collection 名只允许 [a-zA-Z0-9._-]，故用中文类别名的 md5 做稳定 ID。
    """
    from langchain_chroma import Chroma

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    collection_name = "kb_" + hashlib.md5(category.encode("utf-8")).hexdigest()[:16]

    # 先看 collection 是否已存在且签名一致
    need_rebuild = force
    store = None
    try:
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        meta = store._collection.metadata or {}
        count = store._collection.count()
        if meta.get("signature") != signature or count != len(chunks):
            need_rebuild = True
    except Exception:
        need_rebuild = True

    if need_rebuild:
        # 先删旧 collection（签名变了说明文档/向量规则变了）
        try:
            if store is not None:
                store.delete_collection()
        except Exception:
            pass
        store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(CHROMA_DIR),
            collection_metadata={"signature": signature, "category": category},
        )
        action = "重建"
    else:
        action = "复用"
    return store, action


def build_retrievers(force_rebuild: bool = False) -> dict:
    """按类别分组建混合检索器，返回 {类别名: HybridRetriever}。"""
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
    embedding_sig = os.getenv("EMBEDDING_MODE", "bge") + ":" + os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    use_hybrid = os.getenv("ENABLE_BM25", "1").strip() != "0"

    retrievers: dict = {}
    for category, cat_chunks in by_category.items():
        # 每库独立 Top-K：默认心理学 2 / 易经 3
        default_k = 2 if category == "心理学" else 3
        k = int(os.getenv(f"TOP_K_{category}", os.getenv("TOP_K", str(default_k))))
        signature = _doc_signature(cat_chunks, embedding_sig)
        store, action = _get_chroma_store(cat_chunks, embeddings, category, signature, force_rebuild)
        retrievers[category] = HybridRetriever(cat_chunks, store, k, use_hybrid)
        mode = "BM25+Chroma RRF" if use_hybrid else "纯 Chroma 向量"
        print(f"[ingest] 知识库「{category}」: {len(cat_chunks)} 块, Top-{k}, {mode}, collection {action}")
    return retrievers


if __name__ == "__main__":
    force = "--rebuild" in sys.argv
    if force:
        print("[ingest] 强制重建所有 collection")
    retrievers = build_retrievers(force_rebuild=force)
    for question in ["考试前特别紧张手心出汗", "我最近失眠，心里很焦虑，压力大", "我最近总梦见牙齿掉了"]:
        print(f"\n>>> 测试问题: {question}")
        for category, retriever in retrievers.items():
            docs, hit = retriever.retrieve_with_diagnostics(question)
            flag = "命中" if hit else "未命中(兜底)"
            print(f"  【{category}】{flag}:")
            for doc in docs:
                print(f"    - {doc.page_content.splitlines()[0][:40]}")
