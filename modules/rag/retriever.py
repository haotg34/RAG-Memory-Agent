from sqlalchemy.orm import Session
from database.models import DocumentChunk
import numpy as np
from config import settings
from typing import List, Tuple
from modules.embedding_client import embed_text
import re

_emb_api_key, _emb_base_url, _emb_model = settings.resolve_embedding()


def _cosine_similarity(a: list, b: list) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)) + 1e-12
    return float(np.dot(a_arr, b_arr) / denom)


def semantic_retrieval(query: str, db: Session, top_k: int | None = None) -> List[Tuple[str, float]]:
    if top_k is None:
        top_k = settings.rag_semantic_top_k
    query_embedding = embed_text(query, _emb_model, _emb_api_key, _emb_base_url)
    all_chunks = db.query(DocumentChunk).all()

    if not all_chunks:
        return []

    results = []
    for chunk in all_chunks:
        similarity = _cosine_similarity(query_embedding, chunk.embedding)
        results.append((chunk.content, similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def _extract_keywords(query: str) -> list[str]:
    q = (query or "").lower()
    words = [w for w in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", q) if len(w) >= 2]
    keywords = set(words)
    for word in words:
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            for n in (2, 3, 4):
                for i in range(0, max(len(word) - n + 1, 0)):
                    keywords.add(word[i : i + n])
    return sorted(keywords, key=len, reverse=True)[:80]


def keyword_retrieval(query: str, db: Session, top_k: int | None = None) -> List[Tuple[str, float]]:
    if top_k is None:
        top_k = settings.rag_keyword_top_k
    keywords = _extract_keywords(query)
    all_chunks = db.query(DocumentChunk).all()

    if not all_chunks or not keywords:
        return []

    results = []
    for chunk in all_chunks:
        content = chunk.content.lower()
        score = 0.0
        for keyword in keywords:
            if keyword in content:
                score += max(len(keyword), 2) / 2
        if score > 0:
            results.append((chunk.content, float(score)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def rrf_fusion(result_sets: list[List[Tuple[str, float]]], k: int = 60) -> List[str]:
    rrf_scores = {}
    for results in result_sets:
        for rank, (content, _) in enumerate(results, 1):
            rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (k + rank)

    sorted_contents = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [content for content, _ in sorted_contents]


def bge_rerank(query: str, documents: List[str], top_k: int | None = None) -> List[str]:
    if top_k is None:
        top_k = settings.rag_top_k

    if not documents:
        return []

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.resolve_llm()[0],
        base_url=settings.resolve_llm()[1],
    )

    prompt = f"""
请根据以下查询，对候选文档进行相关性排序，只返回最相关的{top_k}个文档的编号，用逗号分隔。

查询：{query}

候选文档：
"""

    for i, doc in enumerate(documents, 1):
        prompt += f"{i}. {doc[:200]}...\n\n"

    prompt += "\n最相关的文档编号："

    response = llm.invoke(prompt)

    try:
        ranks = [
            int(x.strip())
            for x in response.content.strip().split(",")
            if x.strip().isdigit()
        ]
        return [documents[i - 1] for i in ranks if 1 <= i <= len(documents)][:top_k]
    except Exception:
        return documents[:top_k]


def _unique_queries(query: str, extra_queries: list[str] | None = None) -> list[str]:
    seen = set()
    queries = []
    for q in [query, *(extra_queries or [])]:
        text = (q or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        queries.append(text)
    return queries[:4]


def hybrid_retrieval(
    query: str,
    db: Session,
    top_k: int | None = None,
    extra_queries: list[str] | None = None,
) -> List[str]:
    if top_k is None:
        top_k = settings.rag_top_k

    result_sets = []
    queries = _unique_queries(query, extra_queries)
    for q in queries:
        result_sets.append(semantic_retrieval(q, db))
        result_sets.append(keyword_retrieval(q, db))

    fused_results = rrf_fusion(result_sets)[: settings.rag_candidate_top_k]
    reranked_results = bge_rerank(query, fused_results, top_k)

    return reranked_results
