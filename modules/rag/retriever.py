from sqlalchemy.orm import Session
from database.models import DocumentChunk
import numpy as np
from config import settings
from typing import List, Tuple
from modules.embedding_client import embed_text

_emb_api_key, _emb_base_url, _emb_model = settings.resolve_embedding()


def _cosine_similarity(a: list, b: list) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)) + 1e-12
    return float(np.dot(a_arr, b_arr) / denom)


def semantic_retrieval(query: str, db: Session, top_k: int = 20) -> List[Tuple[str, float]]:
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


def keyword_retrieval(query: str, db: Session, top_k: int = 20) -> List[Tuple[str, float]]:
    keywords = query.lower().split()
    all_chunks = db.query(DocumentChunk).all()

    if not all_chunks:
        return []

    results = []
    for chunk in all_chunks:
        content = chunk.content.lower()
        score = 0
        for keyword in keywords:
            if keyword in content:
                score += 1
        if score > 0:
            results.append((chunk.content, float(score)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def rrf_fusion(
    semantic_results: List[Tuple[str, float]],
    keyword_results: List[Tuple[str, float]],
    k: int = 60,
) -> List[str]:
    semantic_ranks = {content: i + 1 for i, (content, _) in enumerate(semantic_results)}
    keyword_ranks = {content: i + 1 for i, (content, _) in enumerate(keyword_results)}

    all_contents = set(semantic_ranks.keys()).union(set(keyword_ranks.keys()))

    rrf_scores = {}
    for content in all_contents:
        score = 0.0
        if content in semantic_ranks:
            score += 1.0 / (k + semantic_ranks[content])
        if content in keyword_ranks:
            score += 1.0 / (k + keyword_ranks[content])
        rrf_scores[content] = score

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


def hybrid_retrieval(query: str, db: Session, top_k: int | None = None) -> List[str]:
    if top_k is None:
        top_k = settings.rag_top_k

    semantic_results = semantic_retrieval(query, db)
    keyword_results = keyword_retrieval(query, db)

    fused_results = rrf_fusion(semantic_results, keyword_results)
    reranked_results = bge_rerank(query, fused_results, top_k)

    return reranked_results
