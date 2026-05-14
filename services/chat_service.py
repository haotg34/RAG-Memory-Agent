from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from modules.memory.manager import MemoryManager
from modules.rag.retriever import hybrid_retrieval
from modules.memory.extractor import extract_user_preferences
from config import settings
import uuid
from database.models import LLMUsageLog, LLMRouteLog
from services.router_service import smart_route, check_answer_yesno


def _extract_token_usage(response) -> dict:
    usage = {}
    um = getattr(response, "usage_metadata", None)
    if isinstance(um, dict):
        usage["prompt_tokens"] = int(um.get("input_tokens") or 0)
        usage["completion_tokens"] = int(um.get("output_tokens") or 0)
        usage["total_tokens"] = int(um.get("total_tokens") or 0)
        return usage

    rm = getattr(response, "response_metadata", None)
    if isinstance(rm, dict):
        tu = rm.get("token_usage") or rm.get("usage") or {}
        if isinstance(tu, dict):
            usage["prompt_tokens"] = int(tu.get("prompt_tokens") or 0)
            usage["completion_tokens"] = int(tu.get("completion_tokens") or 0)
            usage["total_tokens"] = int(tu.get("total_tokens") or 0)
            return usage

    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def get_llm(
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> ChatOpenAI:
    provider = llm_provider.lower() if llm_provider else None
    api_key, base_url, model = settings.resolve_llm_for(provider)
    if llm_model:
        model = llm_model
    return ChatOpenAI(
        model=model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=api_key,
        base_url=base_url,
    )


def rewrite_query(query: str, llm: ChatOpenAI) -> str:
    prompt = f"""
请将以下用户查询改写为更适合语义检索的形式，保持原意不变。
用户查询：{query}
改写后的查询：
"""
    return llm.invoke(prompt).content.strip()


def build_system_prompt(long_term_memories: list, rag_results: list, user_preferences: dict) -> str:
    system_prompt = "你是一个智能AI助手，能够基于知识库和用户的历史对话提供准确、有用的回答。\n\n"

    if user_preferences:
        system_prompt += "用户偏好：\n"
        for key, value in user_preferences.items():
            system_prompt += f"- {key}: {value}\n"
        system_prompt += "\n"

    if long_term_memories:
        system_prompt += "用户的长期记忆：\n"
        for memory in long_term_memories:
            system_prompt += f"- {memory}\n"
        system_prompt += "\n"

    if rag_results:
        system_prompt += "参考知识库内容：\n"
        for i, result in enumerate(rag_results, 1):
            system_prompt += f"{i}. {result}\n"
        system_prompt += "\n"

    system_prompt += "请根据以上信息回答用户的问题，如果信息不足，请直接说明。"
    return system_prompt


def _resolve_model_for_tier(tier: str, llm_provider: str | None) -> tuple[str | None, str | None]:
    t = (tier or "").lower()
    if t == "small":
        return llm_provider, (settings.llm_small_model or settings.llm_model)
    if t == "medium":
        return llm_provider, (settings.llm_medium_model or settings.llm_model)
    if t == "large":
        return llm_provider, (settings.llm_large_model or settings.llm_model)
    return llm_provider, settings.llm_model


def _run_once(
    user_id: str,
    message: str,
    session_id: str,
    llm_provider: str | None,
    llm_model: str | None,
    db: Session | None,
) -> dict:
    memory_manager = MemoryManager(user_id, db)
    llm = get_llm(llm_provider=llm_provider, llm_model=llm_model)

    rewritten_query = rewrite_query(message, llm)
    long_term_memories = memory_manager.retrieve_long_term_memory(rewritten_query)
    rag_results = hybrid_retrieval(rewritten_query, db)
    recent_memory = memory_manager.get_recent_memory(session_id)
    user_preferences = memory_manager.get_user_preferences()

    system_prompt = build_system_prompt(long_term_memories, rag_results, user_preferences)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(recent_memory)
    messages.append({"role": "user", "content": message})

    response = llm.invoke(messages)
    token_usage = _extract_token_usage(response)

    provider_used = (llm_provider or settings.llm_provider or "openai").lower()
    model_used = llm_model or settings.llm_model
    return {
        "memory_manager": memory_manager,
        "recent_memory": recent_memory,
        "response": response,
        "token_usage": token_usage,
        "provider_used": provider_used,
        "model_used": model_used,
        "long_term_memories": long_term_memories,
        "rag_results": rag_results,
    }


def _persist_final(
    *,
    user_id: str,
    session_id: str,
    message: str,
    reply: str,
    memory_manager: MemoryManager,
    recent_memory: list,
    provider_used: str,
    model_used: str,
    token_usage: dict,
    db: Session | None,
):
    memory_manager.add_message("user", message, session_id)
    memory_manager.add_message("assistant", reply, session_id)

    try:
        db.add(
            LLMUsageLog(
                user_id=user_id,
                session_id=session_id,
                provider=provider_used,
                model=model_used,
                prompt_tokens=token_usage.get("prompt_tokens", 0),
                completion_tokens=token_usage.get("completion_tokens", 0),
                total_tokens=token_usage.get("total_tokens", 0),
            )
        )
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass

    if len(recent_memory) % 10 == 0 and len(recent_memory) > 0:
        full_conversation = memory_manager.get_recent_memory(session_id, limit=10)
        memory_manager.save_long_term_memory(full_conversation)

        new_preferences = extract_user_preferences(full_conversation)
        if new_preferences:
            memory_manager.update_user_preferences(new_preferences)


def chat(
    user_id: str,
    message: str,
    session_id: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    db: Session | None = None,
) -> dict:
    if not session_id:
        session_id = str(uuid.uuid4())

    if not settings.routing_enabled or llm_model:
        provider_used = (llm_provider or settings.llm_provider or "openai").lower()
        model_used = llm_model or settings.llm_model
        try:
            run = _run_once(
                user_id=user_id,
                message=message,
                session_id=session_id,
                llm_provider=llm_provider,
                llm_model=llm_model,
                db=db,
            )
        except Exception as e:
            raise RuntimeError(f"chat失败: {e}") from e

        response = run["response"]
        token_usage = run["token_usage"]
        _persist_final(
            user_id=user_id,
            session_id=session_id,
            message=message,
            reply=response.content,
            memory_manager=run["memory_manager"],
            recent_memory=run["recent_memory"],
            provider_used=provider_used,
            model_used=model_used,
            token_usage=token_usage,
            db=db,
        )
        long_term_memories = run["long_term_memories"]
        rag_results = run["rag_results"]
        return {
            "session_id": session_id,
            "response": response.content,
            "retrieved_memories": long_term_memories,
            "retrieved_knowledge": rag_results,
            "token_usage": token_usage,
        }

    router_provider, router_model = _resolve_model_for_tier("small", llm_provider)
    llm_router = get_llm(llm_provider=router_provider, llm_model=router_model)
    decision = smart_route(
        query=message,
        llm_for_score=llm_router,
        simple_len=settings.routing_simple_len,
        th_small=settings.routing_score_threshold_small,
        th_medium=settings.routing_score_threshold_medium,
    )

    decided_tier = "large" if decision.force_large else decision.tier
    tiers = ["large"] if decided_tier == "large" else ["small", "medium", "large"]

    final_run = None
    checker_raw = None
    upgraded = 0
    degraded = 0
    attempts: list[dict] = []
    for idx, tier in enumerate(tiers):
        provider_t, model_t = _resolve_model_for_tier(tier, llm_provider)
        try:
            run = _run_once(
                user_id=user_id,
                message=message,
                session_id=session_id,
                llm_provider=provider_t,
                llm_model=model_t,
                db=db,
            )
            attempts.append(
                {
                    "tier": tier,
                    "provider": run["provider_used"],
                    "model": run["model_used"],
                    "token_usage": run["token_usage"],
                }
            )
        except Exception as e:
            degraded = 1
            attempts.append({"tier": tier, "error": type(e).__name__})
            if decided_tier == "large" and tier == "large":
                continue
            if tier == "small":
                continue
            raise RuntimeError(f"{tier}模型调用失败: {e}") from e

        response = run["response"]
        if tier == "large" or decision.force_large or idx == len(tiers) - 1:
            final_run = run
            break

        ok, checker_raw = check_answer_yesno(message, response.content, llm_router)
        if ok:
            final_run = run
            break
        upgraded = 1

    if final_run is None:
        raise RuntimeError("所有模型尝试均失败")

    response = final_run["response"]
    token_usage = final_run["token_usage"]
    provider_used = final_run["provider_used"]
    model_used = final_run["model_used"]

    try:
        db.add(
            LLMRouteLog(
                user_id=user_id,
                session_id=session_id,
                query=message,
                decided_tier=decided_tier,
                decided_score=(None if decision.score is None else str(decision.score)),
                rule_hit=decision.rule_hit,
                checker_raw=checker_raw,
                upgraded=upgraded,
                degraded=degraded,
                final_provider=provider_used,
                final_model=model_used,
                meta={"attempts": attempts},
            )
        )
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass

    _persist_final(
        user_id=user_id,
        session_id=session_id,
        message=message,
        reply=response.content,
        memory_manager=final_run["memory_manager"],
        recent_memory=final_run["recent_memory"],
        provider_used=provider_used,
        model_used=model_used,
        token_usage=token_usage,
        db=db,
    )

    long_term_memories = final_run["long_term_memories"]
    rag_results = final_run["rag_results"]

    return {
        "session_id": session_id,
        "response": response.content,
        "retrieved_memories": long_term_memories,
        "retrieved_knowledge": rag_results,
        "token_usage": token_usage,
    }
