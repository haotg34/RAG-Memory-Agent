from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from modules.memory.manager import MemoryManager
from modules.rag.retriever import hybrid_retrieval
from modules.memory.extractor import extract_user_preferences
from config import settings
import uuid
from database.models import LLMUsageLog


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

    memory_manager = MemoryManager(user_id, db)

    llm = get_llm(llm_provider=llm_provider, llm_model=llm_model)
    try:
        rewritten_query = rewrite_query(message, llm)
    except Exception as e:
        raise RuntimeError(f"rewrite_query失败: {e}") from e

    try:
        long_term_memories = memory_manager.retrieve_long_term_memory(rewritten_query)
    except Exception as e:
        raise RuntimeError(f"retrieve_long_term_memory失败: {e}") from e

    try:
        rag_results = hybrid_retrieval(rewritten_query, db)
    except Exception as e:
        raise RuntimeError(f"hybrid_retrieval失败: {e}") from e

    try:
        recent_memory = memory_manager.get_recent_memory(session_id)
    except Exception as e:
        raise RuntimeError(f"get_recent_memory失败: {e}") from e

    user_preferences = memory_manager.get_user_preferences()

    system_prompt = build_system_prompt(long_term_memories, rag_results, user_preferences)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(recent_memory)
    messages.append({"role": "user", "content": message})

    try:
        response = llm.invoke(messages)
    except Exception as e:
        raise RuntimeError(f"llm.invoke失败: {e}") from e

    memory_manager.add_message("user", message, session_id)
    memory_manager.add_message("assistant", response.content, session_id)

    provider_used = (llm_provider or settings.llm_provider or "openai").lower()
    model_used = llm_model or settings.llm_model
    token_usage = _extract_token_usage(response)
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

    return {
        "session_id": session_id,
        "response": response.content,
        "retrieved_memories": long_term_memories,
        "retrieved_knowledge": rag_results,
        "token_usage": token_usage,
    }
