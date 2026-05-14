from langchain_openai import ChatOpenAI
import json
from config import settings

_llm_api_key, _llm_base_url, _llm_model = settings.resolve_llm()
llm = ChatOpenAI(model=_llm_model, temperature=0, api_key=_llm_api_key, base_url=_llm_base_url)


def extract_long_term_memory(conversation: list) -> list:
    prompt = f"""
请仔细分析以下对话，提取用户的关键信息、偏好、习惯和重要事实。
要求：
1. 每条记忆独立成一行
2. 语言简洁准确，不超过20个字
3. 只提取确定的信息，不要猜测
4. 忽略无关的闲聊内容

对话内容：
{json.dumps(conversation, ensure_ascii=False)}
"""

    response = llm.invoke(prompt)
    memories = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
    return memories


def extract_user_preferences(conversation: list) -> dict:
    prompt = f"""
请从以下对话中提取用户的偏好设置，返回严格的JSON格式。
可能的偏好包括：回答风格(简洁/详细/幽默)、语言偏好、是否喜欢使用emoji等。
如果没有提取到任何偏好，返回空对象{{}}。

对话内容：
{json.dumps(conversation, ensure_ascii=False)}
"""

    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except Exception:
        return {}
