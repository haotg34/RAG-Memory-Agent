import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    tier: str
    score: float | None
    rule_hit: str | None
    force_large: bool


_SIMPLE_WORDS = ("你好", "在吗", "谢谢", "再见", "介绍一下", "hi", "hello")
_FORCE_LARGE_WORDS = (
    "密码",
    "验证码",
    "密钥",
    "token",
    "apikey",
    "api key",
    "身份证",
    "银行卡",
    "工资",
    "发票",
    "合同金额",
    "财务",
)


def rule_route(query: str, simple_len: int) -> tuple[str | None, str | None, bool]:
    q = (query or "").strip()
    low = q.lower()
    if any(w in q for w in _SIMPLE_WORDS) or len(q) < simple_len:
        return "small", "simple_rule", False
    if any(w in q for w in _FORCE_LARGE_WORDS) or any(w in low for w in _FORCE_LARGE_WORDS):
        return "large", "force_large_rule", True
    return None, None, False


def _parse_score(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"([01](?:\.\d+)?)", text.strip())
    if not m:
        return None
    try:
        v = float(m.group(1))
    except Exception:
        return None
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return v


def score_difficulty(query: str, llm) -> float | None:
    prompt = f"""
评估用户问题难度，仅输出0~1的小数，数字越大难度越高。
闲聊=0.1，普通文档查询=0.4，复杂推理=0.8及以上。
用户问题：{query}
仅输出数字：
""".strip()
    try:
        r = llm.invoke(prompt)
        return _parse_score(getattr(r, "content", "") or "")
    except Exception:
        return None


def smart_route(
    query: str,
    llm_for_score,
    simple_len: int,
    th_small: float,
    th_medium: float,
) -> RouteDecision:
    tier, rule_hit, force_large = rule_route(query, simple_len=simple_len)
    if tier is not None:
        return RouteDecision(tier=tier, score=None, rule_hit=rule_hit, force_large=force_large)

    score = score_difficulty(query, llm_for_score)
    if score is None:
        return RouteDecision(tier="medium", score=None, rule_hit="score_parse_failed", force_large=False)

    if score < th_small:
        return RouteDecision(tier="small", score=score, rule_hit=None, force_large=False)
    if score < th_medium:
        return RouteDecision(tier="medium", score=score, rule_hit=None, force_large=False)
    return RouteDecision(tier="large", score=score, rule_hit=None, force_large=False)


def check_answer_yesno(user_query: str, answer: str, llm_checker) -> tuple[bool, str]:
    prompt = f"""
用户问题：{user_query}
现有回答：{answer}
判断回答是否准确完整、无幻觉、满足用户需求，只输出 YES / NO
""".strip()
    try:
        r = llm_checker.invoke(prompt)
        raw = (getattr(r, "content", "") or "").strip().upper()
        if "YES" in raw:
            return True, raw
        if "NO" in raw:
            return False, raw
        return False, raw or "PARSE_FAILED"
    except Exception as e:
        return False, f"CHECK_ERROR:{type(e).__name__}"

