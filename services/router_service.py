import re
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    tier: str
    score: float | None
    rule_hit: str | None
    force_large: bool
    raw: str | None = None
    parse_ok: bool = True


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


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _normalize_score(value) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return v


def _parse_score(text: str) -> float | None:
    data = _extract_json(text)
    if data and "score" in data:
        return _normalize_score(data.get("score"))
    if not text:
        return None
    m = re.search(r"([01](?:\.\d+)?)", text.strip())
    return _normalize_score(m.group(1)) if m else None


def score_difficulty_detail(query: str, llm) -> dict:
    prompt = f"""
评估用户问题难度，必须只输出JSON，不要输出解释。
JSON格式：
{{"score": 0.4, "reason": "一句话理由"}}

打分规则：
闲聊=0.1，普通文档查询=0.4，复杂推理=0.8及以上。
用户问题：{query}
""".strip()
    try:
        r = llm.invoke(prompt)
        raw = getattr(r, "content", "") or ""
        score = _parse_score(raw)
        data = _extract_json(raw) or {}
        return {
            "score": score,
            "reason": str(data.get("reason") or ""),
            "raw": raw,
            "parse_ok": score is not None,
        }
    except Exception as e:
        return {
            "score": None,
            "reason": f"score_error:{type(e).__name__}",
            "raw": "",
            "parse_ok": False,
        }


def score_difficulty(query: str, llm) -> float | None:
    return score_difficulty_detail(query, llm).get("score")


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

    detail = score_difficulty_detail(query, llm_for_score)
    score = detail.get("score")
    if score is None:
        return RouteDecision(
            tier="medium",
            score=None,
            rule_hit="score_parse_failed",
            force_large=False,
            raw=detail.get("raw"),
            parse_ok=False,
        )

    if score < th_small:
        return RouteDecision(tier="small", score=score, rule_hit=None, force_large=False, raw=detail.get("raw"))
    if score < th_medium:
        return RouteDecision(tier="medium", score=score, rule_hit=None, force_large=False, raw=detail.get("raw"))
    return RouteDecision(tier="large", score=score, rule_hit=None, force_large=False, raw=detail.get("raw"))


def check_answer_yesno(user_query: str, answer: str, llm_checker) -> tuple[bool, str]:
    prompt = f"""
用户问题：{user_query}
现有回答：{answer}
判断回答是否准确完整、无幻觉、满足用户需求。
必须只输出JSON，不要输出解释。
JSON格式：
{{"passed": true, "reason": "一句话理由"}}
""".strip()
    try:
        r = llm_checker.invoke(prompt)
        raw = (getattr(r, "content", "") or "").strip()
        data = _extract_json(raw)
        if data and isinstance(data.get("passed"), bool):
            return bool(data["passed"]), json.dumps(data, ensure_ascii=False)
        upper = raw.upper()
        if "YES" in upper:
            return True, raw
        if "NO" in upper:
            return False, raw
        return False, raw or "PARSE_FAILED"
    except Exception as e:
        return False, f"CHECK_ERROR:{type(e).__name__}"

