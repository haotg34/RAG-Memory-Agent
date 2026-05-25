import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402


def _read_questions(path: Path, limit: int | None) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    questions = [(r.get("question") or "").strip() for r in rows]
    questions = [q for q in questions if q]
    return questions[:limit] if limit else questions


def _post_chat(base_url: str, payload: dict) -> tuple[dict, float]:
    req = request.Request(
        url=f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=240) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Agent HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"无法连接 Agent 服务：{e}") from e
    return body, (time.perf_counter() - start) * 1000


def _price_arg(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _model_tier(model: str | None, small: str, medium: str, large: str) -> str:
    if model == small:
        return "small"
    if model == medium:
        return "medium"
    if model == large:
        return "large"
    return "unknown"


def _estimate_cost(row: dict, prices: dict[str, tuple[float, float]]) -> float:
    tier = row["tier"]
    prompt_tokens = int(row["prompt_tokens"] or 0)
    completion_tokens = int(row["completion_tokens"] or 0)
    input_price, output_price = prices.get(tier, prices["large"])
    return (prompt_tokens / 1_000_000 * input_price) + (completion_tokens / 1_000_000 * output_price)


def _pct_down(old: float, new: float) -> float:
    if old <= 0:
        return 0.0
    return (old - new) / old * 100


def _avg(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return ordered[idx]


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_one(
    *,
    base_url: str,
    user_id: str,
    session_id: str,
    question: str,
    provider: str | None,
    model: str | None,
) -> tuple[dict, float]:
    payload = {"user_id": user_id, "session_id": session_id, "message": question}
    if provider:
        payload["llm_provider"] = provider
    if model:
        payload["llm_model"] = model
    return _post_chat(base_url, payload)


def _to_row(mode: str, idx: int, question: str, result: dict, elapsed_ms: float, default_tier: str) -> dict:
    token_usage = result.get("token_usage") or {}
    route = result.get("route") or {}
    model_used = result.get("model_used") or ""
    decided_tier = route.get("decided_tier")
    return {
        "mode": mode,
        "idx": idx,
        "question": question,
        "latency_ms": round(elapsed_ms, 2),
        "model_used": model_used,
        "tier": default_tier,
        "decided_tier": decided_tier,
        "decided_score": route.get("decided_score"),
        "rule_hit": route.get("rule_hit"),
        "upgraded": int(route.get("upgraded") or 0),
        "degraded": int(route.get("degraded") or 0),
        "prompt_tokens": int(token_usage.get("prompt_tokens") or 0),
        "completion_tokens": int(token_usage.get("completion_tokens") or 0),
        "total_tokens": int(token_usage.get("total_tokens") or 0),
        "answer_chars": len(result.get("response") or ""),
        "context_count": len(result.get("retrieved_knowledge") or []),
    }


def _summarize(rows: list[dict], prices: dict[str, tuple[float, float]]) -> dict:
    baseline = [r for r in rows if r["mode"] == "baseline"]
    optimized = [r for r in rows if r["mode"] == "optimized"]
    for row in rows:
        row["estimated_cost"] = round(_estimate_cost(row, prices), 8)

    b_cost = sum(float(r["estimated_cost"]) for r in baseline)
    o_cost = sum(float(r["estimated_cost"]) for r in optimized)
    b_latency = [float(r["latency_ms"]) for r in baseline]
    o_latency = [float(r["latency_ms"]) for r in optimized]
    route_mix = Counter(r["tier"] for r in optimized)
    upgraded = sum(int(r["upgraded"]) for r in optimized)

    return {
        "sample_count": len(optimized),
        "pricing": {
            tier: {
                "input_price_per_1m_tokens": value[0],
                "output_price_per_1m_tokens": value[1],
            }
            for tier, value in prices.items()
            if tier != "unknown"
        },
        "cost_note": "成本按每百万输入/输出token价格估算；默认值是相对价格，可通过命令行参数或BENCH_*环境变量替换为真实供应商价格。",
        "baseline_total_cost": round(b_cost, 8),
        "optimized_total_cost": round(o_cost, 8),
        "cost_reduction_pct": round(_pct_down(b_cost, o_cost), 2),
        "baseline_avg_latency_ms": round(_avg(b_latency), 2),
        "optimized_avg_latency_ms": round(_avg(o_latency), 2),
        "avg_latency_reduction_pct": round(_pct_down(_avg(b_latency), _avg(o_latency)), 2),
        "baseline_p95_latency_ms": round(_p95(b_latency), 2),
        "optimized_p95_latency_ms": round(_p95(o_latency), 2),
        "p95_latency_reduction_pct": round(_pct_down(_p95(b_latency), _p95(o_latency)), 2),
        "optimized_route_mix": dict(route_mix),
        "small_medium_large_hit_rate": {
            k: round(v / max(len(optimized), 1) * 100, 2) for k, v in route_mix.items()
        },
        "upgrade_rate_pct": round(upgraded / max(len(optimized), 1) * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/ragas_employee_handbook_testset.csv")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--baseline-model", default=None)
    parser.add_argument("--small-model", default=None)
    parser.add_argument("--medium-model", default=None)
    parser.add_argument("--large-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--details-output", default="outputs/routing_benchmark_details.csv")
    parser.add_argument("--summary-output", default="outputs/routing_benchmark_summary.json")
    parser.add_argument("--small-input-price", type=float, default=_price_arg("BENCH_SMALL_INPUT_PRICE_PER_1M", 0.1))
    parser.add_argument("--small-output-price", type=float, default=_price_arg("BENCH_SMALL_OUTPUT_PRICE_PER_1M", 0.1))
    parser.add_argument("--medium-input-price", type=float, default=_price_arg("BENCH_MEDIUM_INPUT_PRICE_PER_1M", 0.4))
    parser.add_argument("--medium-output-price", type=float, default=_price_arg("BENCH_MEDIUM_OUTPUT_PRICE_PER_1M", 0.4))
    parser.add_argument("--large-input-price", type=float, default=_price_arg("BENCH_LARGE_INPUT_PRICE_PER_1M", 1.0))
    parser.add_argument("--large-output-price", type=float, default=_price_arg("BENCH_LARGE_OUTPUT_PRICE_PER_1M", 1.0))
    args = parser.parse_args()

    small_model = args.small_model or settings.llm_small_model or settings.llm_model
    medium_model = args.medium_model or settings.llm_medium_model or settings.llm_model
    large_model = args.large_model or settings.llm_large_model or settings.llm_model
    baseline_model = args.baseline_model or large_model
    provider = args.provider

    prices = {
        "small": (args.small_input_price, args.small_output_price),
        "medium": (args.medium_input_price, args.medium_output_price),
        "large": (args.large_input_price, args.large_output_price),
        "unknown": (args.large_input_price, args.large_output_price),
    }

    questions = _read_questions(Path(args.input), args.limit)
    rows = []
    run_id = int(time.time())
    print(f"开始对比评测：{len(questions)} 条问题")

    for idx, question in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] baseline: {question}")
        baseline_result, baseline_ms = _run_one(
            base_url=args.base_url,
            user_id="benchmark_baseline",
            session_id=f"bench_b_{run_id}_{idx}",
            question=question,
            provider=provider,
            model=baseline_model,
        )
        rows.append(_to_row("baseline", idx, question, baseline_result, baseline_ms, "large"))

        print(f"[{idx}/{len(questions)}] optimized: {question}")
        optimized_result, optimized_ms = _run_one(
            base_url=args.base_url,
            user_id="benchmark_optimized",
            session_id=f"bench_o_{run_id}_{idx}",
            question=question,
            provider=provider,
            model=None,
        )
        row = _to_row("optimized", idx, question, optimized_result, optimized_ms, "unknown")
        row["tier"] = _model_tier(row["model_used"], small_model, medium_model, large_model)
        rows.append(row)

    summary = _summarize(rows, prices)
    _write_csv(Path(args.details_output), rows)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n量化结果：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n明细：{args.details_output}")
    print(f"汇总：{args.summary_output}")


if __name__ == "__main__":
    main()
