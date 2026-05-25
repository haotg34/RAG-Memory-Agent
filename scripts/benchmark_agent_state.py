import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError


REQUIRED_EVENTS = {
    "chat_request_received",
    "route_start",
    "route_decided",
    "cascade_attempt_start",
    "rewrite_query_start",
    "rewrite_query_done",
    "memory_context_ready",
    "rag_retrieval_done",
    "llm_generate_start",
    "llm_generate_done",
    "persist_start",
    "usage_log_done",
    "chat_done",
}


def _read_rows(path: Path, limit: int | None) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def _post_chat(base_url: str, payload: dict) -> tuple[dict, float, str | None]:
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
        return body, (time.perf_counter() - start) * 1000, None
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        return {}, (time.perf_counter() - start) * 1000, f"HTTP_{e.code}:{detail}"
    except URLError as e:
        return {}, (time.perf_counter() - start) * 1000, f"URL_ERROR:{e}"


def _checker_parse_ok(raw: str | None) -> bool:
    if not raw:
        return True
    try:
        parsed = json.loads(raw)
        return isinstance(parsed, dict) and isinstance(parsed.get("passed"), bool)
    except Exception:
        return False


def _route_parse_ok(route: dict) -> bool:
    if route.get("rule_hit"):
        return True
    score = route.get("decided_score")
    if score is None:
        return route.get("decided_tier") in {"medium", "large", "small"}
    try:
        value = float(score)
    except Exception:
        return False
    return 0.0 <= value <= 1.0


def _to_row(idx: int, source: dict, result: dict, latency_ms: float, error: str | None) -> dict:
    agent_state = result.get("agent_state") or {}
    events = agent_state.get("events") or []
    event_names = [e.get("name") for e in events]
    route = result.get("route") or {}
    missing_events = sorted(REQUIRED_EVENTS - set(event_names))
    checker_events = [e for e in events if e.get("name") == "answer_check_done"]
    checker_parse_values = [_checker_parse_ok((e.get("data") or {}).get("checker_raw")) for e in checker_events]

    return {
        "idx": idx,
        "type": source.get("type", ""),
        "question": source.get("question", ""),
        "ok": int(error is None and bool(result.get("response"))),
        "error": error or "",
        "latency_ms": round(latency_ms, 2),
        "final_stage": agent_state.get("current_stage", ""),
        "event_count": len(events),
        "state_complete": int(not missing_events and agent_state.get("current_stage") == "done"),
        "missing_events": "|".join(missing_events),
        "route_tier": route.get("decided_tier"),
        "route_score": route.get("decided_score"),
        "rule_hit": route.get("rule_hit"),
        "route_parse_ok": int(_route_parse_ok(route)),
        "checker_parse_ok": int(all(checker_parse_values) if checker_parse_values else 1),
        "has_error_stage": int(any(e.get("stage") == "error" for e in events)),
        "event_names": "|".join(str(x) for x in event_names),
    }


def _avg(rows: list[dict], key: str) -> float:
    values = [float(r[key]) for r in rows]
    return sum(values) / len(values) if values else 0.0


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/agent_state_benchmark_questions.csv")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output", default="outputs/agent_state_benchmark_details.csv")
    parser.add_argument("--summary-output", default="outputs/agent_state_benchmark_summary.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = _read_rows(Path(args.input), args.limit)
    details = []
    run_id = int(time.time())
    for idx, row in enumerate(rows, 1):
        print(f"[{idx}/{len(rows)}] {row.get('question')}")
        result, latency_ms, error = _post_chat(
            args.base_url,
            {
                "user_id": "agent_state_benchmark",
                "session_id": f"agent_state_{run_id}_{idx}",
                "message": row.get("question", ""),
            },
        )
        details.append(_to_row(idx, row, result, latency_ms, error))

    type_counter = Counter(r["type"] for r in details)
    summary = {
        "sample_count": len(details),
        "type_distribution": dict(type_counter),
        "success_rate": round(_avg(details, "ok") * 100, 2),
        "state_complete_rate": round(_avg(details, "state_complete") * 100, 2),
        "route_parse_ok_rate": round(_avg(details, "route_parse_ok") * 100, 2),
        "checker_parse_ok_rate": round(_avg(details, "checker_parse_ok") * 100, 2),
        "error_stage_coverage_rate": round(_avg(details, "has_error_stage") * 100, 2),
        "avg_latency_ms": round(sum(float(r["latency_ms"]) for r in details) / max(len(details), 1), 2),
        "route_mix": dict(Counter(str(r["route_tier"]) for r in details)),
    }
    _write_csv(Path(args.output), details)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
