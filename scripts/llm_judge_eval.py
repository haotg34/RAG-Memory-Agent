import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from services.chat_service import get_llm  # noqa: E402


def _parse_list_cell(value) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [text]


def _read_rows(path: Path, limit: int | None) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("judge输出不是JSON")
    return json.loads(m.group(0))


def _clamp_score(value) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))


def _judge_row(llm, row: dict) -> dict:
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    ground_truths = _parse_list_cell(row.get("ground_truths") or row.get("ground_truth"))
    ground_truth = "\n".join(f"- {x}" for x in ground_truths)
    prompt = f"""
你是企业知识库RAG评测专家。请基于用户问题、标准答案和系统回答，分别给出两个0到1的分数。

评分定义：
1. answer_relevancy：系统回答是否直接回答用户问题，是否少跑题、少冗余。
2. answer_correctness：系统回答与标准答案在事实层面是否一致，是否覆盖关键要点。

只输出JSON，格式如下：
{{
  "answer_relevancy": 0.0,
  "answer_correctness": 0.0,
  "reason": "简短原因"
}}

用户问题：
{question}

标准答案：
{ground_truth}

系统回答：
{answer}
""".strip()
    raw = llm.invoke(prompt).content
    parsed = _extract_json(raw)
    return {
        **row,
        "llm_judge_answer_relevancy": _clamp_score(parsed.get("answer_relevancy")),
        "llm_judge_answer_correctness": _clamp_score(parsed.get("answer_correctness")),
        "llm_judge_reason": str(parsed.get("reason") or "").strip(),
    }


def _avg(rows: list[dict], key: str) -> float:
    values = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
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
    parser.add_argument("--input", default="outputs/ragas_enterprise_faithfulness_collected.csv")
    parser.add_argument("--output", default="outputs/llm_judge_answer_metrics.csv")
    parser.add_argument("--summary-output", default="outputs/llm_judge_answer_metrics_summary.json")
    parser.add_argument("--judge-provider", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = _read_rows(Path(args.input), args.limit)
    llm = get_llm(
        llm_provider=args.judge_provider,
        llm_model=args.judge_model or settings.llm_large_model or settings.llm_model,
    )
    judged = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] 评测：{row.get('question', '')}")
        judged.append(_judge_row(llm, row))

    summary = {
        "sample_count": len(judged),
        "answer_relevancy": round(_avg(judged, "llm_judge_answer_relevancy"), 4),
        "answer_correctness": round(_avg(judged, "llm_judge_answer_correctness"), 4),
        "method": "LLM-as-Judge, no embedding dependency",
    }
    _write_csv(Path(args.output), judged)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
