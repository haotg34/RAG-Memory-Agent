import argparse
import csv
import json
import sys
import time
from pathlib import Path
from urllib import request
from urllib.error import URLError, HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from services.chat_service import get_llm  # noqa: E402


def _read_rows(path: Path, limit: int | None) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit:
        return rows[:limit]
    return rows


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
    if "||" in text:
        return [x.strip() for x in text.split("||") if x.strip()]
    return [text]


def _call_agent(
    base_url: str,
    user_id: str,
    session_id: str,
    question: str,
    provider: str | None,
    model: str | None,
) -> dict:
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "message": question,
    }
    if provider:
        payload["llm_provider"] = provider
    if model:
        payload["llm_model"] = model

    req = request.Request(
        url=f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Agent HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"无法连接 Agent 服务：{e}") from e


def _collect_dataset(args) -> tuple[dict, list[dict]]:
    rows = _read_rows(Path(args.input), args.limit)
    if not rows or "question" not in rows[0]:
        raise ValueError("输入CSV必须包含 question 列")

    has_ground_truths = any((r.get("ground_truths") or r.get("ground_truth") or "").strip() for r in rows)
    data = {"question": [], "answer": [], "contexts": []}
    if has_ground_truths:
        data["ground_truth"] = []
        data["ground_truths"] = []

    collected_rows = []
    for i, row in enumerate(rows, 1):
        question = (row.get("question") or "").strip()
        if not question:
            continue

        session_id = f"{args.session_prefix}_{int(time.time())}_{i}"
        print(f"[{i}/{len(rows)}] 调用Agent：{question}")
        result = _call_agent(
            base_url=args.base_url,
            user_id=args.user_id,
            session_id=session_id,
            question=question,
            provider=args.provider,
            model=args.model,
        )

        answer = result.get("response") or ""
        contexts = result.get("retrieved_knowledge") or []
        if not isinstance(contexts, list):
            contexts = [str(contexts)]
        contexts = [str(x) for x in contexts if str(x).strip()]

        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)

        gt = row.get("ground_truths") or row.get("ground_truth")
        ground_truths = _parse_list_cell(gt)
        if has_ground_truths:
            data["ground_truth"].append(ground_truths[0] if ground_truths else "")
            data["ground_truths"].append(ground_truths)

        collected_rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": json.dumps(contexts, ensure_ascii=False),
                "ground_truths": json.dumps(ground_truths, ensure_ascii=False),
            }
        )
    return data, collected_rows


def _save_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_ragas_metrics(has_ground_truths: bool, metric_set: str):
    from ragas.metrics import faithfulness, answer_relevancy

    if metric_set == "faithfulness":
        return [faithfulness]

    if metric_set == "answer_relevancy":
        return [answer_relevancy]

    if metric_set == "answer_correctness":
        if not has_ground_truths:
            raise ValueError("answer_correctness指标需要输入CSV提供 ground_truths 或 ground_truth")
        from ragas.metrics import answer_correctness

        return [answer_correctness]

    if metric_set == "retrieval":
        if not has_ground_truths:
            raise ValueError("retrieval指标需要输入CSV提供 ground_truths 或 ground_truth")
        from ragas.metrics import context_precision, context_recall

        return [context_precision, context_recall]

    if metric_set == "generation":
        return [faithfulness, answer_relevancy]

    metrics = [faithfulness, answer_relevancy]
    try:
        from ragas.metrics import context_relevancy

        metrics.append(context_relevancy)
    except Exception:
        pass

    if has_ground_truths:
        from ragas.metrics import context_precision, context_recall, answer_correctness

        metrics.extend([context_precision, context_recall, answer_correctness])
    return metrics


def _build_embeddings():
    from langchain_openai import OpenAIEmbeddings

    api_key, base_url, model = settings.resolve_embedding()
    return OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url)


def _run_ragas(data: dict, output: Path, judge_provider: str | None, judge_model: str | None, metric_set: str):
    from datasets import Dataset
    from ragas import evaluate

    has_ground_truths = "ground_truths" in data
    dataset = Dataset.from_dict(data)
    llm = get_llm(
        llm_provider=judge_provider,
        llm_model=judge_model or settings.llm_large_model or settings.llm_model,
    )
    result = evaluate(
        dataset=dataset,
        metrics=_build_ragas_metrics(has_ground_truths, metric_set),
        llm=llm,
        embeddings=_build_embeddings(),
    )

    df = result.to_pandas()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print("\n整体平均得分：")
    print(result)
    print(f"\n明细结果已保存：{output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/ragas_questions.csv")
    parser.add_argument("--output", default="outputs/ragas_eval_results.csv")
    parser.add_argument("--collected-output", default="outputs/ragas_collected_dataset.csv")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--user-id", default="ragas_eval")
    parser.add_argument("--session-prefix", default="ragas")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge-provider", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--metric-set",
        choices=[
            "all",
            "retrieval",
            "generation",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
        ],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    data, collected_rows = _collect_dataset(args)
    _save_rows(Path(args.collected_output), collected_rows)
    print(f"\nAgent输出数据已保存：{args.collected_output}")

    if args.collect_only:
        return
    _run_ragas(data, Path(args.output), args.judge_provider, args.judge_model, args.metric_set)


if __name__ == "__main__":
    main()
