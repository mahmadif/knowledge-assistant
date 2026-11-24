import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from rag_assistant.config import settings
from rag_assistant.qa import KnowledgeAssistant

console = Console()

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "results.md"


class JudgeScore(BaseModel):
    faithfulness: int = Field(
        description="1 if the answer is strictly grounded in the retrieved sources without hallucination, else 0."
    )
    correctness: int = Field(
        description="1 if the answer accurately matches the ground truth (or properly abstains when unanswerable), "
        "else 0."
    )
    reason: str = Field(description="Brief one-sentence explanation for the scores.")


JUDGE_PROMPT = """
You are a strict RAG evaluation judge. Grade the assistant's answer against the ground truth and retrieved context.

Question ID: {question_id}
Question: {question}
Answerable from docs: {answerable}
Ground Truth: {ground_truth}
Retrieved Context:
{retrieved_context}

Assistant Answer: {answer}

Rules:
- faithfulness: Score 1 ONLY if every claim in Assistant Answer is supported by the Retrieved Context above (or if it 
states it does not know). Score 0 if it invents or alters facts not in Retrieved Context.
- correctness:
  - If Answerable is True: Score 1 if Assistant Answer accurately includes all core facts from Ground Truth without 
  contradicting them (additional true details from the Retrieved Context are acceptable as long as 
  the core facts are clear). Score 0 if any core fact is missing, buried, or contradicted.
  - If Answerable is False: Score 1 ONLY if Assistant Answer clearly states the information is not available 
  in the documents. Score 0 if it guesses.
"""


def compute_retrieval_metrics(
    retrieved: list[str],
    expected: list[str],
) -> tuple[float, float]:
    """Compute Hit@k and MRR for a single question's retrieved vs expected sources."""
    if not expected or not retrieved:
        return 0.0, 0.0

    ret_list = [r.strip().lower() for r in retrieved]
    exp_set = {e.strip().lower() for e in expected}

    hit_all = float(exp_set.issubset(ret_list))
    first_rr = next(
        (1.0 / rank for rank, ret in enumerate(ret_list, start=1) if ret in exp_set),
        0.0,
    )
    return hit_all, first_rr


def _judge_answer(
    judge_llm: Any,
    item: dict[str, Any],
    retrieved_context: str,
    answer: str,
) -> JudgeScore:
    """Invoke the LLM judge on a single question/answer pair."""
    prompt = JUDGE_PROMPT.format(
        question_id=item["question_id"],
        question=item["question"],
        answerable=bool(item.get("answerable", True)),
        ground_truth=item["ground_truth"],
        retrieved_context=retrieved_context or "None",
        answer=answer,
    )
    return judge_llm.invoke(prompt)


def _compute_summary_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate retrieval, faithfulness, correctness, and abstention metrics."""
    n = len(rows)
    hits = [float(r["hit"]) for r in rows if r["hit"] != "N/A"]
    mrrs = [float(r["rr"]) for r in rows if r["rr"] != "N/A"]
    abstentions = [r["correct"] for r in rows if r["hit"] == "N/A"]

    return {
        "total": n,
        "answerable_count": len(hits),
        "unanswerable_count": len(abstentions),
        "hit_rate": (sum(hits) / len(hits) * 100) if hits else 0.0,
        "mrr": (sum(mrrs) / len(mrrs)) if mrrs else 0.0,
        "faith_rate": (sum(r["faithful"] for r in rows) / n * 100) if n else 0.0,
        "corr_rate": (sum(r["correct"] for r in rows) / n * 100) if n else 0.0,
        "abst_rate": ((sum(abstentions) / len(abstentions) * 100) if abstentions else 0.0),
    }


def _print_summary_table(
    rows: list[dict[str, Any]],
    metrics: dict[str, float],
) -> None:
    """Render the per-question evaluation table and summary line to the console."""
    table = Table(title="RAG Evaluation Results")
    for col in ["ID", "Category", "Question", "Hit@k", "MRR", "Faithful", "Correct", "Judge Note"]:
        table.add_column(col)

    for r in rows:
        q_short = r["question"][:42] + ("..." if len(r["question"]) > 42 else "")
        table.add_row(
            r["id"], r["category"], q_short, r["hit"], r["rr"], str(r["faithful"]), str(r["correct"]), r["reason"]
        )

    console.print(table)
    console.print(
        f"\n[bold]Summary (k={settings.retrieval_k}, chunk_size={settings.chunk_size}):[/bold] "
        f"Hit@{settings.retrieval_k}={metrics['hit_rate']:.1f}% | MRR={metrics['mrr']:.2f} | "
        f"Faithfulness={metrics['faith_rate']:.1f}% | Correctness={metrics['corr_rate']:.1f}% | "
        f"Abstention={metrics['abst_rate']:.1f}%"
    )


def _write_markdown_report(
    rows: list[dict[str, Any]],
    metrics: dict[str, float],
) -> None:
    """Write the Markdown benchmark summary and per-question breakdown to RESULTS_PATH."""

    now_cet = lambda: f"{datetime.now(ZoneInfo('Europe/Berlin')):%Y-%m-%d %H:%M %Z}"  # noqa: E731

    detail_lines = [
        "| ID | Category | Question | Hit@k | MRR | Faithful | Correct | Judge Note |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for r in rows:
        q_text = r["question"].replace("|", "\\|")
        note = r["reason"].replace("|", "\\|")
        detail_lines.append(
            f"| {r['id']} | {r['category']} | {q_text} | {r['hit']} | {r['rr']} "
            f"| {r['faithful']} | {r['correct']} | {note} |"
        )

    report = (
        "# RAG Evaluation Benchmark\n\n"
        f"- **Run At:** `{now_cet()}`\n"
        f"- **Dataset Size:** `{metrics['total']}` questions (`{metrics['answerable_count']}` answerable, "
        f"`{metrics['unanswerable_count']}` unanswerable)\n"
        f"- **LLM Provider:** `{settings.llm_provider}`\n"
        f"- **LLM Model:** `{settings.ollama_model}`\n"
        f"- **Embedding Model:** `{settings.embedding_model}`\n"
        f"- **Chunk Size / Overlap:** `{settings.chunk_size}` / `{settings.chunk_overlap}`\n"
        f"- **Retrieval Top-K:** `{settings.retrieval_k}`\n\n"
        "## Summary Metrics\n\n"
        "| Metric | Score |\n| :--- | :--- |\n"
        f"| Multi-Source Hit Rate @ {settings.retrieval_k} | **{metrics['hit_rate']:.1f}%** |\n"
        f"| Mean Reciprocal Rank (MRR) | **{metrics['mrr']:.2f}** |\n"
        f"| Faithfulness (Groundedness) | **{metrics['faith_rate']:.1f}%** |\n"
        f"| Overall Answer Correctness | **{metrics['corr_rate']:.1f}%** |\n"
        f"| Unanswerable Abstention Rate | **{metrics['abst_rate']:.1f}%** |\n\n"
        "## Detailed Results\n\n" + "\n".join(detail_lines) + "\n"
    )
    RESULTS_PATH.write_text(report, encoding="utf-8")


def evaluate() -> None:
    """Run the RAG evaluation benchmark on the golden dataset and save results to a Markdown report."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Golden dataset not found at {DATASET_PATH}")

    dataset: list[dict[str, Any]] = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    assistant = KnowledgeAssistant()
    judge_llm = assistant.llm.with_structured_output(JudgeScore)

    rows: list[dict[str, Any]] = []
    for item in tqdm(dataset, desc="Evaluating", unit="question"):
        answer, sources = assistant.answer(item["question"])
        retrieved_titles = [s.get("title") or s.get("source", "") for s in sources]
        retrieved_context = "\n\n---\n\n".join(
            f"[{s.get('label', '')}] {s.get('title') or s.get('source', '')}:\n{s.get('content', '')}" for s in sources
        )

        answerable = bool(item.get("answerable", True))
        hit, rr = (
            compute_retrieval_metrics(retrieved_titles, item.get("expected_sources", []))
            if answerable
            else (None, None)
        )
        verdict = _judge_answer(judge_llm, item, retrieved_context, answer)

        rows.append(
            {
                "id": item["question_id"],
                "category": item.get("category", "general"),
                "question": item["question"],
                "hit": f"{hit:.0f}" if hit is not None else "N/A",
                "rr": f"{rr:.2f}" if rr is not None else "N/A",
                "faithful": verdict.faithfulness,
                "correct": verdict.correctness,
                "reason": verdict.reason,
            }
        )

    metrics = _compute_summary_metrics(rows)
    _print_summary_table(rows, metrics)
    _write_markdown_report(rows, metrics)


if __name__ == "__main__":
    evaluate()
