import json
from unittest.mock import MagicMock, patch

import pytest

from eval import run_eval as eval_mod
from eval.run_eval import (
    JudgeScore,
    _compute_summary_metrics,
    _write_markdown_report,
    compute_retrieval_metrics,
    evaluate,
)


def test_compute_retrieval_metrics_single_and_multi_hop():
    assert compute_retrieval_metrics(["on_call.md", "hr.md"], ["on_call.md"]) == (
        1.0,
        1.0,
    )

    assert compute_retrieval_metrics(["hr.md", " On_Call.md "], ["on_call.md"]) == (
        1.0,
        0.5,
    )

    assert compute_retrieval_metrics(
        ["access_management.md", "hr.md"],
        ["access_management.md", "security_policy.md"],
    ) == (
        0.0,
        1.0,
    )

    assert compute_retrieval_metrics(
        ["hr.md", "access_management.md", "security_policy.md"],
        ["access_management.md", "security_policy.md"],
    ) == (1.0, 0.5)

    assert compute_retrieval_metrics([], ["on_call.md"]) == (0.0, 0.0)
    assert compute_retrieval_metrics(["on_call.md"], []) == (0.0, 0.0)


def test_compute_summary_metrics():
    rows = [
        {"hit": "1", "rr": "1.00", "faithful": 1, "correct": 1},
        {"hit": "0", "rr": "0.50", "faithful": 1, "correct": 0},
        {"hit": "N/A", "rr": "N/A", "faithful": 1, "correct": 1},
        {"hit": "N/A", "rr": "N/A", "faithful": 0, "correct": 0},
    ]

    metrics = _compute_summary_metrics(rows)

    assert metrics["total"] == 4
    assert metrics["answerable_count"] == 2
    assert metrics["unanswerable_count"] == 2
    assert metrics["hit_rate"] == pytest.approx(50.0)
    assert metrics["mrr"] == pytest.approx(0.75)
    assert metrics["faith_rate"] == pytest.approx(75.0)
    assert metrics["corr_rate"] == pytest.approx(50.0)
    assert metrics["abst_rate"] == pytest.approx(50.0)


def test_write_markdown_report_escapes_pipes(tmp_path, monkeypatch):
    report_file = tmp_path / "results.md"
    monkeypatch.setattr(eval_mod, "RESULTS_PATH", report_file)

    rows = [
        {
            "id": "Q1",
            "category": "single_hop",
            "question": "Option A | Option B?",
            "hit": "1",
            "rr": "1.00",
            "faithful": 1,
            "correct": 1,
            "reason": "Matched A | B cleanly.",
        }
    ]
    metrics = _compute_summary_metrics(rows)
    _write_markdown_report(rows, metrics)

    content = report_file.read_text(encoding="utf-8")
    assert "# RAG Evaluation Benchmark" in content
    assert "Option A \\| Option B?" in content
    assert "Matched A \\| B cleanly." in content


def test_evaluate_end_to_end_with_mocks(tmp_path, monkeypatch):
    dataset_file = tmp_path / "golden_dataset.json"
    report_file = tmp_path / "results.md"
    dataset_file.write_text(
        json.dumps(
            [
                {
                    "question_id": "Q1",
                    "category": "single_hop",
                    "question": "What is the escalation timeout?",
                    "ground_truth": "15 minutes.",
                    "expected_sources": ["on_call.md"],
                    "answerable": True,
                },
                {
                    "question_id": "Q2",
                    "category": "unanswerable",
                    "question": "Which cloud hosts Kubernetes?",
                    "ground_truth": "Not covered.",
                    "expected_sources": [],
                    "answerable": False,
                },
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(eval_mod, "DATASET_PATH", dataset_file)
    monkeypatch.setattr(eval_mod, "RESULTS_PATH", report_file)

    mock_judge = MagicMock()
    mock_judge.invoke.side_effect = [
        JudgeScore(faithfulness=1, correctness=1, reason="Exact match."),
        JudgeScore(faithfulness=1, correctness=1, reason="Proper abstention."),
    ]

    mock_assistant = MagicMock()
    mock_assistant.llm.with_structured_output.return_value = mock_judge
    mock_assistant.answer.side_effect = [
        (
            "15 minutes.",
            [
                {
                    "label": "S1",
                    "title": "on_call.md",
                    "content": "Full chunk with 15 minutes rule.",
                }
            ],
        ),
        (
            "Cannot determine from the indexed documents.",
            [
                {
                    "label": "S1",
                    "title": "release_process.md",
                    "content": "Release rules.",
                }
            ],
        ),
    ]

    with patch("eval.run_eval.KnowledgeAssistant", return_value=mock_assistant):
        evaluate()

    # Regression check: verify full chunk content was passed into the judge prompt
    first_judge_prompt = mock_judge.invoke.call_args_list[0].args[0]
    assert "Full chunk with 15 minutes rule." in first_judge_prompt

    report_text = report_file.read_text(encoding="utf-8")
    assert "**100.0%**" in report_text
    assert "| Q1 |" in report_text
    assert "| Q2 |" in report_text
