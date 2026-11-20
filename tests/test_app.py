from unittest.mock import MagicMock, patch
import pytest

import app


@pytest.fixture(autouse=True)
def reset_global_assistant(monkeypatch):
    """Ensure app.assistant starts as None before every test."""
    monkeypatch.setattr(app, "assistant", None)


@patch("app.KnowledgeAssistant")
def test_respond_initializes_assistant_once_and_formats_sources(mock_qa_cls):
    mock_instance = MagicMock()
    mock_instance.answer.return_value = (
        "Deployments require two approvals.",
        [
            {"label": "1", "title": "Release Policy", "page_id": "48291"},
            {"label": "2", "title": "local_notes.md", "page_id": None},
        ],
    )
    mock_qa_cls.return_value = mock_instance

    # First message initializes KnowledgeAssistant
    reply1 = app.respond("How do deployments work?", history=[])

    assert "Deployments require two approvals." in reply1
    assert "**Retrieved sources**" in reply1
    assert "- **[1] Release Policy** (page ID: 48291)" in reply1
    assert "- **[2] local_notes.md**" in reply1
    assert "(page ID: None)" not in reply1

    # Second message reuses the cached assistant instance
    app.respond("Second question?", history=[])
    mock_qa_cls.assert_called_once()
    assert mock_instance.answer.call_count == 2


@patch("app.KnowledgeAssistant")
def test_respond_omits_sources_section_when_sources_empty(mock_qa_cls):
    mock_instance = MagicMock()
    mock_instance.answer.return_value = ("I don't know based on the docs.", [])
    mock_qa_cls.return_value = mock_instance

    reply = app.respond("Unrelated question?", history=[])

    assert reply == "I don't know based on the docs."
    assert "**Retrieved sources**" not in reply


@patch("app.KnowledgeAssistant")
def test_respond_handles_exceptions_gracefully(mock_qa_cls):
    mock_qa_cls.side_effect = RuntimeError("Missing Azure settings")

    reply = app.respond("Hello?", history=[])

    assert "The assistant could not complete the request." in reply
    assert "Technical detail: `RuntimeError: Missing Azure settings`" in reply


def test_gradio_demo_interface_initialized():
    assert app.demo is not None
    assert app.demo.title == "Knowledge Assistant"
