import sys
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

import ingest


@patch("ingest.build_or_update_index", return_value=4)
@patch("ingest.load_local_documents")
def test_main_defaults_to_local_source_without_reset(
    mock_load_local,
    mock_build_index,
    monkeypatch,
):
    fake_docs = [Document(page_content="Local guide", metadata={"source": "a.md"})]
    mock_load_local.return_value = fake_docs

    monkeypatch.setattr(sys, "argv", ["ingest.py"])
    ingest.main()

    mock_load_local.assert_called_once_with(ingest.settings.data_dir)
    mock_build_index.assert_called_once_with(fake_docs, reset=False)


@patch("ingest.build_or_update_index", return_value=2)
@patch("ingest.load_local_documents")
def test_main_local_source_with_custom_path_and_reset(
    mock_load_local,
    mock_build_index,
    monkeypatch,
):
    fake_docs = [Document(page_content="Custom dir doc", metadata={"source": "b.md"})]
    mock_load_local.return_value = fake_docs

    monkeypatch.setattr(
        sys,
        "argv",
        ["ingest.py", "--source", "local", "--path", "/tmp/docs", "--reset"],
    )
    ingest.main()

    mock_load_local.assert_called_once_with("/tmp/docs")
    mock_build_index.assert_called_once_with(fake_docs, reset=True)


@patch("ingest.build_or_update_index", return_value=7)
@patch("ingest.load_confluence_documents")
def test_main_confluence_source_passes_settings_and_reset(
    mock_load_confluence,
    mock_build_index,
    monkeypatch,
):
    fake_docs = [Document(page_content="Wiki page", metadata={"title": "Runbook"})]
    mock_load_confluence.return_value = fake_docs

    monkeypatch.setattr(sys, "argv", ["ingest.py", "--source", "confluence", "--reset"])
    ingest.main()

    mock_load_confluence.assert_called_once_with(
        ingest.settings.confluence_url,
        ingest.settings.confluence_username,
        ingest.settings.confluence_api_token,
        ingest.settings.confluence_space_key,
    )
    mock_build_index.assert_called_once_with(fake_docs, reset=True)


def test_main_rejects_invalid_source_argument(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ingest.py", "--source", "notion"])

    with pytest.raises(SystemExit) as exc_info:
        ingest.main()

    assert exc_info.value.code == 2
