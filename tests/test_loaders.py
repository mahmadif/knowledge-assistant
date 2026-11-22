from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_assistant import loaders


def test_load_local_documents_reads_md_and_txt_recursively(tmp_path):
    # Create valid files in root and nested subfolder
    doc1 = tmp_path / "guide.md"
    doc1.write_text("# Setup Guide\nInstall dependencies.", encoding="utf-8")

    nested = tmp_path / "sub"
    nested.mkdir()
    doc2 = nested / "notes.TXT"  # Uppercase extension to test .lower()
    doc2.write_text("  Release notes content.  ", encoding="utf-8")

    # Create ignored files (unsupported extension & whitespace-only)
    (tmp_path / "ignore.json").write_text('{"key": "value"}', encoding="utf-8")
    (tmp_path / "empty.md").write_text("   \n\t  ", encoding="utf-8")

    docs = loaders.load_local_documents(str(tmp_path))

    assert len(docs) == 2
    assert docs[0].page_content == "# Setup Guide\nInstall dependencies."
    assert docs[0].metadata == {"source": str(doc1), "title": "guide.md"}

    assert docs[1].page_content == "Release notes content."
    assert docs[1].metadata == {"source": str(doc2), "title": "notes.TXT"}


def test_load_local_documents_raises_on_missing_directory(tmp_path):
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="Directory does not exist"):
        loaders.load_local_documents(str(missing_dir))


def test_load_local_documents_raises_when_no_valid_files_found(tmp_path):
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "script.py").write_text("print('hi')", encoding="utf-8")

    with pytest.raises(ValueError, match="No non-empty Markdown or text files found"):
        loaders.load_local_documents(str(tmp_path))


def test_load_confluence_documents_raises_on_missing_settings():
    with pytest.raises(RuntimeError) as exc_info:
        loaders.load_confluence_documents(
            url="https://atlassian.net",
            username="",
            api_token=None,
            space_key="ENG",
        )

    error_msg = str(exc_info.value)
    assert "Missing Confluence settings:" in error_msg
    assert "CONFLUENCE_USERNAME" in error_msg
    assert "CONFLUENCE_API_TOKEN" in error_msg
    assert "CONFLUENCE_URL" not in error_msg


@patch("langchain_community.document_loaders.ConfluenceLoader")
def test_load_confluence_documents_filters_empty_pages(mock_loader_cls):
    mock_loader = MagicMock()
    mock_loader.load.return_value = [
        Document(page_content="Valid Confluence page", metadata={"title": "Page 1"}),
        Document(page_content="   \n  ", metadata={"title": "Blank Page"}),
    ]
    mock_loader_cls.return_value = mock_loader

    docs = loaders.load_confluence_documents(
        url="https://atlassian.net",
        username="user@example.com",
        api_token="secret-token",
        space_key="ENG",
    )

    assert len(docs) == 1
    assert docs[0].page_content == "Valid Confluence page"
    mock_loader_cls.assert_called_once_with(
        url="https://atlassian.net",
        username="user@example.com",
        api_key="secret-token",
        space_key="ENG",
        limit=100,
    )
