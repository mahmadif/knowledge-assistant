from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_assistant import indexing


def test_split_documents_splits_long_text_and_adds_start_index(monkeypatch):
    # Create a modified copy of the frozen settings dataclass
    test_settings = replace(indexing.settings, chunk_size=50, chunk_overlap=10)
    monkeypatch.setattr(indexing, "settings", test_settings)

    docs = [
        Document(
            page_content="A" * 120,
            metadata={"source": "https://example.com"},
        )
    ]

    chunks = indexing.split_documents(docs)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 50 for c in chunks)
    assert all("start_index" in c.metadata for c in chunks)
    assert chunks[0].metadata["source"] == "https://example.com"


def test_split_documents_filters_whitespace_only_chunks():
    docs = [
        Document(page_content="   \n\t  "),
        Document(page_content="Valid content"),
    ]

    chunks = indexing.split_documents(docs)

    assert len(chunks) == 1
    assert chunks[0].page_content == "Valid content"


def test_split_documents_empty_input_returns_empty_list():
    assert indexing.split_documents([]) == []


def test_build_or_update_index_raises_on_empty_chunks():
    empty_docs = [Document(page_content="   ")]

    with pytest.raises(ValueError, match="No non-empty chunks were produced"):
        indexing.build_or_update_index(empty_docs)


@patch("rag_assistant.indexing._init_chroma_store")
def test_build_or_update_index_adds_documents_without_reset(mock_init_store):
    mock_store = MagicMock()
    mock_init_store.return_value = mock_store

    docs = [Document(page_content="Confluence engineering documentation.")]
    count = indexing.build_or_update_index(docs, reset=False)

    assert count == 1
    mock_init_store.assert_called_once()
    mock_store.delete_collection.assert_not_called()
    mock_store.add_documents.assert_called_once()


@patch("rag_assistant.indexing._init_chroma_store")
def test_build_or_update_index_resets_collection_when_requested(mock_init_store):
    existing_store = MagicMock()
    new_store = MagicMock()
    mock_init_store.side_effect = [existing_store, new_store]

    docs = [Document(page_content="Fresh Confluence content.")]
    count = indexing.build_or_update_index(docs, reset=True)

    assert count == 1
    assert mock_init_store.call_count == 2
    existing_store.delete_collection.assert_called_once()
    new_store.add_documents.assert_called_once()


@patch("rag_assistant.indexing.Chroma")
@patch("rag_assistant.indexing.get_embedding_model")
def test_open_vector_store_initializes_chroma_with_cosine_space(
    mock_get_embeddings,
    mock_chroma,
    monkeypatch,
    tmp_path,
):
    fake_embeddings = MagicMock()
    mock_get_embeddings.return_value = fake_embeddings

    test_settings = replace(indexing.settings, chroma_persist_dir=str(tmp_path))
    monkeypatch.setattr(indexing, "settings", test_settings)

    indexing.open_vector_store()

    mock_chroma.assert_called_once_with(
        collection_name="confluence_knowledge",
        embedding_function=fake_embeddings,
        persist_directory=str(tmp_path),
        collection_metadata={"hnsw:space": "cosine"},
    )
