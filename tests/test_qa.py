from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag_assistant.qa import KnowledgeAssistant, build_prompt, format_context


def test_format_context_with_documents():
    docs = [
        Document(
            page_content="page_content1",
            metadata={"title": "Runbooks", "source": "https://example.com"},
        ),
        Document(
            page_content="page_content2",
            metadata={"title": "", "source": "example.md"},
        ),
        Document(
            page_content="page_content3",
            metadata={"title": "", "source": ""},
        ),
    ]

    formatted = format_context(docs)

    expected_items = [
        "page_content1",
        "page_content2",
        "page_content3",
        "Runbooks",
        "example.md",
        "Unknown source",
        "[S1]",
    ]

    assert all(item in formatted for item in expected_items)


def test_format_context_empty_list():
    formatted = format_context([])
    assert "No relevant context found in the indexed documents." in formatted


def test_build_prompt_includes_context_and_question():
    docs = [
        Document(
            page_content="Our API rate limit is 100 requests per minute.",
            metadata={"title": "API Docs"},
        )
    ]
    question = "What is the API rate limit?"

    prompt = build_prompt(question=question, documents=docs)

    assert "100 requests per minute" in prompt
    assert "What is the API rate limit?" in prompt


def test_qa_pipeline_with_mocked_retriever_and_llm():
    # Mock retriever returning a single relevant document
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="Onboarding takes 3 days.", metadata={"title": "HR"})]

    # Mock LLM returning a deterministic AIMessage
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Onboarding takes 3 days.")

    retrieved_docs = mock_retriever.invoke("How long is onboarding?")
    prompt = build_prompt(question="How long is onboarding?", documents=retrieved_docs)
    response = mock_llm.invoke(prompt)

    assert len(retrieved_docs) == 1
    assert response.content == "Onboarding takes 3 days."

    mock_retriever.invoke.assert_called_once()
    mock_llm.invoke.assert_called_once()


def test_answer_preserves_full_content():
    long_text = "A" * 500 + " Critical rule at the end of the chunk."
    mock_doc = Document(
        page_content=long_text,
        metadata={"title": "access_management.md", "id": "page-123"},
    )

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [mock_doc]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Critical rule at the end.")

    with (
        patch("rag_assistant.qa.open_vector_store", return_value=mock_store),
        patch("rag_assistant.qa.get_chat_model", return_value=mock_llm),
    ):
        assistant = KnowledgeAssistant()
        answer, sources = assistant.answer("What is the critical rule?")

    assert answer == "Critical rule at the end."
    assert len(sources) == 1
    # assert len(sources[0]["excerpt"]) == 500
    # assert "Critical rule at the end of the chunk." not in sources[0]["excerpt"]
    assert sources[0]["content"] == long_text
