from collections.abc import Sequence
from typing import Any

from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from rag_assistant.config import settings
from rag_assistant.indexing import open_vector_store

SYSTEM_INSTRUCTIONS = """You are an expert assistant answering questions about a document collection.
Use only the supplied CONTEXT. Do not rely on outside knowledge.
If the context is insufficient, say you cannot determine the answer from the indexed documents.
Address every part of the user's question by checking all provided context, not just the first one.
Answer only the specific question asked. Be concise and direct, and omit unrelated policy details.
Preserve exact numbers, timelines, and conditions from the context without reinterpreting or mixing them up.
Do not invent facts or citations. Refer to source labels when making factual claims.
If sources disagree, describe the disagreement and identify the relevant sources.

CONTEXT:
{context}
"""


def format_context(documents: Sequence[Document]) -> str:
    """Format the retrieved documents into a context string for the LLM."""
    blocks = []
    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("title") or metadata.get("source") or "Unknown source"
        page_id = metadata.get("id") or metadata.get("page_id")
        label = f"[S{index}] {source}"
        if page_id:
            label += f" (page ID: {page_id})"
        blocks.append(f"{label}\n{doc.page_content}")

    out = "\n\n---\n\n".join(blocks)

    return out if out else "No relevant context found in the indexed documents."


def build_prompt(question: str, documents: Sequence[Document]) -> str:
    """Build a prompt for the LLM using the question and retrieved documents."""
    context = format_context(documents)

    return "\n\n".join(
        [
            SYSTEM_INSTRUCTIONS.format(context=context),
            f"QUESTION:\n{question}",
            "ANSWER:",
        ]
    )


def get_chat_model() -> Any:
    """Return a LangChain chat model for the configured provider (ollama | azure)."""
    settings.validate_llm_configs()

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    return AzureChatOpenAI(
        azure_endpoint=settings.azure_endpoint,
        api_key=SecretStr(settings.azure_api_key),
        api_version=settings.azure_api_version,
        azure_deployment=settings.azure_deployment,
        temperature=0,
    )


class KnowledgeAssistant:
    """A question-answering assistant that uses a vector store and LLM to answer questions."""

    def __init__(self):
        """Initialize the assistant with a vector store and chat model."""
        self.vector_store = open_vector_store()
        self.llm = get_chat_model()

    def answer(self, question):
        """Answer a question using the indexed documents."""
        question = question.strip()

        if not question:
            return "Please enter a question.", []

        docs = self.vector_store.similarity_search(question, k=settings.retrieval_k)

        if not docs:
            return "I could not find relevant information in the indexed documents.", []

        response = self.llm.invoke(build_prompt(question, docs))

        answer = response.content if isinstance(response.content, str) else str(response.content)

        sources = []
        for i, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            sources.append(
                {
                    "label": f"S{i}",
                    "title": metadata.get("title") or metadata.get("source") or "Unknown source",
                    "page_id": metadata.get("id") or metadata.get("page_id", ""),
                    # "excerpt": doc.page_content[:500],
                    "content": doc.page_content,
                }
            )

        return answer, sources
