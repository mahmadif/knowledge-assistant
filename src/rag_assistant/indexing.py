from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

from .config import settings

console = Console()

COLLECTION_METADATA = {"hnsw:space": "cosine"}


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Get the embedding model based on the configured settings."""
    # Local sentence-transformer inference; CPU is sufficient for this demo.
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    return [chunk for chunk in splitter.split_documents(documents) if chunk.page_content.strip()]


def _init_chroma_store() -> Chroma:
    """Initialize the Chroma vector store."""
    embedding_model = get_embedding_model()
    return Chroma(
        collection_name="confluence_knowledge",
        embedding_function=embedding_model,
        persist_directory=settings.chroma_persist_dir,
        # set distance metric to cosine for semantic search
        collection_metadata=COLLECTION_METADATA,
    )


def build_or_update_index(documents: list[Document], reset: bool = False) -> int:
    """Build or update the Chroma vector store with the provided documents."""
    console.print(f"Splitting {len(documents)} document(s) into chunks...", style="purple")
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No non-empty chunks were produced.")

    if reset:
        console.print("Resetting the existing collection...", style="purple")
        existing = _init_chroma_store()
        existing.delete_collection()

    store = _init_chroma_store()
    # No page-level upsert/deduplication is implemented in this demo.
    store.add_documents(chunks)

    return len(chunks)


def open_vector_store() -> Chroma:
    """Open the existing Chroma vector store."""
    return _init_chroma_store()
