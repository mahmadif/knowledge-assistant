import argparse

from rich.console import Console

from rag_assistant.config import settings
from rag_assistant.indexing import build_or_update_index
from rag_assistant.loaders import load_confluence_documents, load_local_documents

console = Console()


def main():
    """Ingest documents into the RAG index."""
    console.print("Starting document ingestion...", style="purple")

    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index.")

    parser.add_argument("--source", choices=["local", "confluence"], default="local")
    parser.add_argument("--path", default=settings.data_dir)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the existing collection.",
    )

    args = parser.parse_args()
    if args.source == "local":
        documents = load_local_documents(args.path)
    else:
        documents = load_confluence_documents(
            settings.confluence_url,
            settings.confluence_username,
            settings.confluence_api_token,
            settings.confluence_space_key,
        )

    chunks = build_or_update_index(documents, reset=args.reset)

    console.print(
        f"Loaded {len(documents)} document(s) and indexed {chunks} chunk(s).",
        style="purple",
    )


if __name__ == "__main__":
    main()
