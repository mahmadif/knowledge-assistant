from pathlib import Path

from langchain_core.documents import Document
from rich.console import Console

console = Console()


def load_local_documents(path: str) -> list[Document]:
    """Load Markdown and plain-text files recursively."""
    root = Path(path)

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    console.print(f"Loading documents from {root}...", style="purple")

    docs = []
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in {".md", ".txt"}:
            content = file_path.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": str(file_path), "title": file_path.name},
                    )
                )

    if not docs:
        raise ValueError(f"No non-empty Markdown or text files found in {root}")

    return docs


def load_confluence_documents(url, username, api_token, space_key):
    """Load pages from an authorized Confluence space."""
    values = {
        "CONFLUENCE_URL": url,
        "CONFLUENCE_USERNAME": username,
        "CONFLUENCE_API_TOKEN": api_token,
        "CONFLUENCE_SPACE_KEY": space_key,
    }

    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError("Missing Confluence settings: " + ", ".join(missing))

    from langchain_community.document_loaders import ConfluenceLoader

    console.print("Loading documents from Confluence space...", style="purple")

    loader = ConfluenceLoader(
        url=url,
        username=username,
        api_key=api_token,
        space_key=space_key,
        limit=100,
    )

    return [doc for doc in loader.load() if getattr(doc, "page_content", "").strip()]
