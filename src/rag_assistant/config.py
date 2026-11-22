import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    # llm provider settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")  # ollama (local, default) | azure
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # indexing settings
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    data_dir: str = os.getenv("DATA_DIR", "./data")

    # retrieval settings
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "3"))

    # azure settings
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    azure_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

    # confluence settings
    confluence_url: str = os.getenv("CONFLUENCE_URL", "")
    confluence_username: str = os.getenv("CONFLUENCE_USERNAME", "")
    confluence_api_token: str = os.getenv("CONFLUENCE_API_TOKEN", "")
    confluence_space_key: str = os.getenv("CONFLUENCE_SPACE_KEY", "")

    def validate_llm_configs(self) -> None:
        """Raise error if the selected LLM provider is not fully configured."""
        if self.llm_provider == "azure":
            required = {
                "AZURE_OPENAI_ENDPOINT": self.azure_endpoint,
                "AZURE_OPENAI_API_KEY": self.azure_api_key,
                "AZURE_OPENAI_DEPLOYMENT": self.azure_deployment,
            }
        else:
            required = {"OLLAMA_MODEL": self.ollama_model}

        missing = [key for key, value in required.items() if not value]

        if missing:
            raise RuntimeError(f"Missing settings for LLM_PROVIDER={self.llm_provider}: " + ", ".join(missing))


settings = Settings()
