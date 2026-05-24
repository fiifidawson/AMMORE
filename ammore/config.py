import os
import platform
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

_AMMORE_ROOT = Path(__file__).parent.parent
load_dotenv(_AMMORE_ROOT / ".env")


def _resolve_milvus_uri(value: str) -> str:
    if value != "auto":
        return value
    # milvus-lite has no Windows wheel -> need a running Milvus server there
    if platform.system() == "Windows":
        return "http://127.0.0.1:19530"
    return str((_AMMORE_ROOT / "milvus.db").resolve())


@dataclass
class Config:
    provider: str
    mistral_model: str
    mistral_base_url: str
    ollama_model: str
    ollama_base_url: str
    retriever_url: str
    auto_launch: bool
    retriever_config_file: str
    retriever_host: str
    retriever_port: int
    startup_timeout: int
    milvus_uri: str
    milvus_db: str
    max_matches: int
    min_similarity: float
    max_chunk_chars: int
    max_total_chars: int
    max_messages: int
    context_head: int
    context_tail: int
    websearch_enabled: bool
    websearch_max_results: int
    websearch_search_depth: str
    metadata_mode: str
    metadata_in_prompt: bool

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        mistral = data.get("mistral", {})
        ollama = data.get("ollama", {})
        mmore_cfg = data.get("mmore", {})
        retrieval = data.get("retrieval", {})
        loop = data.get("loop", {})
        websearch = data.get("websearch", {})
        metadata = data.get("document_metadata", {})

        retriever_cfg = mmore_cfg.get("retriever_config_file", "")
        if retriever_cfg and not os.path.isabs(retriever_cfg):
            retriever_cfg = str((Path(path).parent / retriever_cfg).resolve())

        host = mmore_cfg.get("host", "127.0.0.1")
        port = mmore_cfg.get("port", 8001)

        return cls(
            provider=data.get("provider", "mistral"),
            mistral_model=mistral.get("model", "mistral-small-latest"),
            mistral_base_url=mistral.get("base_url", "https://api.mistral.ai/v1"),
            ollama_model=ollama.get("model", "llama3.2:3b"),
            ollama_base_url=ollama.get("base_url", "http://localhost:11434"),
            retriever_url=mmore_cfg.get(
                "retriever_url", f"http://{host}:{port}/v1/retrieve"
            ),
            auto_launch=mmore_cfg.get("auto_launch", True),
            retriever_config_file=retriever_cfg,
            retriever_host=host,
            retriever_port=port,
            startup_timeout=mmore_cfg.get("startup_timeout", 180),
            milvus_uri=_resolve_milvus_uri(mmore_cfg.get("milvus_uri", "auto")),
            milvus_db=mmore_cfg.get("milvus_db", "my_db"),
            max_matches=retrieval.get("max_matches", 3),
            min_similarity=retrieval.get("min_similarity", -1.0),
            max_chunk_chars=retrieval.get("max_chunk_chars", 1200),
            max_total_chars=retrieval.get("max_total_chars", 5000),
            max_messages=loop.get("max_messages", 24),
            context_head=loop.get("context_head", 2),
            context_tail=loop.get("context_tail", 8),
            websearch_enabled=websearch.get("enabled", False),
            websearch_max_results=websearch.get("max_results", 5),
            websearch_search_depth=websearch.get("search_depth", "basic"),
            metadata_mode=metadata.get("mode", "cheap"),
            metadata_in_prompt=metadata.get("include_in_prompt", True),
        )


config = Config.load(_AMMORE_ROOT / "config.yaml")


def get_api_key() -> str:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise ValueError("MISTRAL_API_KEY not set in .env")
    return key


def get_tavily_key() -> str:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise ValueError("TAVILY_API_KEY not set in .env")
    return key
