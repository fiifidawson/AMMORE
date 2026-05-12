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
    context_window: int
    websearch_enabled: bool
    websearch_max_results: int
    websearch_search_depth: str

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        mmore_cfg = data["mmore"]

        retriever_cfg = mmore_cfg.get("retriever_config_file", "")
        if retriever_cfg and not os.path.isabs(retriever_cfg):
            retriever_cfg = str((Path(path).parent / retriever_cfg).resolve())

        return cls(
            provider=data["provider"],
            mistral_model=data["mistral"]["model"],
            mistral_base_url=data["mistral"]["base_url"],
            ollama_model=data["ollama"]["model"],
            ollama_base_url=data["ollama"]["base_url"],
            retriever_url=mmore_cfg["retriever_url"],
            auto_launch=mmore_cfg.get("auto_launch", True),
            retriever_config_file=retriever_cfg,
            retriever_host=mmore_cfg.get("host", "127.0.0.1"),
            retriever_port=mmore_cfg.get("port", 8001),
            startup_timeout=mmore_cfg.get("startup_timeout", 30),
            milvus_uri=_resolve_milvus_uri(mmore_cfg.get("milvus_uri", "auto")),
            milvus_db=mmore_cfg.get("milvus_db", "my_db"),
            max_matches=data["retrieval"]["max_matches"],
            min_similarity=data["retrieval"]["min_similarity"],
            max_chunk_chars=data["retrieval"].get("max_chunk_chars", 2500),
            max_total_chars=data["retrieval"].get("max_total_chars", 16000),
            max_messages=data["loop"]["max_messages"],
            context_window=data["loop"].get("context_window", 6),
            websearch_enabled=data.get("websearch", {}).get("enabled", False),
            websearch_max_results=data.get("websearch", {}).get("max_results", 5),
            websearch_search_depth=data.get("websearch", {}).get(
                "search_depth", "basic"
            ),
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
