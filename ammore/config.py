"""Loads config.yaml once and exposes it as a single object."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


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
    max_matches: int
    min_similarity: float
    max_messages: int

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        mmore_cfg = data["mmore"]

        # Resolve retriever config path relative to AMMORE root if relative
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
            max_matches=data["retrieval"]["max_matches"],
            min_similarity=data["retrieval"]["min_similarity"],
            max_messages=data["loop"]["max_messages"],
        )


# Resolve path relative to AMMORE root (one level up from this file)
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
config = Config.load(_CONFIG_PATH)


def get_api_key() -> str:
    """Mistral API key — still loaded from .env for security."""
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise ValueError("MISTRAL_API_KEY not set in .env")
    return key
