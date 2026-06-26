import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

import dacite
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
class MistralConfig:
    model: str = "mistral-small-latest"
    base_url: str = "https://api.mistral.ai/v1"


@dataclass
class OpenAIConfig:
    model: str = "gpt-4o-mini"
    base_url: str = ""  # empty -> OpenAI default endpoint


@dataclass
class KimiConfig:
    model: str = "kimi-k2.6"
    base_url: str = "https://api.moonshot.ai/v1"


@dataclass
class OllamaConfig:
    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"


@dataclass
class MmoreConfig:
    retriever_url: str = ""
    auto_launch: bool = True
    retriever_config_file: str = ""
    host: str = "127.0.0.1"
    port: int = 8001
    startup_timeout: int = 180
    milvus_uri: str = "auto"
    milvus_db: str = "my_db"

    def __post_init__(self):
        if not self.retriever_url:
            self.retriever_url = f"http://{self.host}:{self.port}/v1/retrieve"
        self.milvus_uri = _resolve_milvus_uri(self.milvus_uri)


@dataclass
class RetrievalConfig:
    max_matches: int = 3
    min_similarity: float = -1.0
    max_chunk_chars: int = 1200
    max_total_chars: int = 5000


@dataclass
class LoopConfig:
    max_messages: int = 24
    context_head: int = 2
    context_tail: int = 8
    context_mode: str = "headtail"  # "headtail" (small models) | "full" (large context)


@dataclass
class WebsearchConfig:
    enabled: bool = False
    max_results: int = 5
    search_depth: str = "basic"


@dataclass
class DocumentMetadataConfig:
    mode: str = "cheap"


@dataclass
class Config:
    provider: str = "mistral"
    mistral: MistralConfig = field(default_factory=MistralConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    kimi: KimiConfig = field(default_factory=KimiConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    mmore: MmoreConfig = field(default_factory=MmoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    websearch: WebsearchConfig = field(default_factory=WebsearchConfig)
    document_metadata: DocumentMetadataConfig = field(
        default_factory=DocumentMetadataConfig
    )


def load_config(path: str | Path, cls: type = Config) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = dacite.from_dict(data_class=cls, data=data)
    if cfg.mmore.retriever_config_file and not os.path.isabs(
        cfg.mmore.retriever_config_file
    ):
        cfg.mmore.retriever_config_file = str(
            (Path(path).parent / cfg.mmore.retriever_config_file).resolve()
        )
    return cfg


config = load_config(_AMMORE_ROOT / "config.yaml")


def get_api_key() -> str:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise ValueError("MISTRAL_API_KEY not set in .env")
    return key


def get_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    return key


def get_kimi_key() -> str:
    key = os.getenv("KIMI_API_KEY")
    if not key:
        raise ValueError("KIMI_API_KEY not set in .env")
    return key


def get_tavily_key() -> str:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise ValueError("TAVILY_API_KEY not set in .env")
    return key
