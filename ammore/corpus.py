import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from pymilvus import MilvusClient

from .config import config


def _collection_name(corpus: Path) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", corpus.name).strip("_") or "ammore_corpus"
    return f"ammore_{name}"


def _already_indexed(collection_name: str) -> bool:
    try:
        client = MilvusClient(uri="http://127.0.0.1:19530", db_name="my_db")
        collections = list(client.list_collections())  # type: ignore[call-overload]
        return collection_name in collections
    except Exception:
        return False


def _write_process_cfg(corpus: Path, work_dir: Path) -> Path:
    cfg = {
        "data_path": str(corpus.resolve()),
        "google_drive_ids": [],
        "previous_results": None,
        "dispatcher_config": {
            "output_path": str((work_dir / "process_out").resolve()),
            "use_fast_processors": True,
            "distributed": False,
            "extract_images": False,
        },
    }
    path = work_dir / "process.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def _write_index_cfg(work_dir: Path, collection_name: str) -> Path:
    results_jsonl = work_dir / "process_out" / "merged" / "results.jsonl"
    cfg = {
        "indexer": {
            "dense_model": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "is_multimodal": False,
            },
            "sparse_model": {"model_name": "splade", "is_multimodal": False},
            "db": {"uri": "http://127.0.0.1:19530", "name": "my_db"},
        },
        "collection_name": collection_name,
        "documents_path": str(results_jsonl.resolve()),
    }
    path = work_dir / "index.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def _write_retriever_cfg(work_dir: Path, collection_name: str) -> Path:
    cfg = {
        "db": {"uri": "http://127.0.0.1:19530", "name": "my_db"},
        "hybrid_search_weight": 0.5,
        "k": config.max_matches,
        "collection_name": collection_name,
    }
    path = work_dir / "retriever.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def prepare_corpus(corpus: Path) -> Path:
    """Index the corpus folder if needed and return a retriever config path.

    Skips processing + indexing when the corresponding Milvus collection
    already exists. Returns the path of the retriever config to use.
    """
    if not corpus.exists() or not corpus.is_dir():
        raise FileNotFoundError(f"Corpus folder not found: {corpus}")

    collection = _collection_name(corpus)
    work_dir = Path(tempfile.gettempdir()) / "ammore" / collection
    work_dir.mkdir(parents=True, exist_ok=True)

    retriever_cfg = _write_retriever_cfg(work_dir, collection)

    if _already_indexed(collection):
        print(f"Corpus already indexed as '{collection}', skipping processing.")
        return retriever_cfg

    print(f"Processing corpus at {corpus}...")
    process_cfg = _write_process_cfg(corpus, work_dir)
    subprocess.run(
        [sys.executable, "-m", "mmore", "process", "--config-file", str(process_cfg)],
        check=True,
    )

    print("Indexing chunks into Milvus...")
    index_cfg = _write_index_cfg(work_dir, collection)
    subprocess.run(
        [sys.executable, "-m", "mmore", "index", "--config-file", str(index_cfg)],
        check=True,
    )

    return retriever_cfg
