import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from pymilvus import MilvusClient

from .config import config
from .document_metadata import build_overview


def _uses_local_milvus() -> bool:
    return "://" not in config.milvus_uri


def _corpus_slug(corpus: Path) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", corpus.name).strip("_") or "ammore_corpus"
    return f"ammore_{name}"


def _collection_name(corpus: Path) -> str:
    # one collection per corpus so switching --corpus doesn't mix results
    return _corpus_slug(corpus)


def document_metadata_path(corpus: Path) -> Path:
    work_dir = Path(tempfile.gettempdir()) / "ammore" / _corpus_slug(corpus)
    return work_dir / "document_metadata.md"


def _already_indexed(collection_name: str) -> bool:
    # Opening a milvus-lite file starts a local server in this Python process.
    # That keeps the DB locked, so the subprocess that runs `mmore index` cannot
    # open it. Only use this optimization for remote Milvus servers.
    if _uses_local_milvus():
        return False

    client = None
    try:
        client = MilvusClient(uri=config.milvus_uri, db_name=config.milvus_db)
        collections = list(client.list_collections())  # type: ignore[call-overload]
        return collection_name in collections
    except Exception:
        return False
    finally:
        if client is not None:
            client.close()


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
    results_jsonl = work_dir / "postprocess_out" / "results.jsonl"
    cfg = {
        "indexer": {
            "dense_model": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "is_multimodal": False,
            },
            "sparse_model": {"model_name": "splade", "is_multimodal": False},
            "db": {"uri": config.milvus_uri, "name": config.milvus_db},
        },
        "collection_name": collection_name,
        "documents_path": str(results_jsonl.resolve()),
    }
    path = work_dir / "index.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def _write_postprocess_cfg(work_dir: Path) -> Path:
    cfg = {
        "previous_results": None,
        "pp_modules": [
            {
                "type": "chunker",
                "args": {
                    "chunking_strategy": "sentence",
                    "table_handling": "single_row",
                },
            }
        ],
        "output": {
            "output_path": str(
                (work_dir / "postprocess_out" / "results.jsonl").resolve()
            ),
            "save_each_step": True,
        },
    }
    path = work_dir / "postprocess.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def _write_retriever_cfg(work_dir: Path, collection_name: str) -> Path:
    cfg = {
        "db": {"uri": config.milvus_uri, "name": config.milvus_db},
        "hybrid_search_weight": 0.5,
        "k": config.max_matches,
        "collection_name": collection_name,
        "reranker_model_name": None,
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
    work_dir = Path(tempfile.gettempdir()) / "ammore" / _corpus_slug(corpus)
    work_dir.mkdir(parents=True, exist_ok=True)

    retriever_cfg = _write_retriever_cfg(work_dir, collection)
    merged = work_dir / "process_out" / "merged" / "merged_results.jsonl"

    if _already_indexed(collection):
        print(f"Corpus already indexed as '{collection}', skipping processing.")
        # still build the overview
        build_overview(merged, work_dir)
        return retriever_cfg

    print(f"Processing corpus at {corpus}...")
    process_cfg = _write_process_cfg(corpus, work_dir)
    subprocess.run(
        [sys.executable, "-m", "mmore", "process", "--config-file", str(process_cfg)],
        check=True,
    )

    overview = build_overview(merged, work_dir)
    if overview is not None:
        print(f"Document overview written to {overview}")

    print("Chunking processed documents...")
    postprocess_cfg = _write_postprocess_cfg(work_dir)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mmore",
            "postprocess",
            "--config-file",
            str(postprocess_cfg),
            "--input-data",
            str(work_dir / "process_out" / "merged" / "merged_results.jsonl"),
        ],
        check=True,
    )

    print("Indexing chunks into Milvus...")
    index_cfg = _write_index_cfg(work_dir, collection)
    subprocess.run(
        [sys.executable, "-m", "mmore", "index", "--config-file", str(index_cfg)],
        check=True,
    )

    return retriever_cfg
