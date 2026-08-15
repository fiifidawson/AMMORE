import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from pymilvus import MilvusClient

from .config import config
from .document_metadata import build_overview
from .splade_compat import bootstrap_cmd


def _uses_local_milvus() -> bool:
    return "://" not in config.mmore.milvus_uri


def _corpus_slug(corpus: Path) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", corpus.name).strip("_") or "ammore_corpus"
    return f"ammore_{name}"


# mmore's retriever ignores config.collection_name and always queries "my_docs",
# so AMMORE indexes every corpus into that single collection and tracks which one
# is currently loaded via an active_corpus stamp file.
_COLLECTION = "my_docs"


def _collection_name(corpus: Path) -> str:
    return _COLLECTION


def document_metadata_path(corpus: Path) -> Path:
    work_dir = Path(tempfile.gettempdir()) / "ammore" / _corpus_slug(corpus)
    return work_dir / "document_metadata.md"


def _active_stamp() -> Path:
    return Path(tempfile.gettempdir()) / "ammore" / "active_corpus.txt"


def _is_active(corpus: Path) -> bool:
    stamp = _active_stamp()
    if not stamp.exists():
        return False
    return stamp.read_text(encoding="utf-8").strip() == _corpus_slug(corpus)


def _mark_active(corpus: Path) -> None:
    stamp = _active_stamp()
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(_corpus_slug(corpus), encoding="utf-8")


def _corpus_signature(corpus: Path) -> str:
    files = (
        [corpus]
        if corpus.is_file()
        else sorted(p for p in corpus.rglob("*") if p.is_file())
    )
    parts = []
    for path in files:
        stat = path.stat()
        name = str(path.name if corpus.is_file() else path.relative_to(corpus))
        parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "\n".join(parts)


def _indexed_stamp(work_dir: Path) -> Path:
    return work_dir / "indexed_corpus.txt"


def _local_db_exists() -> bool:
    return Path(config.mmore.milvus_uri).exists()


def _has_indexed_stamp(corpus: Path, work_dir: Path) -> bool:
    stamp = _indexed_stamp(work_dir)
    if not stamp.exists() or not _local_db_exists():
        return False
    return stamp.read_text(encoding="utf-8") == _corpus_signature(corpus)


def _mark_indexed(corpus: Path, work_dir: Path) -> None:
    _indexed_stamp(work_dir).write_text(_corpus_signature(corpus), encoding="utf-8")


def _ensure_database() -> None:
    # milvus-lite creates its database on demand, but a real server ships with only
    # "default" -- a fresh container otherwise fails the index run with
    # "database not found[database=my_db]".
    if _uses_local_milvus():
        return
    client = None
    try:
        client = MilvusClient(uri=config.mmore.milvus_uri)
        if config.mmore.milvus_db not in client.list_databases():
            client.create_database(config.mmore.milvus_db)
            print(f"Created Milvus database '{config.mmore.milvus_db}'")
    except Exception as e:
        print(f"Warning: could not create '{config.mmore.milvus_db}': {e}")
    finally:
        if client is not None:
            client.close()


def _drop_collection_if_exists(name: str) -> None:
    if _uses_local_milvus():
        return
    client = None
    try:
        client = MilvusClient(
            uri=config.mmore.milvus_uri, db_name=config.mmore.milvus_db
        )
        if name in list(client.list_collections()):  # type: ignore[call-overload]
            client.drop_collection(name)
            print(f"Dropped previous '{name}' collection")
    except Exception as e:
        print(f"Warning: could not drop '{name}': {e}")
    finally:
        if client is not None:
            client.close()


def _already_indexed(corpus: Path, work_dir: Path, collection_name: str) -> bool:
    # Opening a milvus-lite file starts a local server in this Python process.
    # That keeps the DB locked, so the subprocess that runs `mmore index` cannot
    # open it. Only use this optimization for remote Milvus servers.
    if _uses_local_milvus():
        return _has_indexed_stamp(corpus, work_dir)

    client = None
    try:
        client = MilvusClient(
            uri=config.mmore.milvus_uri, db_name=config.mmore.milvus_db
        )
        collections = list(client.list_collections())  # type: ignore[call-overload]
        return collection_name in collections
    except Exception:
        return False
    finally:
        if client is not None:
            client.close()


def _run_index(index_cfg: Path) -> None:
    # Inserting into a freshly-created Milvus collection can segfault natively on
    # Windows (the pymilvus sparse-vector path); the failed attempt still creates
    # the empty collection, so a retry appends to the now-existing one and works.
    for attempt in (1, 2, 3):
        try:
            subprocess.run(
                bootstrap_cmd("index", "--config-file", str(index_cfg)),
                check=True,
            )
            return
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            print(
                f"  index attempt {attempt} crashed (fresh-collection insert); retrying..."
            )


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
            "db": {"uri": config.mmore.milvus_uri, "name": config.mmore.milvus_db},
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
        "db": {"uri": config.mmore.milvus_uri, "name": config.mmore.milvus_db},
        "hybrid_search_weight": 0.5,
        "k": config.retrieval.max_matches,
        "collection_name": collection_name,
        "reranker_model_name": None,
    }
    path = work_dir / "retriever.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def prepare_corpus(corpus: Path) -> Path:
    """Index the corpus and return a retriever config path.

    Accepts either a folder of documents (runs mmore process) or a JSON file
    with pre-extracted papers (skips mmore process, uses extracted_text).
    Caches the indexed collection.
    """
    if corpus.is_file() and corpus.suffix.lower() == ".json":
        return _prepare_from_json(corpus)

    if not corpus.exists() or not corpus.is_dir():
        raise FileNotFoundError(f"Corpus not found: {corpus}")

    collection = _collection_name(corpus)
    work_dir = Path(tempfile.gettempdir()) / "ammore" / _corpus_slug(corpus)
    work_dir.mkdir(parents=True, exist_ok=True)

    _ensure_database()
    retriever_cfg = _write_retriever_cfg(work_dir, collection)
    merged = work_dir / "process_out" / "merged" / "merged_results.jsonl"

    if _is_active(corpus) and _already_indexed(corpus, work_dir, collection):
        print(
            f"Corpus '{corpus.name}' already loaded into '{collection}', skipping processing."
        )
        # still build the overview
        build_overview(merged, work_dir)
        return retriever_cfg

    _drop_collection_if_exists(collection)

    # Re-extracting many PDFs spawns one worker per CPU and can exhaust memory.
    # When the extraction output is already cached, reuse it and only re-chunk/index.
    if os.getenv("AMMORE_REUSE_EXTRACTION") == "1" and merged.exists():
        print(f"Reusing cached extraction at {merged} (skipping mmore process)")
    else:
        print(f"Processing corpus at {corpus}...")
        process_cfg = _write_process_cfg(corpus, work_dir)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mmore",
                "process",
                "--config-file",
                str(process_cfg),
            ],
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
    _run_index(index_cfg)

    _mark_indexed(corpus, work_dir)
    _mark_active(corpus)
    return retriever_cfg


def _prepare_from_json(json_path: Path) -> Path:
    collection = _collection_name(json_path)
    work_dir = Path(tempfile.gettempdir()) / "ammore" / _corpus_slug(json_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    merged = work_dir / "process_out" / "merged" / "merged_results.jsonl"
    merged.parent.mkdir(parents=True, exist_ok=True)

    _ensure_database()
    retriever_cfg = _write_retriever_cfg(work_dir, collection)

    if _is_active(json_path) and _already_indexed(json_path, work_dir, collection):
        print(
            f"Corpus '{json_path.name}' already loaded into '{collection}', skipping rebuild."
        )
        if not (work_dir / "document_metadata.md").exists():
            _write_synthetic_jsonl(json_path, merged, work_dir)
        return retriever_cfg

    _drop_collection_if_exists(collection)

    print(f"Loading papers from {json_path}...")
    n = _write_synthetic_jsonl(json_path, merged, work_dir)
    print(f"Wrote {n} papers to {merged}")

    print("Chunking papers...")
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
            str(merged),
        ],
        check=True,
    )

    print("Indexing chunks into Milvus...")
    index_cfg = _write_index_cfg(work_dir, collection)
    _run_index(index_cfg)

    _mark_indexed(json_path, work_dir)
    _mark_active(json_path)
    return retriever_cfg


def _write_synthetic_jsonl(json_path: Path, merged: Path, work_dir: Path) -> int:
    papers = json.loads(json_path.read_text(encoding="utf-8"))
    title_map: dict[str, str] = {}
    overview_lines = ["# Corpus overview\n"]
    used_slugs: set[str] = set()
    written = 0

    with open(merged, "w", encoding="utf-8") as out:
        for i, paper in enumerate(papers, 1):
            text = (paper.get("extracted_text") or "").strip()
            if not text:
                continue
            title = (paper.get("title") or f"Paper {i}").strip() or f"Paper {i}"
            base = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")[:60]
            slug = base or f"paper_{i}"
            # disambiguate duplicate titles
            unique = slug
            k = 2
            while unique in used_slugs:
                unique = f"{slug}_{k}"
                k += 1
            used_slugs.add(unique)
            file_path = f"{unique}.json"

            entry = {
                "text": text,
                "modalities": [],
                "metadata": {
                    "file_path": file_path,
                    "document_type": "json",
                    "processor_type": "AMMOREJSONProcessor",
                    "title": title,
                    "abstract": paper.get("abstract", ""),
                    "authors": paper.get("authors", ""),
                    "url": paper.get("url", ""),
                    "year": paper.get("year"),
                },
            }
            # ensure_ascii=True because mmore.postprocess reads jsonl with the system
            # default encoding (cp1252 on Windows) and falls over on en-dashes etc.
            out.write(json.dumps(entry) + "\n")
            written += 1

            title_map[file_path] = title
            overview_lines.append(f"## {title}")
            overview_lines.append(f"- file: {file_path}")
            abstract = (paper.get("abstract") or "").strip()
            if abstract:
                overview_lines.append(f"- summary: {abstract}")
            overview_lines.append("")

    (work_dir / "document_metadata.md").write_text(
        "\n".join(overview_lines), encoding="utf-8"
    )
    (work_dir / "title_map.json").write_text(
        json.dumps(title_map, ensure_ascii=False), encoding="utf-8"
    )
    (work_dir / "metadata_mode.txt").write_text("json", encoding="utf-8")
    return written
