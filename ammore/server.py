"""FastAPI server for the web UI. The CLI stays in main.py."""

import asyncio
import json
import threading
import uuid
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .agents import create_planner, reset_search_state
from .corpus import document_metadata_path, prepare_corpus
from .document_metadata import load_title_map
from .llm import get_model_client
from .loop import build_team, task_text
from .main import save_output
from .mmore_client import set_title_map
from .retriever_launcher import auto_retriever

app = FastAPI(title="ammore")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS = Path("outputs")
UPLOADS = Path("uploads")

_corpus = {"status": "none", "path": None, "error": None, "files": 0}
_retriever_stack = ExitStack()
_runs: dict[str, asyncio.Queue] = {}
_run_lock = asyncio.Lock()


# ---------- reviews (history) ----------


@app.get("/api/reviews")
def list_reviews():
    if not OUTPUTS.exists():
        return []
    items = []
    for p in sorted(OUTPUTS.glob("review_*.md"), reverse=True):
        first = p.read_text(encoding="utf-8").splitlines()
        title = first[0].lstrip("# ").strip() if first else p.stem
        items.append(
            {
                "name": p.name,
                "title": title,
                "mtime": p.stat().st_mtime,
            }
        )
    return items


@app.get("/api/reviews/{name}")
def get_review(name: str):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "bad name")
    p = OUTPUTS / name
    if not p.exists():
        raise HTTPException(404, "not found")
    return {"name": name, "content": p.read_text(encoding="utf-8")}


# ---------- corpus ----------


class CorpusBody(BaseModel):
    path: str


@app.post("/api/corpus/inspect")
def inspect_corpus(body: CorpusBody):
    p = Path(body.path)
    if p.is_file() and p.suffix.lower() == ".json":
        try:
            papers = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(400, f"can't read JSON: {e}")
        usable = [x for x in papers if (x.get("extracted_text") or "").strip()]
        titles = [(x.get("title") or "?")[:80] for x in usable[:10]]
        return {
            "kind": "json",
            "total": len(papers),
            "usable": len(usable),
            "sample": titles,
        }
    if p.is_dir():
        files = [f.name for f in sorted(p.iterdir()) if f.is_file()]
        return {"kind": "folder", "total": len(files), "sample": files[:20]}
    raise HTTPException(400, "path is neither a folder nor a .json file")


@app.post("/api/corpus/upload")
async def upload_corpus(files: list[UploadFile]):
    dest = UPLOADS / datetime.now().strftime("%Y%m%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = Path(f.filename or "file").name
        (dest / name).write_bytes(await f.read())
    return {"path": str(dest.resolve()), "count": len(files)}


def _prepare(path_str: str):
    global _retriever_stack
    try:
        corpus = Path(path_str)
        retriever_cfg = prepare_corpus(corpus)
        set_title_map(load_title_map(document_metadata_path(corpus).parent))
        _retriever_stack.close()
        _retriever_stack = ExitStack()
        _retriever_stack.enter_context(auto_retriever(retriever_cfg))
        _corpus.update(status="ready", path=path_str, error=None)
    except Exception as e:
        _corpus.update(status="error", error=str(e))


@app.post("/api/corpus/prepare")
def prepare(body: CorpusBody):
    if _corpus["status"] == "indexing":
        raise HTTPException(409, "already indexing")
    p = Path(body.path)
    if not p.exists():
        raise HTTPException(400, "path not found")
    _corpus.update(status="indexing", path=body.path, error=None)
    threading.Thread(target=_prepare, args=(body.path,), daemon=True).start()
    return {"status": "indexing"}


@app.get("/api/corpus/status")
def corpus_status():
    return _corpus


# ---------- plan + run ----------


class PlanBody(BaseModel):
    question: str


@app.post("/api/plan")
async def make_plan(body: PlanBody):
    model_client = get_model_client()
    planner = create_planner(model_client)
    result = await planner.run(task=body.question)
    text = result.messages[-1].content
    lines = [
        line.split(".", 1)[1].strip() if "." in line[:4] else line.strip()
        for line in str(text).splitlines()
        if line.strip()
    ]
    return {"raw": text, "questions": lines}


class RunBody(BaseModel):
    question: str
    plan: list[str]
    web_search: bool = False
    web_only: bool = False


async def _run_loop(run_id: str, body: RunBody):
    queue = _runs[run_id]

    def emit(kind, agent="", content=""):
        queue.put_nowait({"kind": kind, "agent": agent, "content": content})

    try:
        async with _run_lock:
            reset_search_state()
            model_client = get_model_client()
            team = build_team(
                model_client,
                web_enabled=body.web_search or body.web_only,
                web_only=body.web_only,
            )
            plan = "\n".join(
                f"{i}. {q}" for i, q in enumerate(body.plan, 1) if q.strip()
            )
            task = task_text(body.question, plan, web_only=body.web_only)

            collected = []
            async for msg in team.run_stream(task=task):
                collected.append(msg)
                src = getattr(msg, "source", "?")
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content:
                    emit("message", src, content)
                elif isinstance(content, list):
                    for item in content:
                        name = getattr(item, "name", None)
                        args = getattr(item, "arguments", "")
                        if name and args:
                            emit("tool_call", src, f"{name}({args})")

            save_output(body.question, collected)
            latest = max(OUTPUTS.glob("review_*.md"), key=lambda p: p.stat().st_mtime)
            emit("done", content=latest.name)
    except Exception as e:
        emit("error", content=str(e))
    finally:
        queue.put_nowait(None)


@app.post("/api/run")
async def start_run(body: RunBody):
    if not body.web_only and _corpus["status"] != "ready":
        raise HTTPException(409, "no corpus ready")
    run_id = uuid.uuid4().hex[:12]
    _runs[run_id] = asyncio.Queue()
    asyncio.get_event_loop().create_task(_run_loop(run_id, body))
    return {"run_id": run_id}


@app.get("/api/run/{run_id}/events")
async def run_events(run_id: str):
    queue = _runs.get(run_id)
    if queue is None:
        raise HTTPException(404, "unknown run")

    async def stream():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        _runs.pop(run_id, None)

    return StreamingResponse(stream(), media_type="text/event-stream")


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8500)


if __name__ == "__main__":
    main()
