"""Spawn the mmore retriever as a subprocess if it's not already running."""

import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import requests

from .config import config
from .splade_compat import bootstrap_cmd


def is_retriever_running() -> bool:
    """Check whether something is already serving on the retriever URL."""
    parsed = urlparse(config.mmore.retriever_url)
    health_url = f"{parsed.scheme}://{parsed.netloc}/docs"
    try:
        requests.get(health_url, timeout=1)
        return True
    except requests.RequestException:
        return False


def _wait_until_ready(timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_retriever_running():
            return True
        time.sleep(1)
    return False


@contextmanager
def auto_retriever(config_file=None):
    """Context manager that ensures the retriever is up for the duration of the block.

    - If the retriever is already running, do nothing (yield immediately).
    - Otherwise, spawn `python -m mmore retrieve ...` and wait until ready.
    - On exit, terminate the subprocess we started (only if we started it).
    - `config_file` overrides the retriever config from config.yaml.
    """
    if not config.mmore.auto_launch:
        yield None
        return

    if is_retriever_running():
        print(f"mmore retriever already running at {config.mmore.retriever_url}")
        yield None
        return

    cfg = str(config_file) if config_file else config.mmore.retriever_config_file
    cmd = bootstrap_cmd(
        "retrieve",
        "--config-file",
        cfg,
        "--host",
        config.mmore.host,
        "--port",
        str(config.mmore.port),
    )

    log_path = Path("traces") / "retriever.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")

    print(f"Starting mmore retriever: {' '.join(cmd)}")
    print(f"(retriever output -> {log_path})")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        if not _wait_until_ready(config.mmore.startup_timeout):
            proc.terminate()
            raise RuntimeError(
                f"mmore retriever did not become ready within {config.mmore.startup_timeout}s. "
                f"Check {log_path} for the actual error."
            )
        print(f"Retriever ready at {config.mmore.retriever_url}\n")
        yield proc
    finally:
        if proc.poll() is None:
            print("\nStopping mmore retriever...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_file.close()
