import requests

from .config import config


def _shorten(text: str, limit: int) -> str:
    """Keep the head and tail of an over-long chunk with a marker in between."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return f"{text[:head]}\n[...truncated {len(text) - limit} chars...]\n{text[-tail:]}"


def retrieve(
    query: str, max_matches: int | None = None, min_similarity: float | None = None
) -> str:
    """Call mmore retriever API and return formatted chunks."""
    max_matches = max_matches if max_matches is not None else config.max_matches
    min_similarity = (
        min_similarity if min_similarity is not None else config.min_similarity
    )

    try:
        response = requests.post(
            config.retriever_url,
            json={
                "query": query,
                "fileIds": [],
                "maxMatches": max_matches,
                "minSimilarity": min_similarity,
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.ConnectionError:
        return "ERROR: can't connect to mmore."
    except requests.RequestException as e:
        return f"ERROR: {e}"

    results = response.json()
    if not results:
        return "No results found."

    chunks = []
    total = 0
    for i, r in enumerate(results, 1):
        file_id = r.get("fileId", "unknown")
        content = _shorten(r.get("content", "").strip(), config.max_chunk_chars)
        block = f"[Chunk {i} | {file_id}]\n{content}"
        if config.max_total_chars and total + len(block) > config.max_total_chars:
            chunks.append(f"[... {len(results) - i + 1} more chunk(s) omitted ...]")
            break
        chunks.append(block)
        total += len(block)

    return "\n\n---\n\n".join(chunks)
