import requests

from .config import config


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
    for i, r in enumerate(results, 1):
        file_id = r.get("fileId", "unknown")
        content = r.get("content", "").strip()
        chunks.append(f"[Chunk {i} | {file_id}]\n{content}")

    return "\n\n---\n\n".join(chunks)
