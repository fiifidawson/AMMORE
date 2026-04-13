import os
import requests
from dotenv import load_dotenv

load_dotenv()

MMORE_URL = os.getenv("MMORE_RETRIEVER_URL", "http://127.0.0.1:8001/v1/retrieve")


def retrieve(query: str, max_matches: int = 10, min_similarity: float = -1) -> str:
    """Call mmore retriever API and return formatted chunks."""
    try:
        response = requests.post(
            MMORE_URL,
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
