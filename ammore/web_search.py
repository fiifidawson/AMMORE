from .config import config, get_tavily_key
from .mmore_client import cite_id


def _shorten(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}\n[...truncated...]"


def search_web(query: str = "") -> str:
    if not query.strip():
        return "ERROR: 'query' is required. Call again with a specific query string."
    if not config.websearch.enabled:
        return "ERROR: web search is disabled in config.yaml."

    try:
        from tavily import TavilyClient
    except ImportError:
        return "ERROR: tavily-python not installed."

    try:
        client = TavilyClient(api_key=get_tavily_key())
        response = client.search(
            query=query,
            max_results=config.websearch.max_results,
            search_depth=config.websearch.search_depth,  # type: ignore[arg-type]
        )
    except Exception as e:
        return f"ERROR: Tavily search failed: {e}"

    results = response.get("results", [])
    if not results:
        return "No web results found."

    chunks = []
    total = 0
    for i, r in enumerate(results, 1):
        title = r.get("title", "untitled")
        url = r.get("url", "")
        content = _shorten(
            r.get("content", "").strip(), config.retrieval.max_chunk_chars
        )
        gid = cite_id(f"web:{url}")
        block = f"[Web {gid} | {title} | {url}]\n{content}"
        if (
            config.retrieval.max_total_chars
            and total + len(block) > config.retrieval.max_total_chars
        ):
            chunks.append(
                f"[... {len(results) - i + 1} more web result(s) omitted ...]"
            )
            break
        chunks.append(block)
        total += len(block)

    return "\n\n---\n\n".join(chunks)
