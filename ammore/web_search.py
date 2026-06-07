from .config import config, get_tavily_key


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
    for i, r in enumerate(results, 1):
        title = r.get("title", "untitled")
        url = r.get("url", "")
        content = r.get("content", "").strip()
        chunks.append(f"[Web {i} | {title} | {url}]\n{content}")

    return "\n\n---\n\n".join(chunks)
