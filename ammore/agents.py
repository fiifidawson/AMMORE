from autogen_agentchat.agents import AssistantAgent
from .mmore_client import retrieve


def search_documents(query: str, max_matches: int = 5) -> str:
    """Search the document corpus via mmore."""
    return retrieve(query, max_matches=max_matches)


def create_planner_only(model_client):
    """Standalone planner used for the human validation step."""
    return AssistantAgent(
        name="Planner",
        model_client=model_client,
        system_message=(
            "You are a research planner. Given a question, break it down into "
            "3-5 specific search queries to run against a document corpus.\n\n"
            "Output ONLY a numbered list of queries, nothing else.\n"
            "Example:\n"
            "1. What methods are used for X?\n"
            "2. What are the results reported for Y?\n"
            "3. What limitations are mentioned?\n"
        ),
    )


def create_agents(model_client):
    """Create the 4 agents for the retrieval loop."""

    planner = AssistantAgent(
        name="Planner",
        model_client=model_client,
        description="Breaks the question into sub-queries.",
        system_message=(
            "You are a research planner. Break the user's question into "
            "3-5 specific search queries for the document corpus.\n"
            "Output a numbered list only."
        ),
    )

    retriever = AssistantAgent(
        name="Retriever",
        model_client=model_client,
        description="Searches the document corpus using mmore.",
        system_message=(
            "You are a retrieval agent. Use the search_documents tool to find "
            "relevant chunks for each query you receive.\n"
            "Don't invent content -- only report what the tool returns."
        ),
        tools=[search_documents],
        reflect_on_tool_use=False,  # avoid crash when tool returns an error
    )

    critic = AssistantAgent(
        name="Critic",
        model_client=model_client,
        description="Checks if we have enough information to write the answer.",
        system_message=(
            "You review the retrieved chunks and decide if we have enough coverage.\n\n"
            "If yes: say COVERAGE_OK\n"
            "If not: say NEEDS_MORE and suggest 1-3 additional queries to fill the gaps."
        ),
    )

    writer = AssistantAgent(
        name="Writer",
        model_client=model_client,
        description="Writes the final synthesis.",
        system_message=(
            "Based on the retrieved chunks, write a structured literature review.\n\n"
            "## Summary\n"
            "## Key Findings\n"
            "## Gaps & Limitations\n"
            "## Sources\n"
            "List ONLY the file IDs that appear in the chunks above (e.g. [Chunk 3 | abc123]). "
            "Do not add any papers from your training knowledge. If a paper is not in the chunks, do not cite it.\n\n"
            "End with: TERMINATE"
        ),
    )

    return planner, retriever, critic, writer
