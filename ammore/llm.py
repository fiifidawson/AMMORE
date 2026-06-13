from .config import config, get_api_key, get_kimi_key, get_openai_key

# kimi and mistral are OpenAI-compatible endpoints autogen doesn't recognize,
# so they need an explicit model_info
_OPENAI_COMPAT_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": True,
}


def get_model_client():
    if config.provider == "mistral":
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        return OpenAIChatCompletionClient(
            model=config.mistral.model,
            api_key=get_api_key(),
            base_url=config.mistral.base_url,
            parallel_tool_calls=False,
            model_info=_OPENAI_COMPAT_INFO,
        )

    elif config.provider == "kimi":
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        return OpenAIChatCompletionClient(
            model=config.kimi.model,
            api_key=get_kimi_key(),
            base_url=config.kimi.base_url,
            model_info=_OPENAI_COMPAT_INFO,
        )

    elif config.provider == "openai":
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        kwargs = {"model": config.openai.model, "api_key": get_openai_key()}
        if config.openai.base_url:
            kwargs["base_url"] = config.openai.base_url
        return OpenAIChatCompletionClient(**kwargs)

    elif config.provider == "ollama":
        from autogen_ext.models.ollama import OllamaChatCompletionClient

        return OllamaChatCompletionClient(
            model=config.ollama.model,
            host=config.ollama.base_url,
        )

    else:
        raise ValueError(
            f"Unknown provider '{config.provider}' in config.yaml. "
            "Use 'mistral', 'openai', 'kimi' or 'ollama'."
        )


def get_judge_client():
    """Kimi K2 client for the eval judge. A different vendor than the system
    under test, so the judge stays independent."""
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    return OpenAIChatCompletionClient(
        model=config.kimi.model,
        api_key=get_kimi_key(),
        base_url=config.kimi.base_url,
        model_info=_OPENAI_COMPAT_INFO,
    )
