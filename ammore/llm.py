from .config import config, get_api_key


def get_model_client():
    if config.provider == "mistral":
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        return OpenAIChatCompletionClient(
            model=config.mistral.model,
            api_key=get_api_key(),
            base_url=config.mistral.base_url,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
        )

    elif config.provider == "ollama":
        from autogen_ext.models.ollama import OllamaChatCompletionClient

        return OllamaChatCompletionClient(
            model=config.ollama.model,
            host=config.ollama.base_url,
        )

    else:
        raise ValueError(
            f"Unknown provider '{config.provider}' in config.yaml. Use 'mistral' or 'ollama'."
        )
