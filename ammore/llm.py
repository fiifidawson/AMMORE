import os
from dotenv import load_dotenv

load_dotenv()


def get_model_client():
    provider = os.getenv("PROVIDER", "mistral").lower()

    if provider == "mistral":
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set in .env")

        model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

        return OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
        )

    elif provider == "ollama":
        from autogen_ext.models.ollama import OllamaChatCompletionClient

        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        return OllamaChatCompletionClient(model=model, host=host)

    else:
        raise ValueError(f"Unknown PROVIDER '{provider}'. Use 'mistral' or 'ollama'.")
