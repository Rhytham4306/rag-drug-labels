"""
Pluggable LLM backend so the same RAG chain can run against:
  - Groq   (free tier, cloud, very fast — default)
  - Ollama (fully local, no API key, no internet)
  - OpenAI (paid, cloud)

Swapping providers is a one-line .env change (LLM_PROVIDER=...), which
matters in healthcare-adjacent settings where an air-gapped/local deployment
option is often a real requirement, not a nice-to-have.
"""
from abc import ABC, abstractmethod

from src.config import settings


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's text completion for the given prompts."""
        raise NotImplementedError


class GroqProvider(LLMProvider):
    def __init__(self):
        from groq import Groq

        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content


class OllamaProvider(LLMProvider):
    def __init__(self):
        import requests  # noqa: F401  (imported here to fail fast if missing)

        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class OpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content


_PROVIDERS = {
    "groq": GroqProvider,
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
}


def get_llm_provider() -> LLMProvider:
    provider_cls = _PROVIDERS.get(settings.llm_provider)
    if provider_cls is None:
        raise RuntimeError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
    return provider_cls()
