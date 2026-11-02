"""Provider routing for Groq Cloud and a local Ollama server."""

from __future__ import annotations

from typing import Any

import requests
from groq import Groq

GROQ_FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def list_groq_models(api_key: str) -> list[str]:
    """Return chat model IDs visible to this Groq key."""
    models = Groq(api_key=api_key).models.list()
    ids = sorted({item.id for item in models.data if getattr(item, "id", None) and not any(tag in item.id.lower() for tag in ("whisper", "guard", "tts", "speech"))})
    return ids or GROQ_FALLBACK_MODELS


def list_provider_models(provider: str, api_key: str, base_url: str = "http://localhost:11434") -> list[str]:
    """List text-capable models available to a cloud account or local Ollama."""
    if provider == "Groq":
        return list_groq_models(api_key)
    if provider == "Ollama":
        return list_ollama_models(base_url)
    if not api_key.strip():
        raise ValueError(f"Add your {provider} API key first.")

    if provider == "OpenAI":
        response = requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        response.raise_for_status()
        ids = [item.get("id", "") for item in response.json().get("data", [])]
        excluded = ("embedding", "embed", "whisper", "transcribe", "tts", "audio", "moderation", "dall-e", "image")
        return sorted({name for name in ids if name and not any(word in name.lower() for word in excluded)})

    if provider == "Anthropic":
        response = requests.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            params={"limit": 1000}, timeout=20,
        )
        response.raise_for_status()
        return sorted({item.get("id", "") for item in response.json().get("data", []) if item.get("id")})

    if provider == "Gemini":
        response = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": api_key}, params={"pageSize": 1000}, timeout=20,
        )
        response.raise_for_status()
        return sorted({
            item.get("name", "").removeprefix("models/")
            for item in response.json().get("models", [])
            if "generateContent" in item.get("supportedGenerationMethods", []) and item.get("name")
        })

    raise ValueError(f"Unsupported provider: {provider}")


def list_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Read installed model names from the local Ollama /api/tags endpoint."""
    response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=4)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return sorted({str(item.get("name")) for item in payload.get("models", []) if item.get("name")})


def complete(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str = "",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.2,
    max_tokens: int = 3500,
    browser_search: bool = False,
    search_domains: list[str] | None = None,
) -> str:
    """Generate a non-streaming completion using the selected provider."""
    if provider == "Groq":
        if not api_key.strip():
            raise ValueError("Add your Groq API key in the sidebar.")
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if browser_search:
            kwargs["tools"] = [{"type": "browser_search"}]
            kwargs["tool_choice"] = "required"
            if search_domains:
                kwargs["search_settings"] = {"include_domains": search_domains}
            kwargs["reasoning_effort"] = "low"
        response = Groq(api_key=api_key).chat.completions.create(**kwargs)
        return response.choices[0].message.content or "The model returned an empty response."

    if provider == "Ollama":
        if not model:
            raise ValueError("No local Ollama model is selected. Start Ollama and refresh the model list.")
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/api/chat",
                json={"model": model, "messages": messages, "stream": False,
                      "options": {"temperature": temperature, "num_predict": max_tokens}},
                timeout=600,
            )
            if response.status_code == 404:
                raise ValueError(f"Ollama could not find model '{model}'. Refresh the model list and select an installed model.")
            response.raise_for_status()
            return str(response.json().get("message", {}).get("content", "")) or "The model returned an empty response."
        except requests.ConnectionError as exc:
            raise ConnectionError(f"Could not connect to Ollama at {base_url}. Make sure Ollama is running on this machine.") from exc
        except requests.Timeout as exc:
            raise TimeoutError("Ollama took too long to respond. Try a smaller local model or a shorter prompt.") from exc

    if provider == "OpenAI":
        if not api_key.strip():
            raise ValueError("Add your OpenAI API key in the sidebar.")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "max_completion_tokens": max_tokens},
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"].get("content", "") or "The model returned an empty response."

    if provider == "Anthropic":
        if not api_key.strip():
            raise ValueError("Add your Anthropic API key in the sidebar.")
        system_text = "\n\n".join(item["content"] for item in messages if item.get("role") == "system")
        conversation = [{"role": item["role"], "content": item["content"]} for item in messages if item.get("role") != "system"]
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            json={"model": model, "system": system_text, "messages": conversation, "temperature": temperature, "max_tokens": max_tokens},
            timeout=180,
        )
        response.raise_for_status()
        return "\n".join(part.get("text", "") for part in response.json().get("content", []) if part.get("type") == "text") or "The model returned an empty response."

    if provider == "Gemini":
        if not api_key.strip():
            raise ValueError("Add your Gemini API key in the sidebar.")
        system_text = "\n\n".join(item["content"] for item in messages if item.get("role") == "system")
        contents = [{
            "role": "model" if item.get("role") == "assistant" else "user",
            "parts": [{"text": item["content"]}],
        } for item in messages if item.get("role") != "system"]
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload, timeout=180,
        )
        response.raise_for_status()
        parts = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "\n".join(part.get("text", "") for part in parts if part.get("text")) or "The model returned an empty response."

    raise ValueError("Choose a supported AI provider.")
