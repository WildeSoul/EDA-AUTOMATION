"""
LLM communication layer — Ollama local/cloud + NVIDIA API fallback.
"""
import requests
import json
from config.settings import (
    OLLAMA_BASE_URL, OLLAMA_NUM_CTX, OLLAMA_GPU_LAYERS,
    NVIDIA_API_BASE, TIMEOUT_LOCAL, TIMEOUT_CLOUD, TIMEOUT_NVIDIA_API,
    PREFERRED_ORDER, CLOUD_MODELS, NEMOTRON_MODELS,
)


import logging
logger = logging.getLogger(__name__)


def _query_ollama_raw(model: str, prompt: str, system: str = "", timeout: int = 300) -> str:
    """Send a prompt to Ollama and return full response text without fallback."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "num_gpu": OLLAMA_GPU_LAYERS},
    }
    if system:
        payload["system"] = system
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.ConnectionError:
        return "❌ Could not connect to Ollama. Make sure `ollama serve` is running."
    except requests.HTTPError:
        try:
            err_body = resp.json().get("error", "")
        except Exception:
            err_body = resp.text
        if "memory" in str(err_body).lower():
            return f"❌ Out of memory: {err_body}"
        return f"❌ Ollama HTTP error ({resp.status_code}): {err_body}"
    except Exception as e:
        return f"❌ Ollama error: {e}"


def query_ollama(model: str, prompt: str, system: str = "", timeout: int = 300) -> str:
    """Send a prompt to Ollama with an automatic fallback chain if primary fails."""
    fallback_chain = [
        "deepseek-v4-pro:cloud",
        "gemma4:31b-cloud",
        "qwen3.5:cloud",
        "mistral-nemo:latest",
    ]
    
    models_to_try = [model]
    for fallback_model in fallback_chain:
        if fallback_model not in models_to_try:
            models_to_try.append(fallback_model)
            
    try:
        available_models = get_ollama_models()
    except Exception:
        available_models = []
        
    last_error = ""
    for current_model in models_to_try:
        # Check if fallback model is installed/reachable (skip check for requested model to be safe)
        if (current_model != model and 
            available_models and 
            "(ollama not reachable)" not in available_models and 
            current_model not in available_models):
            continue
            
        if current_model != model:
            logger.warning("Primary model '%s' failed. Falling back to '%s'...", model, current_model)
            
        response = _query_ollama_raw(current_model, prompt, system, timeout)
        
        if not response.strip().startswith("❌"):
            return response
            
        last_error = response
        
    return f"❌ All models in fallback chain failed. Last error: {last_error}"


def stream_ollama(model: str, prompt: str, system: str = ""):
    """Stream tokens from the local Ollama server as a generator."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "num_gpu": OLLAMA_GPU_LAYERS},
    }
    if system:
        payload["system"] = system
    try:
        with requests.post(url, json=payload, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
    except requests.ConnectionError:
        yield "❌ Could not connect to Ollama. Make sure `ollama serve` is running."
    except Exception as e:
        yield f"❌ Ollama error: {e}"


def unload_ollama_model(model: str):
    """Tell Ollama to unload a model from memory to free RAM/VRAM."""
    try:
        requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "keep_alive": 0, "options": {"num_ctx": OLLAMA_NUM_CTX}},
            timeout=10,
        )
    except Exception:
        pass


def get_ollama_models() -> list[str]:
    """Fetch locally installed Ollama models, sorted by preference."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not models:
            return ["(no models found)"]

        def sort_key(name):
            try:
                return PREFERRED_ORDER.index(name)
            except ValueError:
                return len(PREFERRED_ORDER)
        return sorted(models, key=sort_key)
    except Exception:
        return ["(ollama not reachable)"]


def query_nvidia_api(prompt: str, system: str = "", api_key: str = "",
                     model: str = "", timeout: int = TIMEOUT_NVIDIA_API) -> str:
    """Call NVIDIA's integrate API (OpenAI-compatible) — fallback for Nemotron."""
    if not api_key:
        return "❌ NVIDIA API key not set."
    url = f"{NVIDIA_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model or NEMOTRON_MODELS["planner"],
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if not content.strip() and reasoning.strip():
            return reasoning.strip()
        return content
    except requests.ConnectionError:
        return "❌ Could not connect to NVIDIA API."
    except requests.HTTPError:
        if resp.status_code == 401:
            return "❌ Invalid NVIDIA API key."
        return f"❌ NVIDIA API error ({resp.status_code})"
    except Exception as e:
        return f"❌ NVIDIA API error: {e}"
