import json
import re
import time
from typing import Any

import requests

from fadegoblin import config


def get_auth_headers() -> dict[str, str]:
    """Generate authentication headers for Pollinations API (preserved for image.py)."""
    headers = {"Content-Type": "application/json"}
    if config.POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {config.POLLINATIONS_API_KEY}"
    return headers


def get_openrouter_headers() -> dict[str, str]:
    """Generate authentication headers for OpenRouter API."""
    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/bscott711/fadegoblin",
        "X-Title": "FadeGoblin",
    }
    if config.OPENROUTER_API_KEY:
        headers["Authorization"] = f"Bearer {config.OPENROUTER_API_KEY}"
    return headers


FREE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-26b-a4b-it:free",
]

def _make_openrouter_request(prompt: str, model: str) -> requests.Response:
    """Helper to make the POST request to the OpenRouter LLM API."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    return requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=get_openrouter_headers(),
        json=payload,
        timeout=240,
    )


def get_ai_text(prompt: str, retries: int = 3) -> str:
    """Generates text from the OpenRouter LLM API with retries and multi-model fallback."""
    retry_delays = [60, 180, 300]
    last_error = "Unknown error"

    for attempt in range(retries):
        for model in FREE_MODELS:
            try:
                response = _make_openrouter_request(prompt, model)

                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"].strip()

                    if "reasoning_content" in text or "role:assistant" in text:
                        clean_match = re.search(r'(["\'])(?:(?=(\\?))\2.)*?\1$', text)
                        text = clean_match.group(0) if clean_match else text

                    text = re.sub(
                        r"^(statement|quote|tweet|text|answer|response|pick)\s*:\s*",
                        "",
                        text,
                        flags=re.IGNORECASE,
                    )
                    text = re.sub(r'^["\'{] | ["\'}]$', "", text)
                    if text.startswith("{") and text.endswith("}"):
                        text = text[1:-1]
                    text = re.sub(r"\s*[:({]\s*[\d.]*\s*[})]?\s*$", "", text)
                    text = text.replace('"', "").replace("\n", " ").strip()

                    if len(text) < 10 or "Statement:" in text:
                        raise ValueError("Generated text was too short or malformed")
                    return text

                error_msg = f"{response.status_code} - {response.text[:50]}"
                print(f"   ⚠️ API Error with {model} (Attempt {attempt + 1}): {error_msg}")
                last_error = f"API Error: {error_msg}"

            except Exception as e:
                error_msg = str(e)
                print(f"   ⚠️ Connection Failed with {model} (Attempt {attempt + 1}): {error_msg}")
                last_error = f"Connection Failed: {error_msg}"

        if attempt < retries - 1:
            wait = retry_delays[attempt]
            print(f"   ⏳ All free models failed. Waiting {wait}s before retry...")
            time.sleep(wait)

    raise Exception(f"LLM text generation failed after {retries} retries. Last error: {last_error}")


def get_ai_json(prompt: str, retries: int = 3) -> dict[str, Any]:
    """Generates JSON from the OpenRouter LLM API with retries and multi-model fallback."""
    retry_delays = [60, 180, 300]
    last_error = "Unknown error"

    for attempt in range(retries):
        for model in FREE_MODELS:
            try:
                response = _make_openrouter_request(prompt, model)

                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"].strip()

                    # Strip markdown JSON block ticks if the LLM added them
                    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
                    text = re.sub(r"\s*```$", "", text)

                    try:
                        return json.loads(text)
                    except json.JSONDecodeError as e:
                        error_msg = f"Failed to parse JSON: {e}. Text: {text[:50]}..."
                        print(f"   ⚠️ {error_msg} from {model} on Attempt {attempt + 1}")
                        last_error = error_msg
                else:
                    error_msg = f"{response.status_code} - {response.text[:50]}"
                    print(f"   ⚠️ API Error with {model} (Attempt {attempt + 1}): {error_msg}")
                    last_error = f"API Error: {error_msg}"

            except Exception as e:
                error_msg = str(e)
                print(f"   ⚠️ Connection Failed with {model} (Attempt {attempt + 1}): {error_msg}")
                last_error = f"Connection Failed: {error_msg}"

        if attempt < retries - 1:
            wait = retry_delays[attempt]
            print(f"   ⏳ All free models failed. Waiting {wait}s before retry...")
            time.sleep(wait)

    raise Exception(f"LLM JSON generation failed after {retries} retries. Last error: {last_error}")
