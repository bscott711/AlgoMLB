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


def _make_openrouter_request(prompt: str, attempt: int) -> requests.Response:
    """Helper to make the POST request to the OpenRouter LLM API."""
    payload = {
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "messages": [{"role": "user", "content": prompt}],
    }
    return requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=get_openrouter_headers(),
        json=payload,
        timeout=180,
    )


def get_ai_text(prompt: str, retries: int = 3) -> str | None:
    """Generates text from the OpenRouter LLM API with retries."""
    retry_delays = [60, 180, 300]

    for attempt in range(retries):
        try:
            response = _make_openrouter_request(prompt, attempt)

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

            print(
                f"   ⚠️ API Error (Attempt {attempt + 1}): "
                f"{response.status_code} - {response.text[:50]}"
            )

        except Exception as e:
            print(f"   ⚠️ Connection Failed (Attempt {attempt + 1}): {e}")

        if attempt < retries - 1:
            wait = retry_delays[attempt]
            print(f"   ⏳ Waiting {wait}s before retry...")
            time.sleep(wait)

    return None


def get_ai_json(prompt: str, retries: int = 3) -> dict[str, Any] | None:
    """Generates JSON from the OpenRouter LLM API with retries."""
    retry_delays = [60, 180, 300]

    for attempt in range(retries):
        try:
            response = _make_openrouter_request(prompt, attempt)

            if response.status_code == 200:
                data = response.json()
                text = data["choices"][0]["message"]["content"].strip()

                # Strip markdown JSON block ticks if the LLM added them
                text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
                text = re.sub(r"\s*```$", "", text)

                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    print(
                        f"   ⚠️ Failed to parse JSON on Attempt {attempt + 1}: "
                        f"{text[:50]}..."
                    )
            else:
                print(f"   ⚠️ API Error (Attempt {attempt + 1}): {response.status_code}")

        except Exception as e:
            print(f"   ⚠️ Connection Failed (Attempt {attempt + 1}): {e}")

        if attempt < retries - 1:
            time.sleep(retry_delays[attempt])

    return None
