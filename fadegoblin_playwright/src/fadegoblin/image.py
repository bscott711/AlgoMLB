import random
import time
import urllib.parse
from pathlib import Path

import requests

from fadegoblin.llm import get_auth_headers
from fadegoblin.prompts import ACTIONS, BACKGROUNDS, CHARACTERS, OUTFITS, STYLES


def generate_goblin_prompt() -> str:
    """Generates a random prompt for the goblin image."""
    selected_char = random.choice(CHARACTERS)
    selected_action = random.choice(ACTIONS)
    selected_outfit = random.choice(OUTFITS)
    selected_style = random.choice(STYLES)
    selected_bg = random.choice(BACKGROUNDS)

    print(
        f"🎲 Image Recipe: {selected_char} {selected_outfit}, {selected_action}, "
        f"{selected_bg} ({selected_style})"
    )

    return (
        f"{selected_char} {selected_outfit} {selected_action} {selected_bg}. "
        f"Crazed, anxious, losing their mind over a sports betting parlay, "
        f"{selected_style}."
    )


def download_goblin_image(prompt: str, output_path: Path) -> Path | None:
    """Downloads an AI-generated image from Pollinations based on the prompt."""
    img_seed = int(time.time()) + random.randint(1, 1000)
    safe_prompt = urllib.parse.quote(prompt)

    image_url = (
        f"https://gen.pollinations.ai/image/{safe_prompt}"
        f"?seed={img_seed}&width=768&height=1024&nologo=true&model=flux"
    )

    for attempt in range(3):
        try:
            print(f"🎨 Downloading image (Attempt {attempt + 1})...")
            response = requests.get(image_url, headers=get_auth_headers(), timeout=45)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)
            return output_path

        except requests.RequestException as e:
            status = getattr(e.response, "status_code", "Unknown")
            print(
                f"   ⚠️ Image Connection Failed (Attempt {attempt + 1}): {status} - {e}"
            )

        if attempt < 2:
            time.sleep(10)

    print("❌ Image generation failed. Proceeding with text only.")
    return None
