import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Directories ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# --- Database ---
# Supports both DATABASE_URL (Standard) and DATABASE__URL (AlgoMLB compat)
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("DATABASE__URL")

# --- Bluesky Credentials ---
BOT_HANDLE = os.getenv("BOT_HANDLE")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# --- Twitter/X Credentials (Browser Automation) ---
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
TWITTER_PASSWORD = os.getenv("TWITTER_PASSWORD")

# --- LLM / Image Generation ---
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_AUTH") or os.getenv("POLLINATIONS_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")


# --- Tuning ---
TEXT_ONLY_ODDS = 0.05  # 5% chance of no image

def validate_config():
    """Ensures critical environment variables are set."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL must be set.")
    
    # We require at least one social target
    bsky_ready = all([BOT_HANDLE, APP_PASSWORD])
    twitter_ready = all([TWITTER_USERNAME, TWITTER_PASSWORD])
    
    if not bsky_ready and not twitter_ready:
        raise ValueError("At least one set of social credentials (Bluesky or Twitter) must be set.")
    
    if bsky_ready:
        print(f"✅ Bluesky configured as @{BOT_HANDLE}")
    if twitter_ready:
        print(f"✅ Twitter/X browser automation configured for @{TWITTER_USERNAME}")
