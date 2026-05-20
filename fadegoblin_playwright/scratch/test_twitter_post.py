import sys
import os
from pathlib import Path

# Add project src to path
project_root = Path("/home/opc/AlgoMLB")
sys.path.insert(0, str(project_root / "fadegoblin_playwright" / "src"))

# Load dotenv
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from fadegoblin.browser_twitter import post_to_twitter_browser

# Safety dry-run check or simple tweet drafting
tweet_text = "🤖 FadeGoblin visual & element selector posting validation test. Hello world, the sportsbooks are ours! 📈💸 #POTD"

print("--- TESTING TWITTER POST AUTOMATION WITH ROBUST SELECTORS ---")
print(f"Drafting tweet:\n\"{tweet_text}\"\n")

try:
    tweet_id = post_to_twitter_browser(tweet_text)
    if tweet_id:
        print(f"✅ Success! Tweet posted successfully! Tweet ID: {tweet_id}")
    else:
        print("❌ Posting did not return a tweet ID (check error log/instructions).")
except Exception as e:
    print(f"❌ Error executing Twitter post: {e}")
