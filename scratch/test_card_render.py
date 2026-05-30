import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "fadegoblin_playwright" / "src"))

from fadegoblin.card import render_bet_card

potd_leg = {
    "id": "mock-id-123",
    "game": "SDP @ WSH",
    "pick": "WSH",
    "odds": "+110",
    "edge": 8.7,
    "implied": 48.3,
    "goblins": "👺",
    "badges": ["🎯 POTD"]
}

bg_path = Path("/home/opc/AlgoMLB/fadegoblin_playwright/temp_meme_1.jpg")
if not bg_path.exists():
    bg_path = None

print("Rendering bet card...")
output = render_bet_card([potd_leg], potd_index=0, background_path=bg_path)
print(f"Success! Output saved to: {output}")

# Copy the output to the artifacts directory as temp_card.png so we can view it
import shutil
dest = Path("/home/opc/.gemini/antigravity-ide/brain/715fad65-eb18-46cf-8edd-7ca92918f5a1/temp_card.png")
shutil.copy(output, dest)
print(f"Copied to artifacts: {dest}")
