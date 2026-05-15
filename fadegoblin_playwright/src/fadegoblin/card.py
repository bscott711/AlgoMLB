"""Renders a super-compact, 2-column grid bet card overlay."""

import math
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from fadegoblin import config

# ── Layout constants ──────────────────────────────────────────────────
CARD_WIDTH = 560
PADDING = 15
ROW_HEIGHT = 40
HEADER_HEIGHT = 44
FOOTER_HEIGHT = 28
COL_WIDTH = (CARD_WIDTH - (PADDING * 3)) // 2

# ── Colour palette ────────────────────────────────────────────────────
BG_COLOR = (12, 12, 18, 235)
POTD_BG = (15, 65, 35, 255)
ACCENT_GREEN = (0, 255, 100)
TEXT_WHITE = (240, 240, 245)
TEXT_DIM = (160, 160, 175)

# ── Font helpers ──────────────────────────────────────────────────────
_FONT_PATH = Path(__file__).parent / "assets" / "Inter.ttf"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(_FONT_PATH), size)
    except (OSError, IOError):
        return ImageFont.load_default()


def render_bet_card(legs: list[dict], potd_index: int, background_path: Path | None = None) -> Path:
    num_legs = len(legs)
    num_rows = math.ceil(num_legs / 2)
    
    # Grid height calculation
    table_content_h = num_rows * (ROW_HEIGHT + 4)
    base_table_height = HEADER_HEIGHT + table_content_h + FOOTER_HEIGHT + 10
    
    if background_path and background_path.exists():
        bg_img = Image.open(background_path).convert("RGB")
        target_h = max(1024, base_table_height + 400)
        img = ImageOps.fit(bg_img, (768, target_h), centering=(0.5, 0.4))
        card_height = target_h
        canvas_width = 768
    else:
        img = Image.new("RGB", (CARD_WIDTH, base_table_height), (18, 18, 24))
        card_height = base_table_height
        canvas_width = CARD_WIDTH

    draw = ImageDraw.Draw(img, "RGBA")
    
    # Position table at bottom
    table_x1 = (canvas_width - CARD_WIDTH) // 2
    table_x2 = table_x1 + CARD_WIDTH
    table_y2 = card_height - 40
    table_y1 = table_y2 - base_table_height
    
    # Main rounded background
    draw.rounded_rectangle([table_x1, table_y1, table_x2, table_y2], radius=14, fill=BG_COLOR)
    
    font_title = _load_font(16)
    font_date = _load_font(10)
    font_row = _load_font(13)
    font_footer = _load_font(9)

    # ── Header ────────────────────────────────────────────────────────
    draw.text((table_x1 + PADDING, table_y1 + 10), "💚 THE GOBLIN'S MANIC SLIP", fill=ACCENT_GREEN, font=font_title)
    date_str = datetime.now().strftime("%b %d, %Y")
    draw.text((table_x1 + PADDING, table_y1 + 28), date_str, fill=TEXT_DIM, font=font_date)

    # ── Grid Rendering ────────────────────────────────────────────────
    start_y = table_y1 + HEADER_HEIGHT
    
    for i, leg in enumerate(legs):
        row = i // 2
        col = i % 2
        
        x = table_x1 + PADDING + col * (COL_WIDTH + PADDING)
        y = start_y + row * (ROW_HEIGHT + 4)
        
        is_potd = i == potd_index
        
        # Row box
        fill = POTD_BG if is_potd else (30, 30, 45, 180)
        draw.rounded_rectangle([x, y, x + COL_WIDTH, y + ROW_HEIGHT], radius=6, fill=fill)
        
        if is_potd:
            draw.rounded_rectangle([x, y, x + 4, y + ROW_HEIGHT], radius=2, fill=ACCENT_GREEN)

        text_y = y + (ROW_HEIGHT - 16) // 2
        
        # Matchup
        game_text = leg["game"]
        pick_text = leg["pick"]
        parts = game_text.split(" @ ")
        
        if len(parts) == 2:
            away, home = parts
            away_color = ACCENT_GREEN if pick_text == away else (TEXT_WHITE if is_potd else TEXT_DIM)
            home_color = ACCENT_GREEN if pick_text == home else (TEXT_WHITE if is_potd else TEXT_DIM)
            
            # Draw Away
            draw.text((x + 10, text_y), away, fill=away_color, font=font_row)
            bbox = draw.textbbox((0, 0), away, font=font_row)
            at_x = x + 10 + (bbox[2] - bbox[0]) + 4
            
            # Draw @
            draw.text((at_x, text_y), "@", fill=TEXT_DIM, font=font_row)
            bbox_at = draw.textbbox((0, 0), "@", font=font_row)
            home_x = at_x + (bbox_at[2] - bbox_at[0]) + 4
            
            # Draw Home
            draw.text((home_x, text_y), home, fill=home_color, font=font_row)
        else:
            draw.text((x + 10, text_y), game_text, fill=TEXT_WHITE, font=font_row)
            
        # Odds (aligned to right of column box)
        odds_str = str(leg["odds"])
        bbox_odds = draw.textbbox((0, 0), odds_str, font=font_row)
        odds_x = x + COL_WIDTH - (bbox_odds[2] - bbox_odds[0]) - 10
        draw.text((odds_x, text_y), odds_str, fill=TEXT_WHITE, font=font_row)

    # ── Footer ────────────────────────────────────────────────────────
    footer_y = table_y2 - 20
    draw.text((table_x1 + PADDING, footer_y), "@fadegoblin  •  AlgoMLB", fill=TEXT_DIM, font=font_footer)

    output_path = config.BASE_DIR / "temp_card.png"
    img.save(str(output_path), "PNG")
    return output_path
