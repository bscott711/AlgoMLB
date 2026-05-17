"""Renders a super-compact, 2-column grid bet card overlay."""

import math
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from fadegoblin import config

# ── Layout constants ──────────────────────────────────────────────────
CARD_WIDTH = 560
PADDING = 15
ROW_HEIGHT = 52          # taller to fit two-line rows (matchup + diagnostics)
HEADER_HEIGHT = 44
FOOTER_HEIGHT = 28
COL_WIDTH = (CARD_WIDTH - (PADDING * 3)) // 2

# ── Colour palette ────────────────────────────────────────────────────
BG_COLOR = (12, 12, 18, 190)
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

    # ── Squad Signals (Top Right) ─────────────────────────────────────
    all_unique_badges = []
    for leg in legs:
        for b in leg.get("badges", []):
            if b not in all_unique_badges:
                all_unique_badges.append(b)
    
    if all_unique_badges:
        signal_text = " • ".join(all_unique_badges)
        bbox_sig = draw.textbbox((0, 0), signal_text, font=font_date)
        draw.text((table_x2 - PADDING - (bbox_sig[2]-bbox_sig[0]), table_y1 + 15), signal_text, fill=ACCENT_GREEN, font=font_date)

    # ── Grid Rendering ────────────────────────────────────────────────
    start_y = table_y1 + HEADER_HEIGHT
    
    for i, leg in enumerate(legs):
        row = i // 2
        col = i % 2
        
        x = table_x1 + PADDING + col * (COL_WIDTH + PADDING)
        y = start_y + row * (ROW_HEIGHT + 4)
        
        is_potd = i == potd_index
        badges = leg.get("badges", [])
        
        # Row box
        fill = POTD_BG if is_potd else (30, 30, 45, 180)
        draw.rounded_rectangle([x, y, x + COL_WIDTH, y + ROW_HEIGHT], radius=6, fill=fill)
        
        if is_potd:
            draw.rounded_rectangle([x, y, x + 4, y + ROW_HEIGHT], radius=2, fill=ACCENT_GREEN)

        # ── Matchup row (line 1) ───────────────────────────────────
        game_text = leg["game"]
        pick_text = leg["pick"]
        parts = game_text.split(" @ ")

        matchup_y = y + 6  # top line: matchup
        diag_y = y + 28    # bottom line: diagnostics

        if len(parts) == 2:
            away, home = parts
            away_color = ACCENT_GREEN if pick_text == away else (TEXT_WHITE if is_potd else TEXT_DIM)
            home_color = ACCENT_GREEN if pick_text == home else (TEXT_WHITE if is_potd else TEXT_DIM)

            draw.text((x + 10, matchup_y), away, fill=away_color, font=font_row)
            bbox = draw.textbbox((0, 0), away, font=font_row)
            at_x = x + 10 + (bbox[2] - bbox[0]) + 3
            draw.text((at_x, matchup_y), "@", fill=TEXT_DIM, font=font_row)
            bbox_at = draw.textbbox((0, 0), "@", font=font_row)
            home_x = at_x + (bbox_at[2] - bbox_at[0]) + 3
            draw.text((home_x, matchup_y), home, fill=home_color, font=font_row)
        else:
            draw.text((x + 10, matchup_y), game_text, fill=TEXT_WHITE, font=font_row)

        # ── Odds aligned right (line 1) ────────────────────────────
        odds_str = str(leg["odds"])
        bbox_odds = draw.textbbox((0, 0), odds_str, font=font_row)
        odds_x = x + COL_WIDTH - (bbox_odds[2] - bbox_odds[0]) - 8
        draw.text((odds_x, matchup_y), odds_str, fill=TEXT_WHITE, font=font_row)

        # ── Diagnostics row (line 2): implied % + edge % + rating ─────
        implied_pct = leg.get("implied")
        edge_pct = leg.get("edge")
        goblins = leg.get("goblins", "")
        
        diag_parts = []
        if implied_pct is not None:
            diag_parts.append(f"impl {implied_pct}%")
        if edge_pct is not None:
            diag_parts.append(f"+{edge_pct}% EV")
        if goblins:
            diag_parts.append(goblins)
            
        diag_text = "  •  ".join(diag_parts)
        draw.text((x + 10, diag_y), diag_text, fill=TEXT_DIM, font=font_date)

    # ── Footer ────────────────────────────────────────────────────────
    footer_y = table_y2 - 20
    draw.text((table_x1 + PADDING, footer_y), "@TheFadeGoblin  •  AlgoMLB", fill=TEXT_DIM, font=font_footer)

    output_path = config.BASE_DIR / "temp_card.png"
    img.save(str(output_path), "PNG")
    return output_path


def render_recap_card(stats: dict, background_path: Path | None = None) -> Path:
    """Renders a nightly recap card showing the day's W/L/Push record."""
    picks = stats.get("picks", [])
    num_picks = len(picks)
    num_rows = math.ceil(num_picks / 2) if num_picks > 0 else 1

    table_content_h = num_rows * (ROW_HEIGHT + 4)
    base_table_height = HEADER_HEIGHT + table_content_h + FOOTER_HEIGHT + 20

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

    table_x1 = (canvas_width - CARD_WIDTH) // 2
    table_x2 = table_x1 + CARD_WIDTH
    table_y2 = card_height - 40
    table_y1 = table_y2 - base_table_height

    draw.rounded_rectangle([table_x1, table_y1, table_x2, table_y2], radius=14, fill=BG_COLOR)

    font_title = _load_font(16)
    font_date = _load_font(10)
    font_row = _load_font(13)
    font_footer = _load_font(9)
    font_record = _load_font(20)

    # ── Header ────────────────────────────────────────────────────────
    date_str = stats.get("date", datetime.now().strftime("%Y-%m-%d"))
    draw.text((table_x1 + PADDING, table_y1 + 10), "👺 GOBLIN RECAP", fill=ACCENT_GREEN, font=font_title)
    draw.text((table_x1 + PADDING, table_y1 + 28), date_str, fill=TEXT_DIM, font=font_date)

    # ── Record (top right) ────────────────────────────────────────────
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    pushes = stats.get("pushes", 0)
    record_str = f"{wins}W-{losses}L" + (f"-{pushes}P" if pushes else "")
    record_color = ACCENT_GREEN if wins > losses else (255, 80, 80) if losses > wins else TEXT_DIM
    bbox_rec = draw.textbbox((0, 0), record_str, font=font_record)
    draw.text(
        (table_x2 - PADDING - (bbox_rec[2] - bbox_rec[0]), table_y1 + 10),
        record_str, fill=record_color, font=font_record,
    )

    # ── Pick rows ─────────────────────────────────────────────────────
    RESULT_COLORS = {
        "WIN": (0, 220, 80),
        "LOSS": (220, 60, 60),
        "PUSH": (180, 180, 60),
        "?": TEXT_DIM,
    }

    start_y = table_y1 + HEADER_HEIGHT
    for i, pick in enumerate(picks):
        row_i = i // 2
        col_i = i % 2
        x = table_x1 + PADDING + col_i * (COL_WIDTH + PADDING)
        y = start_y + row_i * (ROW_HEIGHT + 4)

        result = pick.get("result", "?")
        row_fill = {
            "WIN": (15, 65, 35, 200),
            "LOSS": (65, 15, 15, 200),
            "PUSH": (50, 50, 15, 200),
        }.get(result, (30, 30, 45, 180))
        draw.rounded_rectangle([x, y, x + COL_WIDTH, y + ROW_HEIGHT], radius=6, fill=row_fill)

        result_color = RESULT_COLORS.get(result, TEXT_DIM)
        result_icon = {"WIN": "✓", "LOSS": "✗", "PUSH": "–", "?": "?"}.get(result, "?")

        matchup_y = y + 6
        diag_y = y + 28

        draw.text((x + 10, matchup_y), f"{pick['pick']} {pick['odds']}", fill=TEXT_WHITE, font=font_row)
        draw.text((x + 10, diag_y), f"{pick['matchup']}", fill=TEXT_DIM, font=font_date)

        # Result icon aligned right
        bbox_icon = draw.textbbox((0, 0), result_icon, font=font_row)
        icon_x = x + COL_WIDTH - (bbox_icon[2] - bbox_icon[0]) - 8
        draw.text((icon_x, matchup_y), result_icon, fill=result_color, font=font_row)

    # ── Net PnL footer note ───────────────────────────────────────────
    net_pnl = stats.get("net_pnl")
    pnl_str = ""
    if net_pnl is not None:
        sign = "+" if net_pnl >= 0 else ""
        pnl_str = f"  |  Net {sign}{net_pnl:.2f}u"

    footer_y = table_y2 - 20
    draw.text(
        (table_x1 + PADDING, footer_y),
        f"@TheFadeGoblin  •  AlgoMLB{pnl_str}",
        fill=TEXT_DIM, font=font_footer,
    )

    output_path = config.BASE_DIR / "temp_recap_card.png"
    img.save(str(output_path), "PNG")
    return output_path
