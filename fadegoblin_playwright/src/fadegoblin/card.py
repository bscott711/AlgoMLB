"""Renders a super-compact, 2-column grid bet card overlay."""

import math
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from fadegoblin import config

# ── Layout constants ──────────────────────────────────────────────────
CARD_WIDTH = 560
PADDING = 15
ROW_HEIGHT = 52  # taller to fit two-line rows (matchup + diagnostics)
HEADER_HEIGHT = 44
FOOTER_HEIGHT = 28
COL_WIDTH = (CARD_WIDTH - (PADDING * 3)) // 2

# ── Colour palette ────────────────────────────────────────────────────
BG_COLOR = (12, 12, 18, 190)
POTD_BG = (15, 65, 35, 255)
ACCENT_GREEN = (0, 255, 100)
POTD_GOLD = (255, 215, 0)
HIGH_CONFIDENCE_BLUE = (0, 191, 255)
TEXT_WHITE = (240, 240, 245)
TEXT_DIM = (160, 160, 175)

# ── Font helpers ──────────────────────────────────────────────────────
_FONT_PATH = Path(__file__).parent / "assets" / "Inter.ttf"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(_FONT_PATH), size)
    except (OSError, IOError):
        return ImageFont.load_default()


SLIP_TITLES = [
    "💚 THE GOBLIN'S MANIC SLIP",
    "💚 DEGEN ALLERGY THERAPY",
    "💚 THE COLD SHOWER TICKET",
    "💚 HIGH CONVICTION GARBAGE",
    "💚 THE ALGO'S PARANOIA SLIP",
    "💚 PURE UNADULTERATED COPE",
    "💚 THE BANKROLL EXORCISM",
    "💚 CHAOTIC NEUTRAL RATION",
    "💚 100% ORGANIC HORSEPLAY",
    "💚 THE SHARP MONEY HALLUCINATION",
    "💚 THE BASE-7 PROPHECY SLIP",
    "💚 RECKLESS SPECULATION SLIP",
    "💚 FINANCIAL RUIN SPEEDRUN",
    "💚 THE GOB'S SHAMANIC SLIP",
    "💚 DUGOUT WATER CONDENSATION",
]


def render_bet_card(
    legs: list[dict],
    potd_index: int,
    background_path: Path | None = None,
    title: str | None = None,
) -> Path:
    if not title:
        import random

        day_seed = datetime.now().strftime("%Y-%m-%d")
        rng = random.Random(day_seed)
        title = rng.choice(SLIP_TITLES)

    num_legs = len(legs)
    if num_legs == 0:
        return Path(__file__).parent / "assets" / "empty_card.png"

    if potd_index < 0 or potd_index >= num_legs:
        potd_index = 0

    if num_legs == 1:
        header_h = 16
        num_rows = 1
    else:
        header_h = HEADER_HEIGHT
        num_rows = math.ceil((num_legs - 1) / 2) + 1

    # Grid height calculation
    table_content_h = num_rows * (ROW_HEIGHT + 4)
    base_table_height = header_h + table_content_h + FOOTER_HEIGHT + 10

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
    if background_path and background_path.exists():
        table_y2 = card_height - 40
        table_y1 = table_y2 - base_table_height
    else:
        table_y2 = card_height
        table_y1 = 0

    # Main rounded background
    if num_legs > 1:
        draw.rounded_rectangle(
            [table_x1, table_y1, table_x2, table_y2], radius=14, fill=BG_COLOR
        )

    font_title = _load_font(16)
    font_date = _load_font(10)
    font_row = _load_font(13)
    font_footer = _load_font(9)

    # ── Header (Only if num_legs > 1) ─────────────────────────────────
    if num_legs > 1:
        draw.text(
            (table_x1 + PADDING, table_y1 + 10),
            title,
            fill=ACCENT_GREEN,
            font=font_title,
        )
        date_str = datetime.now().strftime("%b %d, %Y")
        draw.text(
            (table_x1 + PADDING, table_y1 + 28), date_str, fill=TEXT_DIM, font=font_date
        )

        # ── Squad Signals (Top Right) ─────────────────────────────────
        all_unique_badges = []
        for leg in legs:
            for b in leg.get("badges", []):
                if b not in all_unique_badges:
                    all_unique_badges.append(b)

        if all_unique_badges:
            has_potd = any("POTD" in b for b in all_unique_badges)
            has_hc = any("HIGH CONFIDENCE" in b for b in all_unique_badges)

            legend_x = table_x2 - PADDING
            legend_y = table_y1 + 18

            if has_hc:
                text = "HIGH CONF"
                bbox = draw.textbbox((0, 0), text, font=font_date)
                text_w = bbox[2] - bbox[0]
                legend_x -= text_w
                draw.text((legend_x, legend_y), text, fill=TEXT_DIM, font=font_date)
                legend_x -= 8
                draw.rounded_rectangle([legend_x - 4, legend_y, legend_x, legend_y + 10], radius=1, fill=HIGH_CONFIDENCE_BLUE)
                legend_x -= 16

            if has_potd:
                text = "POTD"
                bbox = draw.textbbox((0, 0), text, font=font_date)
                text_w = bbox[2] - bbox[0]
                legend_x -= text_w
                draw.text((legend_x, legend_y), text, fill=TEXT_DIM, font=font_date)
                legend_x -= 8
                draw.rounded_rectangle([legend_x - 4, legend_y, legend_x, legend_y + 10], radius=1, fill=POTD_GOLD)

    # ── Rendering helper for a single leg box ─────────────────────────
    def draw_leg_box(leg: dict, lx: int, ly: int, lwidth: int, is_potd: bool):
        fill = POTD_BG if is_potd else (30, 30, 45, 180)
        draw.rounded_rectangle(
            [lx, ly, lx + lwidth, ly + ROW_HEIGHT], radius=6, fill=fill
        )

        is_high_confidence = "💎 HIGH CONFIDENCE" in leg.get("badges", [])

        if is_potd and is_high_confidence:
            draw.rounded_rectangle(
                [lx, ly, lx + 4, ly + ROW_HEIGHT // 2 + 2], radius=2, fill=POTD_GOLD
            )
            draw.rounded_rectangle(
                [lx, ly + ROW_HEIGHT // 2 - 2, lx + 4, ly + ROW_HEIGHT], radius=2, fill=HIGH_CONFIDENCE_BLUE
            )
        elif is_potd:
            draw.rounded_rectangle(
                [lx, ly, lx + 4, ly + ROW_HEIGHT], radius=2, fill=POTD_GOLD
            )
        elif is_high_confidence:
            draw.rounded_rectangle(
                [lx, ly, lx + 4, ly + ROW_HEIGHT], radius=2, fill=HIGH_CONFIDENCE_BLUE
            )

        # ── Matchup row (line 1) ───────────────────────────────────
        game_text = leg["game"]
        pick_text = leg["pick"]
        parts = game_text.split(" @ ")

        matchup_y = ly + 6  # top line: matchup
        diag_y = ly + 28  # bottom line: diagnostics

        if len(parts) == 2:
            away, home = parts
            away_color = (
                ACCENT_GREEN
                if pick_text == away
                else (TEXT_WHITE if is_potd else TEXT_DIM)
            )
            home_color = (
                ACCENT_GREEN
                if pick_text == home
                else (TEXT_WHITE if is_potd else TEXT_DIM)
            )

            draw.text((lx + 10, matchup_y), away, fill=away_color, font=font_row)
            bbox = draw.textbbox((0, 0), away, font=font_row)
            at_x = lx + 10 + (bbox[2] - bbox[0]) + 3
            draw.text((at_x, matchup_y), "@", fill=TEXT_DIM, font=font_row)
            bbox_at = draw.textbbox((0, 0), "@", font=font_row)
            home_x = at_x + (bbox_at[2] - bbox_at[0]) + 3
            draw.text((home_x, matchup_y), home, fill=home_color, font=font_row)
        else:
            draw.text((lx + 10, matchup_y), game_text, fill=TEXT_WHITE, font=font_row)

        # ── Odds aligned right (line 1) ────────────────────────────
        odds_str = str(leg["odds"])
        bbox_odds = draw.textbbox((0, 0), odds_str, font=font_row)
        odds_x = lx + lwidth - (bbox_odds[2] - bbox_odds[0]) - 8
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
        draw.text((lx + 10, diag_y), diag_text, fill=TEXT_DIM, font=font_date)

    # ── Grid Rendering ────────────────────────────────────────────────
    start_y = table_y1 + header_h

    if num_legs == 1:
        # Graffiti style for single POTD / Sniper
        potd_leg = legs[potd_index]
        
        def draw_graffiti_text(x, y, text, font, fill_color, glow_color):
            draw.text((x, y), text, font=font, fill=glow_color, stroke_width=8, stroke_fill=glow_color)
            draw.text((x, y), text, font=font, fill=fill_color, stroke_width=4, stroke_fill="black")

        _GRAFFITI_FONT_PATH = Path(__file__).parent / "assets" / "elfont-block.ttf"
        try:
            font_main = ImageFont.truetype(str(_GRAFFITI_FONT_PATH), 100)
            font_odds = ImageFont.truetype(str(_GRAFFITI_FONT_PATH), 76)
            font_sub = _load_font(28)
        except (OSError, IOError):
            font_main = _load_font(90)
            font_odds = _load_font(76)
            font_sub = _load_font(28)
        
        cy = card_height - 265
        
        game_text = potd_leg["game"]
        pick_text = potd_leg["pick"]
        parts = game_text.split(" @ ")
        
        if len(parts) == 2:
            away, home = parts
            away_color = (0, 255, 100) if pick_text == away else "white"
            home_color = (0, 255, 100) if pick_text == home else "white"
            
            # Dynamically shrink font if too wide
            main_font_size = 100
            font_main = ImageFont.truetype(str(_GRAFFITI_FONT_PATH), main_font_size)
            at_font_size = int(main_font_size * 0.55)
            font_at = _load_font(at_font_size)
            at_pad = 16  # horizontal padding on each side of @
            while True:
                at_font_size = int(main_font_size * 0.55)
                font_at = _load_font(at_font_size)
                bbox1 = draw.textbbox((0, 0), away, font=font_main)
                w1 = bbox1[2] - bbox1[0]
                bbox2 = draw.textbbox((0, 0), "@", font=font_at)
                w2 = bbox2[2] - bbox2[0]
                bbox3 = draw.textbbox((0, 0), home, font=font_main)
                w3 = bbox3[2] - bbox3[0]
                
                total_w = w1 + at_pad + w2 + at_pad + w3
                if total_w < canvas_width - 60 or main_font_size <= 30:
                    break
                main_font_size -= 4
                font_main = ImageFont.truetype(str(_GRAFFITI_FONT_PATH), main_font_size)
            
            start_x = (canvas_width - total_w) / 2
            
            # Position @ at visual center of graffiti glyphs.
            # elfont-block renders glyphs with visual mass around 40-55% of
            # the font size.  Use font-size proportions instead of textbbox
            # (unreliable across mismatched font families).
            name_visual_mid = cy + main_font_size * 0.42
            at_visual_h = at_font_size * 0.75
            at_y = int(name_visual_mid - at_visual_h / 2)
            
            draw_graffiti_text(start_x, cy, away, font_main, away_color, (0, 255, 255))
            draw_graffiti_text(start_x + w1 + at_pad, at_y, "@", font_at, "white", (0, 255, 255))
            draw_graffiti_text(start_x + w1 + at_pad + w2 + at_pad, cy, home, font_main, home_color, (0, 255, 255))
            cy += 135
        else:
            main_font_size = 100
            font_main = ImageFont.truetype(str(_GRAFFITI_FONT_PATH), main_font_size)
            bbox = draw.textbbox((0, 0), game_text, font=font_main)
            w = bbox[2] - bbox[0]
            while w > canvas_width - 60 and main_font_size > 30:
                main_font_size -= 4
                font_main = ImageFont.truetype(str(_GRAFFITI_FONT_PATH), main_font_size)
                bbox = draw.textbbox((0, 0), game_text, font=font_main)
                w = bbox[2] - bbox[0]
                
            draw_graffiti_text((canvas_width - w)/2, cy, game_text, font_main, (0, 255, 100), (0, 255, 255))
            cy += 135
        
        odds_str = str(potd_leg["odds"])
        if "🍭" in odds_str and "(" in odds_str:
            start_idx = odds_str.find("(") + 1
            end_idx = odds_str.find(" 🍭")
            if end_idx == -1: end_idx = odds_str.find("🍭")
            odds_str = odds_str[start_idx:end_idx].strip()
        else:
            odds_str = odds_str.replace(" 🍭", "")
        
        # Check for + or - sign
        sign = ""
        num_part = odds_str
        if odds_str.startswith("+"):
            sign = "+"
            num_part = odds_str[1:]
        elif odds_str.startswith("-") or odds_str.startswith("–") or odds_str.startswith("—"):
            sign = "-"
            num_part = odds_str[1:]
            
        odds_font_size = 76
        # Sign drawn as a thick geometric shape; allocate fixed width
        sign_shape_w = int(odds_font_size * 0.55)
        sign_gap = 10
        
        while True:
            font_odds = ImageFont.truetype(str(_GRAFFITI_FONT_PATH), odds_font_size)
            sign_shape_w = int(odds_font_size * 0.55)
            
            bbox_num = draw.textbbox((0, 0), num_part, font=font_odds)
            w_num = bbox_num[2] - bbox_num[0]
            
            if sign:
                total_odds_w = sign_shape_w + sign_gap + w_num
            else:
                total_odds_w = w_num
                
            if total_odds_w < canvas_width - 80 or odds_font_size <= 24:
                break
            odds_font_size -= 4
            
        odds_x = (canvas_width - total_odds_w) / 2
        
        if sign:
            num_x = odds_x + sign_shape_w + sign_gap
            # Draw the number
            draw_graffiti_text(num_x, cy, num_part, font_odds, (255, 255, 100), (255, 0, 255))
            
            # Draw sign as a thick geometric shape with graffiti glow
            num_visual_mid_y = int(cy + odds_font_size * 0.45)
            bar_thickness = max(8, odds_font_size // 8)
            bar_width = sign_shape_w - 8
            bar_x1 = int(odds_x + 4)
            bar_x2 = int(bar_x1 + bar_width)
            bar_y1 = int(num_visual_mid_y - bar_thickness // 2)
            bar_y2 = int(bar_y1 + bar_thickness)
            
            # Glow layer
            glow = 6
            draw.rounded_rectangle(
                [bar_x1 - glow, bar_y1 - glow, bar_x2 + glow, bar_y2 + glow],
                radius=4, fill=(255, 0, 255)
            )
            # Black outline
            outline = 3
            draw.rounded_rectangle(
                [bar_x1 - outline, bar_y1 - outline, bar_x2 + outline, bar_y2 + outline],
                radius=3, fill="black"
            )
            # Main bar (horizontal — serves as minus)
            draw.rounded_rectangle(
                [bar_x1, bar_y1, bar_x2, bar_y2],
                radius=2, fill=(255, 255, 100)
            )
            
            if sign == "+":
                # Add vertical bar for plus
                vert_x1 = int(bar_x1 + bar_width // 2 - bar_thickness // 2)
                vert_x2 = int(vert_x1 + bar_thickness)
                vert_y1 = int(num_visual_mid_y - bar_width // 2)
                vert_y2 = int(num_visual_mid_y + bar_width // 2)
                draw.rounded_rectangle(
                    [vert_x1 - glow, vert_y1 - glow, vert_x2 + glow, vert_y2 + glow],
                    radius=4, fill=(255, 0, 255)
                )
                draw.rounded_rectangle(
                    [vert_x1 - outline, vert_y1 - outline, vert_x2 + outline, vert_y2 + outline],
                    radius=3, fill="black"
                )
                draw.rounded_rectangle(
                    [vert_x1, vert_y1, vert_x2, vert_y2],
                    radius=2, fill=(255, 255, 100)
                )
                # Redraw horizontal on top so intersection is clean
                draw.rounded_rectangle(
                    [bar_x1, bar_y1, bar_x2, bar_y2],
                    radius=2, fill=(255, 255, 100)
                )
        else:
            draw_graffiti_text(odds_x, cy, odds_str, font_odds, (255, 255, 100), (255, 0, 255))
        cy += 105
        
        implied_pct = potd_leg.get("implied")
        edge_pct = potd_leg.get("edge")
        diag_parts = []
        if implied_pct is not None:
            diag_parts.append(f"impl {implied_pct}%")
        if edge_pct is not None:
            diag_parts.append(f"+{edge_pct}% EV")
        if diag_parts:
            diag_text = "  •  ".join(diag_parts)
            bbox = draw.textbbox((0, 0), diag_text, font=font_sub)
            w = bbox[2] - bbox[0]
            draw_graffiti_text((canvas_width - w)/2, cy, diag_text, font_sub, "white", "black")
    else:
        # 1. Render the centered POTD leg at row 0 (same width as other columns)
        potd_leg = legs[potd_index]
        potd_x = table_x1 + (CARD_WIDTH - COL_WIDTH) // 2
        draw_leg_box(potd_leg, potd_x, start_y, COL_WIDTH, is_potd=True)

        # 2. Render other legs in a 2-column grid starting below the POTD
        other_legs = [leg for idx, leg in enumerate(legs) if idx != potd_index]
        for i, leg in enumerate(other_legs):
            row = i // 2 + 1  # start at row 1
            col = i % 2

            lx = table_x1 + PADDING + col * (COL_WIDTH + PADDING)
            ly = start_y + row * (ROW_HEIGHT + 4)

            draw_leg_box(leg, lx, ly, COL_WIDTH, is_potd=False)

    # ── Footer (skip for single-leg graffiti cards) ────────────────────
    if num_legs > 1:
        footer_y = table_y2 - 20
        draw.text(
            (table_x1 + PADDING, footer_y),
            "@TheFadeGoblin  •  AlgoMLB",
            fill=TEXT_DIM,
            font=font_footer,
        )

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
    if background_path and background_path.exists():
        table_y2 = card_height - 40
        table_y1 = table_y2 - base_table_height
    else:
        table_y2 = card_height
        table_y1 = 0

    draw.rounded_rectangle(
        [table_x1, table_y1, table_x2, table_y2], radius=14, fill=BG_COLOR
    )

    font_title = _load_font(16)
    font_date = _load_font(10)
    font_row = _load_font(13)
    font_footer = _load_font(9)
    font_record = _load_font(20)

    # ── Header ────────────────────────────────────────────────────────
    date_str = stats.get("date", datetime.now().strftime("%Y-%m-%d"))
    title_str = "👺 WEEKLY RECAP" if "to" in date_str else "👺 GOBLIN RECAP"
    draw.text(
        (table_x1 + PADDING, table_y1 + 10),
        title_str,
        fill=ACCENT_GREEN,
        font=font_title,
    )
    draw.text(
        (table_x1 + PADDING, table_y1 + 28), date_str, fill=TEXT_DIM, font=font_date
    )

    # ── Record (top right) ────────────────────────────────────────────
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    pushes = stats.get("pushes", 0)
    record_str = f"{wins}W-{losses}L" + (f"-{pushes}P" if pushes else "")
    record_color = (
        ACCENT_GREEN if wins > losses else (255, 80, 80) if losses > wins else TEXT_DIM
    )
    bbox_rec = draw.textbbox((0, 0), record_str, font=font_record)
    draw.text(
        (table_x2 - PADDING - (bbox_rec[2] - bbox_rec[0]), table_y1 + 10),
        record_str,
        fill=record_color,
        font=font_record,
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
        draw.rounded_rectangle(
            [x, y, x + COL_WIDTH, y + ROW_HEIGHT], radius=6, fill=row_fill
        )

        result_color = RESULT_COLORS.get(result, TEXT_DIM)
        result_icon = {"WIN": "✓", "LOSS": "✗", "PUSH": "–", "?": "?"}.get(result, "?")

        matchup_y = y + 6
        diag_y = y + 28

        draw.text(
            (x + 10, matchup_y),
            f"{pick['pick']} {pick['odds']}",
            fill=TEXT_WHITE,
            font=font_row,
        )
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
        fill=TEXT_DIM,
        font=font_footer,
    )

    output_path = config.BASE_DIR / "temp_recap_card.png"
    img.save(str(output_path), "PNG")
    return output_path
