import random
from datetime import datetime
from typing import Any

from fadegoblin.llm import get_ai_text
from fadegoblin.prompts import FALLBACK_QUOTES, PERSONAS


def generate_post_content(
    chosen_legs: list[dict[str, Any]], final_odds_str: str
) -> str:
    """Generates the social media post content based on the selected legs."""
    if not chosen_legs:
        return random.choice(FALLBACK_QUOTES)

    # SHRUNK: Extremely compact one-line ticket
    leg_descriptions = [f"{leg['pick']} ({leg['game']})" for leg in chosen_legs]
    locked_bet_text = " + ".join(leg_descriptions)

    if len(chosen_legs) > 1:
        bet_summary = "a parlay on: " + " AND ".join(
            [leg["pick"] for leg in chosen_legs]
        )
    else:
        bet_summary = "a straight bet on: " + chosen_legs[0]["pick"]

    print(f"📈 Locked Ticket: {locked_bet_text} [{final_odds_str}]")

    # --- PERSONA & THEME INCEPTION ---
    current_day = datetime.now().weekday()
    daily_pair = [PERSONAS[current_day * 2], PERSONAS[current_day * 2 + 1]]
    selected_style = random.choice(daily_pair)

    print(f"🎭 Selected Persona: {selected_style['name']}")
    print("   🧠 Brainstorming abstract logic...")

    theme_prompt = (
        f"Brainstorm 3 highly specific, absurd reasons to bet {bet_summary}. "
        f"Persona: '{selected_style['name']}'. "
        f"CRITICAL RULES: DO NOT narrate a physical scene. Focus entirely on bizarre logic. "
        f"Output ONLY the 3 concepts separated by a pipe character (|)."
    )

    raw_themes = get_ai_text(theme_prompt)
    chosen_theme = "my gut feeling"
    if raw_themes:
        themes = [t.strip() for t in raw_themes.split("|") if len(t.strip()) > 5]
        if themes:
            chosen_theme = random.choice(themes)
            print(f"   💡 Chosen Concept: {chosen_theme}")

    # --- FINAL TWEET GENERATION ---
    full_prompt = (
        f"You are FadeGoblin, a chaotic, hyper-confident, degenerate sports bettor who completely embodies the randomly assigned persona.\n"
        f"Persona: {selected_style['prompt']}\n"
        f"Task: Write a short, unhinged social media post announcing your bet.\n"
        f"ADAPT this specific bizarre logic into your own words: '{chosen_theme}'.\n"
        f"The bet is on:\n{locked_bet_text}\n\n"
        f"RULES FOR THE TWEET:\n"
        f"1. NEVER break character. Be chaotic and highly confident. Keep it STRICTLY under 250 characters.\n"
        f"2. DO NOT write a clinical summary. Write a punchy, unhinged rant.\n"
        f"3. WEAVE the exact bet naturally into your manic rant.\n"
        f"4. DO NOT append a formal ticket or odds list at the bottom. The system will do this automatically.\n"
        f"5. DO NOT start with 'Locked', 'Locking in', or 'Placing'. Jump straight into the logic.\n"
        f"6. Use 1-2 relevant emojis, but don't overdo it.\n\n"
        f"Output ONLY the final in-character text, nothing else."
    )

    quote = get_ai_text(full_prompt)

    if not quote or "Do you want me to" in quote or "Options:" in quote:
        print("⚠️ API broke character. Using fallback.")
        quote = random.choice(FALLBACK_QUOTES)

    # Clean up the quote just in case it added quotes around its response
    quote = quote.strip('"').strip("'")

    # The system explicitly appends the exact compact ticket at the very end
    final_post = f"{quote}\n\n{locked_bet_text} [{final_odds_str}]"

    return final_post


def generate_preview_post_content(potd_leg: dict) -> str:
    """Generates a night-hype POTD preview post for an upcoming game tonight.

    Same chaotic persona as the morning POTD but framed as an evening reminder —
    'game is about to start, here's the lock' energy.
    """
    pick_line = f"{potd_leg['pick']} ({potd_leg['game']})"
    edge_str = f"+{potd_leg['edge']}%" if potd_leg["edge"] > 0 else f"{potd_leg['edge']}%"
    goblins = potd_leg.get("goblins", "👺")

    print(f"🌙 Night Preview: {pick_line} {potd_leg['odds']} | Edge: {edge_str}")

    # Same persona pool as POTD
    current_day = datetime.now().weekday()
    daily_pair = [PERSONAS[current_day * 2], PERSONAS[current_day * 2 + 1]]
    selected_style = random.choice(daily_pair)

    print(f"🎭 Night Persona: {selected_style['name']}")

    theme_prompt = (
        f"Brainstorm 3 highly specific, absurd reasons to bet {pick_line} TONIGHT. "
        f"Persona: '{selected_style['name']}'. "
        f"CRITICAL RULES: DO NOT narrate a physical scene. Focus entirely on bizarre logic. "
        f"Output ONLY the 3 concepts separated by a pipe character (|)."
    )

    raw_themes = get_ai_text(theme_prompt)
    chosen_theme = "my gut feeling"
    if raw_themes:
        themes = [t.strip() for t in raw_themes.split("|") if len(t.strip()) > 5]
        if themes:
            chosen_theme = random.choice(themes)

    badges = potd_leg.get("badges", [])
    badges_str = " | ".join(badges)
    badges_info = f"\nSYSTEM SIGNALS: {badges_str}\n" if badges else ""

    full_prompt = (
        f"You are FadeGoblin, a chaotic, hyper-confident, degenerate sports bettor.\n"
        f"Persona: {selected_style['prompt']}\n"
        f"Task: Write a short, unhinged social media post hyping your PLAY OF THE NIGHT — a game starting TONIGHT.\n"
        f"ADAPT this specific bizarre logic into your own words: '{chosen_theme}'.\n"
        f"The pick is: {pick_line} at {potd_leg['odds']}\n"
        f"{badges_info}\n"
        f"RULES FOR THE TWEET:\n"
        f"1. NEVER break character. Be chaotic and highly confident. Keep it STRICTLY under 220 characters.\n"
        f"2. Emphasize that this game is TONIGHT — create urgency.\n"
        f"3. WEAVE the exact pick naturally into your manic rant.\n"
        f"4. DO NOT append a formal ticket or odds list at the bottom. The system will do this automatically.\n"
        f"5. DO NOT start with 'Locked', 'Locking in', or 'Placing'. Jump straight into the logic.\n"
        f"6. Use 1-2 relevant emojis only.\n\n"
        f"Output ONLY the final in-character text, nothing else."
    )

    quote = get_ai_text(full_prompt)

    if not quote or "Do you want me to" in quote or "Options:" in quote:
        print("⚠️ API broke character. Using fallback.")
        quote = random.choice(FALLBACK_QUOTES)

    quote = quote.strip('"').strip("'")

    final_post = (
        f"🌙 {quote}\n\n"
        f"⭐ Tonight's Lock: {potd_leg['pick']} {potd_leg['odds']}  {goblins}\n"
        f"({potd_leg['game']})"
    )

    return final_post


def generate_recap_post_content(stats: dict) -> str:
    """Generates an unhinged FadeGoblin nightly recap post using the same persona system."""

    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    pushes = stats.get("pushes", 0)
    total = stats.get("total", 0)
    net_pnl = stats.get("net_pnl")
    date_str = stats.get("date", "yesterday")
    picks = stats.get("picks", [])

    if total == 0:
        return random.choice(FALLBACK_QUOTES) + "\n\n👺 No picks to recap today."

    # Build record string
    record = f"{wins}W-{losses}L" + (f"-{pushes}P" if pushes else "")
    pnl_note = ""
    if net_pnl is not None:
        sign = "+" if net_pnl >= 0 else ""
        pnl_note = f" ({sign}{net_pnl:.2f}u)"

    # Summarize results as compact lines for context
    pick_lines = []
    for p in picks:
        icon = {"WIN": "✅", "LOSS": "❌", "PUSH": "➡️"}.get(p.get("result", "?"), "❓")
        pick_lines.append(f"{icon} {p['pick']} {p['odds']}")
    results_block = "\n".join(pick_lines)

    print(f"📊 Recap: {record}{pnl_note} on {date_str}")

    # --- PERSONA — same pool as POTD ---
    current_day = datetime.now().weekday()
    daily_pair = [PERSONAS[current_day * 2], PERSONAS[current_day * 2 + 1]]
    selected_style = random.choice(daily_pair)

    print(f"🎭 Recap Persona: {selected_style['name']}")

    # --- GENERATE RECAP TEXT ---
    vibe = "glorious victory" if wins > losses else ("brutal suffering" if losses > wins else "chaotic neutral draw")
    full_prompt = (
        f"You are FadeGoblin, a chaotic, hyper-confident, degenerate sports bettor giving tonight's recap.\n"
        f"Persona: {selected_style['prompt']}\n"
        f"Today's record was {record}{pnl_note} — a day of {vibe}.\n"
        f"Task: Write a short, unhinged social media recap post. Reflect on the day's gambling results in character.\n"
        f"RULES:\n"
        f"1. NEVER break character. Be chaotic and emotionally unhinged about the result.\n"
        f"2. Keep it STRICTLY under 220 characters.\n"
        f"3. Reference the record ({record}) naturally — don't just announce it clinically.\n"
        f"4. DO NOT list every pick. The card shows that. Just react to the outcome.\n"
        f"5. Use 1-2 relevant emojis only.\n"
        f"6. DO NOT start with 'Locked' or 'Placing'.\n\n"
        f"Output ONLY the final in-character text, nothing else."
    )

    quote = get_ai_text(full_prompt)

    if not quote or "Do you want me to" in quote or "Options:" in quote:
        print("⚠️ API broke character. Using fallback.")
        quote = random.choice(FALLBACK_QUOTES)

    quote = quote.strip('"').strip("'")

    final_post = (
        f"👺 {quote}\n\n"
        f"📊 {date_str} Record: {record}{pnl_note}\n"
        f"Full card ⬇️"
    )

    return final_post


def generate_sniper_post_content(potd_leg: dict[str, Any]) -> str:
    """Generates an unhinged post focused on a single Play of the Day pick."""

    pick_line = f"{potd_leg['pick']} ({potd_leg['game']})"
    edge_str = f"+{potd_leg['edge']}%" if potd_leg["edge"] > 0 else f"{potd_leg['edge']}%"

    print(f"⭐ POTD: {pick_line} {potd_leg['odds']} | Edge: {edge_str}")

    # --- PERSONA & THEME INCEPTION ---
    current_day = datetime.now().weekday()
    daily_pair = [PERSONAS[current_day * 2], PERSONAS[current_day * 2 + 1]]
    selected_style = random.choice(daily_pair)

    print(f"🎭 Selected Persona: {selected_style['name']}")
    print("   🧠 Brainstorming chaotic POTD logic...")

    theme_prompt = (
        f"Brainstorm 3 highly specific, absurd reasons to bet {pick_line}. "
        f"Persona: '{selected_style['name']}'. "
        f"CRITICAL RULES: DO NOT narrate a physical scene. Focus entirely on bizarre logic. "
        f"Output ONLY the 3 concepts separated by a pipe character (|)."
    )

    raw_themes = get_ai_text(theme_prompt)
    chosen_theme = "my gut feeling"
    if raw_themes:
        themes = [t.strip() for t in raw_themes.split("|") if len(t.strip()) > 5]
        if themes:
            chosen_theme = random.choice(themes)
            print(f"   💡 Chosen Concept: {chosen_theme}")

    # --- BADGE CONTEXT ---
    badges = potd_leg.get("badges", [])
    badges_str = " | ".join(badges)
    badges_info = f"\nSYSTEM SIGNALS: {badges_str}\n(Note: Sharp Move means smart money is with us. High Confidence means model prob is elite.)\n" if badges else ""

    # --- FINAL TWEET GENERATION ---
    full_prompt = (
        f"You are FadeGoblin, a chaotic, hyper-confident, degenerate sports bettor.\n"
        f"Persona: {selected_style['prompt']}\n"
        f"Task: Write a short, unhinged social media post announcing your PLAY OF THE DAY.\n"
        f"ADAPT this specific bizarre logic into your own words: '{chosen_theme}'.\n"
        f"The pick is: {pick_line} at {potd_leg['odds']}\n"
        f"{badges_info}\n"
        f"RULES FOR THE TWEET:\n"
        f"1. NEVER break character. Be chaotic and highly confident. Keep it STRICTLY under 220 characters.\n"
        f"2. DO NOT write a clinical summary. Write a punchy, unhinged rant.\n"
        f"3. WEAVE the exact pick AND the system signals (badges) naturally into your manic rant.\n"
        f"4. DO NOT append a formal ticket or odds list at the bottom. The system will do this automatically.\n"
        f"5. DO NOT start with 'Locked', 'Locking in', or 'Placing'. Jump straight into the logic.\n"
        f"6. Use 1-2 relevant emojis, but don't overdo it.\n\n"
        f"Output ONLY the final in-character text, nothing else."
    )


    quote = get_ai_text(full_prompt)

    if not quote or "Do you want me to" in quote or "Options:" in quote:
        print("⚠️ API broke character. Using fallback.")
        quote = random.choice(FALLBACK_QUOTES)

    quote = quote.strip('"').strip("'")

    # Append the compact POTD ticket line
    final_post = (
        f"👺 {quote}\n\n"
        f"⭐ POTD: {potd_leg['pick']} ML {potd_leg['odds']}\n"
        f"Full card ⬇️"
    )

    return final_post

