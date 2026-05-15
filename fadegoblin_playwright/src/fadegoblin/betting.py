import random
from typing import Any


def calculate_parlay_odds(odds_list: list[int]) -> str:
    """Safely converts American odds to Decimal, multiplies them, and converts back."""
    if not odds_list:
        return "N/A"

    if len(odds_list) == 1:
        odd = odds_list[0]
        return f"+{odd}" if odd > 0 else str(odd)

    decimal_total = 1.0
    for odd in odds_list:
        if odd > 0:
            decimal_total *= (odd / 100.0) + 1.0
        elif odd < 0:
            decimal_total *= (100.0 / abs(odd)) + 1.0

    if decimal_total >= 2.0:
        american = int(round((decimal_total - 1.0) * 100.0))
        return f"+{american}"

    american = int(round(-100.0 / (decimal_total - 1.0)))
    return str(american)


def build_parlay(games: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Filters valid legs and constructs a mathematically sound parlay."""
    valid_legs = []
    for g in games:
        for side, odds_val in [
            (g["home"], g["home_odds"]),
            (g["away"], g["away_odds"]),
            ("Draw", g.get("draw_odds", "N/A")),
        ]:
            if odds_val != "N/A":
                int_odds = int(odds_val)
                # Drop heavy favorites worse than -350
                if int_odds >= -350:
                    valid_legs.append(
                        {
                            "game": f"{g['away']} @ {g['home']}",
                            "pick": side,
                            "odds": int_odds,
                        }
                    )

    if not valid_legs:
        return [], "N/A"

    chosen_legs = []
    final_odds_str = "N/A"

    # Try up to 100 times to build a mathematically sound bet (capped roughly to +400)
    for _ in range(100):
        num_legs = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
        sample = random.sample(valid_legs, min(num_legs, len(valid_legs)))

        # Prevent taking two sides of the same game
        seen_games = set()
        conflict = False
        for leg in sample:
            if leg["game"] in seen_games:
                conflict = True
                break
            seen_games.add(leg["game"])

        if conflict:
            continue

        odds_ints = [leg["odds"] for leg in sample]
        calc_str = calculate_parlay_odds(odds_ints)
        calc_int = int(calc_str)

        # Keep the final ticket reasonable
        if -200 <= calc_int <= 400:
            chosen_legs = sample
            final_odds_str = calc_str
            break

    # Safety net if the loop fails
    if not chosen_legs:
        chosen_legs = [random.choice(valid_legs)]
        final_odds_str = calculate_parlay_odds([chosen_legs[0]["odds"]])

    return chosen_legs, final_odds_str
