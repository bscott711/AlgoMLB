# src/algomlb/ml/sabermetrics.py
"""
Fundamental sabermetric features for AlgoMLB.

Computes team-level structural metrics that encode known baseball laws:
- Pythagorean Expectation (expected win% from runs scored/allowed)
- Rolling run differentials

These are computed from game_results only — no market data.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from algomlb.core.logger import logger


def compute_pythagorean_features(
    games: pd.DataFrame,
    window: int = 30,
    exponent: float = 1.83,
) -> pd.DataFrame:
    """
    Compute rolling Pythagorean win expectation for each team at each game.

    The Pythagorean theorem of baseball:
        WinPct = RS^exp / (RS^exp + RA^exp)

    Uses a trailing window of `window` games per team.

    Args:
        games: DataFrame with game_pk, game_date, home_team, away_team,
               home_score, away_score
        window: Number of trailing games to use (default 30)
        exponent: Pythagorean exponent (1.83 is the MLB standard)

    Returns:
        DataFrame with game_pk, team_id, is_home,
        pythag_win_pct, roll_run_diff, roll_rs_per_game, roll_ra_per_game
    """
    games = games.copy().sort_values("game_date")

    # Build a per-team chronological log of (game_pk, RS, RA, game_date, is_home)
    team_logs: dict[str, list[dict]] = defaultdict(list)

    for _, g in games.iterrows():
        game_pk = int(g["game_pk"])
        game_date = g["game_date"]
        hs = float(g["home_score"])
        aws = float(g["away_score"])
        home = str(g["home_team"])
        away = str(g["away_team"])

        team_logs[home].append(
            dict(game_pk=game_pk, game_date=game_date, rs=hs, ra=aws, is_home=True)
        )
        team_logs[away].append(
            dict(game_pk=game_pk, game_date=game_date, rs=aws, ra=hs, is_home=False)
        )

    rows: list[dict] = []

    for team_id, log in team_logs.items():
        for i, entry in enumerate(log):
            # Trailing window (exclusive of current game — look-back only)
            start = max(0, i - window)
            trail = log[start:i]

            if len(trail) < 5:
                # Not enough history for a stable estimate
                pythag = 0.5
                rs_pg = 0.0
                ra_pg = 0.0
                run_diff = 0.0
            else:
                total_rs = sum(t["rs"] for t in trail)
                total_ra = sum(t["ra"] for t in trail)
                n = len(trail)
                rs_pg = total_rs / n
                ra_pg = total_ra / n
                run_diff = (total_rs - total_ra) / n

                # Pythagorean formula
                rs_exp = total_rs**exponent
                ra_exp = total_ra**exponent
                denom = rs_exp + ra_exp
                pythag = rs_exp / denom if denom > 0 else 0.5

            rows.append(
                dict(
                    game_pk=entry["game_pk"],
                    game_date=entry["game_date"],
                    team_id=team_id,
                    is_home=entry["is_home"],
                    pythag_win_pct=pythag,
                    roll_run_diff=run_diff,
                    roll_rs_per_game=rs_pg,
                    roll_ra_per_game=ra_pg,
                )
            )

    result = pd.DataFrame(rows)
    logger.info(
        f"Computed Pythagorean features for {len(team_logs)} teams, "
        f"{len(result)} rows."
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
# RE24: Run Expectancy based on 24 base-out states
# ═══════════════════════════════════════════════════════════════════════

def _encode_base_out_state(outs: int, br1: str | None, br2: str | None, br3: str | None) -> str:
    """
    Encode the base-out state into a canonical string.
    Example: "2_100" = 2 outs, runner on 1st only.
    """
    b1 = "1" if (br1 is not None and str(br1).strip() != "") else "0"
    b2 = "1" if (br2 is not None and str(br2).strip() != "") else "0"
    b3 = "1" if (br3 is not None and str(br3).strip() != "") else "0"
    return f"{min(int(outs), 2)}_{b1}{b2}{b3}"


def build_run_expectancy_matrix(events_df: pd.DataFrame) -> dict[str, float]:
    """
    Build a Run Expectancy matrix from Retrosheet event data.

    For each of the 24 base-out states, compute the average number of runs
    scored from that state through the end of the half-inning.

    Args:
        events_df: Retrosheet events with columns:
            game_id, inning, top_bot, outs_pre, br1_pre, br2_pre, br3_pre,
            runs, pa_flag

    Returns:
        Dict mapping state string (e.g. "0_000") to expected runs.
    """
    df = events_df.copy()

    # Only plate appearances
    df = df[df["pa_flag"] == 1].copy()

    # Encode pre-state
    df["state"] = df.apply(
        lambda r: _encode_base_out_state(r["outs_pre"], r.get("br1_pre"), r.get("br2_pre"), r.get("br3_pre")),
        axis=1,
    )

    # For each half-inning, compute total runs scored from each PA onward
    df["half_inning_id"] = df["game_id"] + "_" + df["inning"].astype(str) + "_" + df["top_bot"].astype(str)

    # Reverse cumulative sum of runs within each half-inning
    df = df.sort_values(["game_id", "inning", "top_bot", "outs_pre"])
    df["runs_rest_of_inning"] = df.groupby("half_inning_id")["runs"].transform(
        lambda x: x.iloc[::-1].cumsum().iloc[::-1]
    )

    # Average RE per state
    re_matrix = df.groupby("state")["runs_rest_of_inning"].mean().to_dict()

    # Ensure all 24 states exist (3 out states × 8 base combos)
    for outs in range(3):
        for bases in range(8):
            b1 = "1" if bases & 1 else "0"
            b2 = "1" if bases & 2 else "0"
            b3 = "1" if bases & 4 else "0"
            state = f"{outs}_{b1}{b2}{b3}"
            if state not in re_matrix:
                re_matrix[state] = 0.0

    logger.info(f"Built RE24 matrix with {len(re_matrix)} states. "
                f"Empty bases/0 outs: {re_matrix.get('0_000', 0):.3f} expected runs.")
    return re_matrix


def compute_re24_per_pa(
    events_df: pd.DataFrame,
    re_matrix: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute RE24 value for each plate appearance.

    RE24 = (runs_scored_on_play + RE_post) - RE_pre

    A positive RE24 means the batter/pitcher created/allowed more runs
    than expected given the situation.

    Args:
        events_df: Retrosheet events DataFrame.
        re_matrix: Pre-computed RE matrix. If None, builds one from the data.

    Returns:
        DataFrame with columns:
            game_id, batter_id, pitcher_id, bat_team, pit_team,
            date, re24_batter (positive = good for batter),
            re24_pitcher (positive = bad for pitcher / runs allowed above average)
    """
    df = events_df.copy()
    df = df[df["pa_flag"] == 1].copy()

    if re_matrix is None:
        re_matrix = build_run_expectancy_matrix(events_df)

    # Pre-state
    df["state_pre"] = df.apply(
        lambda r: _encode_base_out_state(r["outs_pre"], r.get("br1_pre"), r.get("br2_pre"), r.get("br3_pre")),
        axis=1,
    )
    df["re_pre"] = df["state_pre"].map(re_matrix).fillna(0.0)

    # Post-state
    df["state_post"] = df.apply(
        lambda r: _encode_base_out_state(r["outs_post"], r.get("br1_post"), r.get("br2_post"), r.get("br3_post")),
        axis=1,
    )
    # If 3 outs, RE = 0 (end of half-inning)
    df["re_post"] = df["state_post"].map(re_matrix).fillna(0.0)
    df.loc[df["outs_post"] >= 3, "re_post"] = 0.0

    # RE24 = runs_scored + RE_post - RE_pre
    df["re24"] = df["runs"].fillna(0).astype(float) + df["re_post"] - df["re_pre"]

    result = df[["game_id", "date", "batter_id", "pitcher_id", "bat_team", "pit_team", "re24"]].copy()
    result = result.rename(columns={"re24": "re24_batter"})
    # For pitchers, negative RE24 is good (prevented runs)
    result["re24_pitcher"] = -result["re24_batter"]

    logger.info(f"Computed RE24 for {len(result)} plate appearances.")
    return result


def compute_rolling_re24(
    re24_pa: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Compute rolling RE24 averages per player for use as Uranium features.

    For batters: rolling mean of re24_batter over last `window` games.
    For pitchers: rolling mean of re24_pitcher over last `window` games.

    Returns DataFrame with:
        player_id, game_date, role, roll_re24
    """
    rows: list[dict] = []

    # ── Batter RE24 ──
    batter_game = (
        re24_pa.groupby(["batter_id", "game_id", "date"])["re24_batter"]
        .sum()
        .reset_index()
        .rename(columns={"batter_id": "player_id", "date": "game_date", "re24_batter": "game_re24"})
    )
    batter_game = batter_game.sort_values(["player_id", "game_date"])
    batter_game["roll_re24"] = batter_game.groupby("player_id")["game_re24"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )
    for _, r in batter_game.dropna(subset=["roll_re24"]).iterrows():
        rows.append(dict(
            player_id=r["player_id"],
            game_date=r["game_date"],
            role="BATTER",
            roll_re24=r["roll_re24"],
        ))

    # ── Pitcher RE24 ──
    pitcher_game = (
        re24_pa.groupby(["pitcher_id", "game_id", "date"])["re24_pitcher"]
        .sum()
        .reset_index()
        .rename(columns={"pitcher_id": "player_id", "date": "game_date", "re24_pitcher": "game_re24"})
    )
    pitcher_game = pitcher_game.sort_values(["player_id", "game_date"])
    pitcher_game["roll_re24"] = pitcher_game.groupby("player_id")["game_re24"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )
    for _, r in pitcher_game.dropna(subset=["roll_re24"]).iterrows():
        rows.append(dict(
            player_id=r["player_id"],
            game_date=r["game_date"],
            role="PITCHER",
            roll_re24=r["roll_re24"],
        ))

    result = pd.DataFrame(rows)
    logger.info(f"Computed rolling RE24 for {len(result)} player-game rows.")
    return result

