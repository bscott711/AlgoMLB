import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text, delete

from algomlb.db.models import GameManagerRegistryORM as Registry
from algomlb.db.session import get_engine
from algomlb.core.logger import logger

# Retrosheet Team Abbreviations to MLB Team IDs Mapping
RETROSHEET_TEAM_MAP = {
    "ANA": 108,
    "ARI": 109,
    "ATL": 144,
    "BAL": 110,
    "BOS": 111,
    "CHA": 145,
    "CHN": 112,
    "CIN": 113,
    "CLE": 114,
    "COL": 115,
    "DET": 116,
    "HOU": 117,
    "KCA": 118,
    "LAN": 119,
    "MIA": 146,
    "MIL": 158,
    "MIN": 142,
    "NYA": 147,
    "NYN": 121,
    "OAK": 133,
    "PHI": 143,
    "PIT": 134,
    "SDN": 135,
    "SEA": 136,
    "SFN": 137,
    "SLN": 138,
    "TBA": 139,
    "TEX": 140,
    "TOR": 141,
    "WAS": 120,
}


def _resolve_manager(df_mgrs, team_id, game_dt, season):
    """Resolves manager_id for a team on a specific date, handling mid-season switches."""
    team_mgrs = df_mgrs[(df_mgrs["team_id"] == team_id) & (df_mgrs["season"] == season)]
    if team_mgrs.empty:
        return None
    if len(team_mgrs) == 1:
        return team_mgrs.iloc[0]["manager_id"]

    valid = team_mgrs[team_mgrs["effective_start_date"] <= game_dt]
    if not valid.empty:
        return valid.sort_values("effective_start_date", ascending=False).iloc[0][
            "manager_id"
        ]
    return team_mgrs.sort_values("effective_start_date").iloc[0]["manager_id"]


def _fetch_registry_data(engine, year):
    """Extraction layer for Retrosheet and Game Results."""
    df_retro = pd.read_sql(
        text(
            "SELECT DISTINCT game_id, date FROM retrosheet_events WHERE EXTRACT(YEAR FROM date) = :year"
        ),
        engine,
        params={"year": year},
    )
    df_results = pd.read_sql(
        text(
            "SELECT game_id as game_pk, game_date, home_team_id, away_team_id, game_type, doubleheader_num FROM game_results WHERE EXTRACT(YEAR FROM game_date) = :year"
        ),
        engine,
        params={"year": year},
    )
    return df_retro, df_results


def _map_and_merge_games(df_retro, df_results):
    """Coordinate-joins Retrosheet markers to MLB Game PKs."""
    df_retro["home_team_id"] = (
        df_retro["game_id"].str[:3].map(RETROSHEET_TEAM_MAP).astype(float)
    )
    df_retro["dh_num"] = df_retro["game_id"].str[-1].astype(float)
    df_retro["date"] = pd.to_datetime(df_retro["date"])

    df_results["dh_num"] = df_results["doubleheader_num"].astype(float)
    df_results["game_date"] = pd.to_datetime(df_results["game_date"])

    df_m = pd.merge(
        df_retro,
        df_results,
        left_on=["date", "home_team_id", "dh_num"],
        right_on=["game_date", "home_team_id", "dh_num"],
        how="inner",
    )
    return df_m.drop_duplicates(subset=["game_id"])


def _compute_tenure_metrics(df_reg):
    """Group-by calculations for stints and day counts."""
    df_reg = df_reg.sort_values(["team_id", "game_date"])
    df_reg["m_diff"] = (
        df_reg.groupby("team_id")["manager_id"].diff().fillna(0).ne(0).cumsum()
    )
    df_reg["manager_stint_start"] = df_reg.groupby(["team_id", "m_diff"])[
        "game_date"
    ].transform("min")
    df_reg["manager_tenure_day"] = df_reg.groupby(["team_id", "m_diff"]).cumcount() + 1
    df_reg["days_since_manager_change"] = (
        pd.to_datetime(df_reg["game_date"])
        - pd.to_datetime(df_reg["manager_stint_start"])
    ).dt.days
    return df_reg.drop(columns=["m_diff"])


def build_manager_registry(
    session: Session, start_year: int = 2019, end_year: int = 2026
):
    """Resolves manager attribution and stint metadata for the entire league."""
    engine, df_mgrs = (
        get_engine(),
        pd.read_sql(
            "SELECT team_id, manager_id, season, effective_start_date FROM team_managers",
            get_engine(),
        ),
    )

    for year in range(start_year, end_year + 1):
        if year == 2020:
            continue
        logger.info(f"Building Manager Registry for {year}...")

        df_retro, df_results = _fetch_registry_data(engine, year)
        if df_retro.empty or df_results.empty:
            continue

        df_m = _map_and_merge_games(df_retro, df_results)
        if df_m.empty:
            continue

        rows = []
        for _, g in df_m.iterrows():
            params = {
                "game_pk": int(g["game_pk"]),
                "retrosheet_game_id": g["game_id"],
                "game_date": g["game_date"],
                "season": year,
                "game_type": g["game_type"],
                "doubleheader_num": int(g["game_id"][-1]),
            }
            rows.append(
                {
                    **params,
                    "team_id": int(g["home_team_id"]),
                    "opponent_id": int(g["away_team_id"]),
                    "home_away": "home",
                }
            )
            rows.append(
                {
                    **params,
                    "team_id": int(g["away_team_id"]),
                    "opponent_id": int(g["home_team_id"]),
                    "home_away": "away",
                }
            )

        df_reg = pd.DataFrame(rows).drop_duplicates(subset=["game_pk", "team_id"])
        df_reg["manager_id"] = df_reg.apply(
            lambda x: _resolve_manager(
                df_mgrs, x["team_id"], x["game_date"], x["season"]
            ),
            axis=1,
        )
        df_reg = _compute_tenure_metrics(df_reg.dropna(subset=["manager_id"]))

        # Persistence
        session.execute(delete(Registry).where(Registry.season == year))
        session.bulk_save_objects(
            [
                Registry(**{str(k): v for k, v in r.items()})
                for r in df_reg.to_dict("records")
            ]
        )
        session.commit()
        logger.success(f"Built {len(df_reg)} registry rows for {year}")
