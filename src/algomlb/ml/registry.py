import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text, delete

from algomlb.db.models import GameManagerRegistryORM, GameManagerRegistryORM as Registry
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


def build_manager_registry(
    session: Session, start_year: int = 2019, end_year: int = 2026
):
    """
    Builds the game_manager_registry by joining retrosheet_events and game_results.
    Resolves manager_id and computes tenure metrics.
    """
    engine = get_engine()

    for year in range(start_year, end_year + 1):
        if year == 2020:
            continue  # Skip shortened season for now or process if data looks clean

        logger.info(f"Building Manager Registry for {year}...")

        # 1. Extract unique retrosheet game markers
        # game_id format: HHHYYYYMMDDG (G=game number, usually 0)
        q_retro = text("""
            SELECT DISTINCT 
                game_id, 
                date
            FROM retrosheet_events
            WHERE EXTRACT(YEAR FROM date) = :year
        """)
        df_retro = pd.read_sql(q_retro, engine, params={"year": year})

        if df_retro.empty:
            logger.warning(f"No retrosheet events found for {year}")
            continue

        # 2. Extract master game keys from game_results
        q_results = text("""
            SELECT 
                game_id as game_pk, 
                game_date, 
                home_team_id, 
                away_team_id,
                game_type as results_game_type,
                doubleheader_num
            FROM game_results
            WHERE EXTRACT(YEAR FROM game_date) = :year
        """)
        df_results = pd.read_sql(q_results, engine, params={"year": year})
        df_results["game_pk"] = df_results["game_pk"].astype(int)

        # 3. Map Retrosheet IDs to Game PKs
        # Retrosheet has separate rows for home and away perspective?
        # Actually retrosheet_events.game_id is global for the game.
        # We need to map team abbreviations for the join.

        # Helper to parse retrosheet home team from game_id (first 3 chars)
        df_retro["home_team_abbr"] = df_retro["game_id"].str[:3]
        df_retro["home_team_id"] = (
            df_retro["home_team_abbr"].map(RETROSHEET_TEAM_MAP).astype(float)
        )
        df_retro["dh_num"] = df_retro["game_id"].str[-1].astype(float)

        df_results["home_team_id"] = df_results["home_team_id"].astype(float)
        df_results["dh_num"] = df_results["doubleheader_num"].astype(float)

        # Join on date and home_team_id
        # Ensure date types match for merge (coerce to datetime)
        df_retro["date"] = pd.to_datetime(df_retro["date"])
        df_results["game_date"] = pd.to_datetime(df_results["game_date"])

        df_mapped = pd.merge(
            df_retro,
            df_results,
            left_on=["date", "home_team_id", "dh_num"],
            right_on=["game_date", "home_team_id", "dh_num"],
            how="inner",
        )
        # Ensure 1:1 mapping (drop any duplicates from results side)
        df_mapped = df_mapped.drop_duplicates(subset=["game_id"])

        logger.info(f"Mapped {len(df_mapped)} games for {year}")
        if df_mapped.empty:
            logger.warning(f"No games mapped correctly for {year} after join")
            continue

        # 4. Resolve managers and compute tenure (Double-pivot for team-game grain)
        # We need to create two rows per game_pk (home and away)
        rows = []
        for _, game in df_mapped.iterrows():
            # Home row
            rows.append(
                {
                    "game_pk": int(game["game_pk"]),
                    "retrosheet_game_id": game["game_id"],
                    "team_id": int(game["home_team_id"]),
                    "opponent_id": int(game["away_team_id"]),
                    "game_date": game["game_date"],
                    "season": year,
                    "home_away": "home",
                    "game_type": game["results_game_type"],
                    "doubleheader_num": int(game["game_id"][-1]),
                }
            )
            # Away row
            rows.append(
                {
                    "game_pk": int(game["game_pk"]),
                    "retrosheet_game_id": game["game_id"],
                    "team_id": int(game["away_team_id"]),
                    "opponent_id": int(game["home_team_id"]),
                    "game_date": game["game_date"],
                    "season": year,
                    "home_away": "away",
                    "game_type": game["results_game_type"],
                    "doubleheader_num": int(game["game_id"][-1]),
                }
            )

        df_registry = pd.DataFrame(rows)
        if df_registry.empty:
            continue

        # Ensure grain integrity
        df_registry = df_registry.drop_duplicates(subset=["game_pk", "team_id"])

        # 5. Attach manager_id from team_managers
        # Note: Handling mid-season switches by sorting and using date logic
        q_mgrs = text(
            "SELECT team_id, manager_id, season, effective_start_date FROM team_managers"
        )
        df_mgrs = pd.read_sql(q_mgrs, engine)

        # Join registry with managers
        # This is where we need the date-aware logic.
        # For now, if effective_start_date is null, use season.
        def resolve_manager(team_id, game_dt, season):
            team_mgrs = df_mgrs[
                (df_mgrs["team_id"] == team_id) & (df_mgrs["season"] == season)
            ]
            if len(team_mgrs) == 0:
                return None
            if len(team_mgrs) == 1:
                return team_mgrs.iloc[0]["manager_id"]

            # Mid-season logic: take the one with latest start_date <= game_dt
            valid_mgrs = team_mgrs[team_mgrs["effective_start_date"] <= game_dt]
            if not valid_mgrs.empty:
                return valid_mgrs.sort_values(
                    "effective_start_date", ascending=False
                ).iloc[0]["manager_id"]
            return team_mgrs.sort_values("effective_start_date").iloc[0]["manager_id"]

        df_registry["manager_id"] = df_registry.apply(
            lambda x: resolve_manager(x["team_id"], x["game_date"], x["season"]), axis=1
        )

        # Drop rows where manager couldn't be resolved (e.g. All-Star, though we filtered those)
        df_registry = df_registry.dropna(subset=["manager_id"])
        df_registry["manager_id"] = df_registry["manager_id"].astype(int)

        # 6. Compute tenure and stints
        # Sort by team, manager, date
        df_registry = df_registry.sort_values(["team_id", "game_date"])

        # Identify stints (consecutive games with same manager/team)
        df_registry["m_diff"] = (
            df_registry.groupby("team_id")["manager_id"].diff().fillna(0).ne(0).cumsum()
        )

        # manager_stint_start as the date of the first game in that stint
        df_registry["manager_stint_start"] = df_registry.groupby(["team_id", "m_diff"])[
            "game_date"
        ].transform("min")

        # manager_tenure_day as row_number within (team, manager, stint)
        df_registry["manager_tenure_day"] = (
            df_registry.groupby(["team_id", "m_diff"]).cumcount() + 1
        )

        # days_since_manager_change
        df_registry["days_since_manager_change"] = (
            pd.to_datetime(df_registry["game_date"])
            - pd.to_datetime(df_registry["manager_stint_start"])
        ).dt.days

        # 7. Persist
        # Clear existing for this year to allow clean re-run
        session.execute(
            delete(GameManagerRegistryORM).where(GameManagerRegistryORM.season == year)
        )

        # Bulk insert
        registry_objs = [
            Registry(**{str(k): v for k, v in row.items()})
            for row in df_registry.drop(columns=["m_diff"]).to_dict("records")
        ]
        session.bulk_save_objects(registry_objs)
        session.commit()

        logger.success(f"Built {len(df_registry)} registry rows for {year}")
