import datetime
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from pathlib import Path

from algomlb.ml.model import MLBModel
from algomlb.db.models import (
    GameResultORM,
    UraniumSimulatedPlayerPropsORM,
    GameLineupORM,
)
from algomlb.ingestion.lineup_ingester import LineupIngester
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
import importlib
import algomlb.ml.monte_carlo.state as mc_state
import algomlb.ml.monte_carlo.loader as mc_loader
import algomlb.ml.monte_carlo.aggregator as mc_aggregator
import algomlb.ml.monte_carlo.engine as mc_engine

# Force reload for stale dashboard environments (must be in order of dependencies)
importlib.reload(mc_state)
importlib.reload(mc_loader)
importlib.reload(mc_aggregator)
importlib.reload(mc_engine)

from algomlb.ml.monte_carlo.loader import MatchupLoader  # noqa: E402
from algomlb.ml.monte_carlo.aggregator import SimulationAggregator  # noqa: E402
from algomlb.ml.monte_carlo.engine import SimulationEngine  # noqa: E402


def get_upcoming_games(session: Session, selected_date: datetime.date) -> pd.DataFrame:
    """Fetch games for a specific date, including upcoming ones."""
    stmt = select(
        GameResultORM.game_id,
        GameResultORM.away_team,
        GameResultORM.home_team,
        GameResultORM.away_pitcher,
        GameResultORM.home_pitcher,
        GameResultORM.away_score,
        GameResultORM.home_score,
        GameResultORM.status,
        GameResultORM.game_date,
    ).where(GameResultORM.game_date == selected_date)

    results = session.execute(stmt).all()
    if not results:
        return pd.DataFrame()

    return pd.DataFrame([dict(r._asdict()) for r in results])


def load_simulation_results(session: Session, game_pk: int) -> pd.DataFrame:
    """Load pre-computed simulation results for a game."""
    stmt = select(UraniumSimulatedPlayerPropsORM).where(
        UraniumSimulatedPlayerPropsORM.game_pk == game_pk
    )
    results = session.execute(stmt).scalars().all()
    if not results:
        return pd.DataFrame()

    data = []
    for r in results:
        data.append(
            {
                "player_id": r.player_id,
                "stat_type": r.stat_type,
                "mean": r.mean,
                "median": r.median,
                "prob_over_0_5": r.prob_over_0_5,
                "prob_over_1_5": r.prob_over_1_5,
                "prob_over_2_5": r.prob_over_2_5,
                "p10": r.p10,
                "p90": r.p90,
                "trials": r.trials,
            }
        )
    return pd.DataFrame(data)


def run_and_persist_simulation(
    session: Session, game_pk: int, trials: int, version: str = "v1.6"
) -> pd.DataFrame:
    """Run a new simulation and save results to the database."""
    # 1. Ensure Lineups Exist (Auto-Ingest if missing)
    lineup_exists = (
        session.execute(
            select(sa.func.count())
            .select_from(GameLineupORM)
            .where(GameLineupORM.game_pk == game_pk)
        ).scalar()
        or 0
    ) > 0

    if not lineup_exists:
        # Fetch game date first
        game_date = session.execute(
            select(GameResultORM.game_date).where(GameResultORM.game_id == str(game_pk))
        ).scalar()
        if game_date:
            ingester = LineupIngester(session)
            ingester.ingest_game(game_pk, game_date)
            session.commit()

    # 2. Load context
    loader = MatchupLoader(session)
    context = loader.load_matchup(game_pk)

    if context is None:
        raise ValueError(
            f"CRITICAL: MatchupLoader returned None for game {game_pk}. This should not happen. Check backend logs."
        )

    # 3. Load model (v1.6 is the new production standard)
    model_path = Path(f".data/models/pa_outcome_{version}.joblib")
    if not model_path.exists():
        # Fallback to the unversioned link if v1.6 explicitly isn't found
        model_path = Path(".data/models/pa_outcome.joblib")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Production model not found at {model_path}")
    
    model = MLBModel.load(model_path)

    # 4. Simulate
    engine = SimulationEngine(pa_model=model)
    trial_results = engine.run_trials(context, trials=trials)

    # 5. Aggregate
    aggregator = SimulationAggregator()
    results_df = aggregator.aggregate_results(
        game_pk, context.game_date.year, trial_results, context
    )

    # 6. Persist
    records = results_df.to_dict(orient="records")
    for rec in records:
        stmt = pg_insert(UraniumSimulatedPlayerPropsORM).values([rec])
        upsert = stmt.on_conflict_do_update(
            index_elements=["game_pk", "player_id", "stat_type"],
            set_={
                "season": stmt.excluded.season,
                "mean": stmt.excluded.mean,
                "median": stmt.excluded.median,
                "prob_over_0_5": stmt.excluded.prob_over_0_5,
                "prob_over_1_5": stmt.excluded.prob_over_1_5,
                "prob_over_2_5": stmt.excluded.prob_over_2_5,
                "prob_over_3_5": stmt.excluded.prob_over_3_5,
                "prob_over_4_5": stmt.excluded.prob_over_4_5,
                "p10": stmt.excluded.p10,
                "p90": stmt.excluded.p90,
                "trials": stmt.excluded.trials,
                "simulated_at": sa.func.now(),
            },
        )
        session.execute(upsert)

    session.commit()
    return results_df


def get_uranium_prediction(context: mc_loader.MatchupContext) -> float:
    """Run a top-down win probability prediction using the production Uranium model."""
    from algomlb.ml.model import MLBModel
    from pathlib import Path
    import pandas as pd

    model_path = Path(".data/models/uranium_win_model.joblib")
    if not model_path.exists():
        model_path = Path(".data/models/home_win_v1.0.joblib")

    if not model_path.exists():
        # Fallback to Elo if no top-down model exists
        h_elo = context.matchup_features.get("home_team_elo_pre", 1500)
        a_elo = context.matchup_features.get("away_team_elo_pre", 1500)
        return 1 / (1 + 10 ** (-(h_elo + 24 - a_elo) / 400))

    model = MLBModel.load(model_path)

    # 1. Build Feature Row
    row = {}

    # Global Matchup Features
    row.update(context.matchup_features)

    # Starting Pitcher Features
    h_sp_id = context.home_starter.pitcher_id
    a_sp_id = context.away_starter.pitcher_id

    for k, v in context.pitcher_features.get(h_sp_id, {}).items():
        row[f"h_sp_{k}"] = v
    for k, v in context.pitcher_features.get(a_sp_id, {}).items():
        row[f"a_sp_{k}"] = v

    # Team Hitting (Mean of lineup)
    def agg_batting(pids, prefix):
        res = {}
        count = 0
        for pid in pids:
            feats = context.batter_features.get(pid, {})
            for k, v in feats.items():
                res[f"{prefix}{k}"] = res.get(f"{prefix}{k}", 0) + v
            count += 1
        if count > 0:
            for k in res:
                res[k] /= count
        return res

    row.update(agg_batting([b.player_id for b in context.home_lineup], "h_bat_"))
    row.update(agg_batting([b.player_id for b in context.away_lineup], "a_bat_"))

    # 2. Run Prediction
    X = pd.DataFrame([row])

    # Reindex to match model features
    base_est = model.get_base_xgb_estimator()
    if hasattr(base_est, "feature_names_in_"):
        expected = base_est.feature_names_in_
        X = X.reindex(columns=expected, fill_value=0.0)

    probs = model.predict_proba(X)[0]
    # We want home win prob (index 1 usually)
    return float(probs[1]) if len(probs) > 1 else float(probs[0])
