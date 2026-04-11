import pandas as pd
import pytest
from algomlb.ml.features import FeaturePipeline


@pytest.fixture
def pipeline():
    return FeaturePipeline()


def test_aggregate_team_batting_full(pipeline):
    # Test valid aggregation (both sides)
    lineups = pd.DataFrame(
        [
            {
                "game_pk": 101.0,
                "game_date": pd.to_datetime("2023-04-01"),
                "team_side": "home",
                "player_id": 1.0,
            },
            {
                "game_pk": 101.0,
                "game_date": pd.to_datetime("2023-04-01"),
                "team_side": "away",
                "player_id": 2.0,
            },
        ]
    )
    batter_gold = pd.DataFrame(
        [
            {
                "game_date": pd.to_datetime("2023-04-01"),
                "player_id": 1.0,
                "roll_pas": 10,
            },
            {
                "game_date": pd.to_datetime("2023-04-01"),
                "player_id": 2.0,
                "roll_pas": 20,
            },
        ]
    )

    h_res = pipeline._aggregate_team_batting(lineups, batter_gold, "home")
    a_res = pipeline._aggregate_team_batting(lineups, batter_gold, "away")
    assert h_res.loc[0, "h_bat_roll_pas"] == 10
    assert a_res.loc[0, "a_bat_roll_pas"] == 20

    # Test empty side branch (line 45)
    assert pipeline._aggregate_team_batting(lineups, batter_gold, "neutral").empty

    # Test no columns branch (line 74)
    bad_gold = pd.DataFrame([{"game_date": "2023-04-01", "player_id": 1.0, "junk": 1}])
    assert pipeline._aggregate_team_batting(lineups, bad_gold, "home").empty


def test_build_uranium_matrix_exhaustive(pipeline):
    pitcher_gold = pd.DataFrame(
        [
            {
                "player_id": 1.0,
                "game_date": pd.to_datetime("2023-04-01"),
                "some_stat": 1.0,
            },
            {
                "player_id": 2.0,
                "game_date": pd.to_datetime("2023-04-01"),
                "some_stat": 0.5,
            },
            {
                "player_id": 1.0,
                "game_date": pd.to_datetime("2023-04-02"),
                "some_stat": 2.0,
            },
            {
                "player_id": 2.0,
                "game_date": pd.to_datetime("2023-04-02"),
                "some_stat": 0.8,
            },
        ]
    )
    # Use datetime64[ns] to match pipeline requirement
    games_df = pd.DataFrame(
        [
            {
                "game_pk": 101.0,
                "game_date": pd.to_datetime("2023-04-01"),
                "home_pitcher_id": 1.0,
                "away_pitcher_id": 2.0,
                "home_score": 5,
                "away_score": 3,
            },
            {
                "game_pk": 102.0,
                "game_date": pd.to_datetime("2023-04-02"),
                "home_pitcher_id": 1.0,
                "away_pitcher_id": 2.0,
                "home_score": 2,
                "away_score": 4,
            },
        ]
    )
    lineups = pd.DataFrame(
        [
            {
                "game_pk": 101.0,
                "game_date": pd.to_datetime("2023-04-01"),
                "team_side": "home",
                "player_id": 3.0,
            },
            {
                "game_pk": 101.0,
                "game_date": pd.to_datetime("2023-04-01"),
                "team_side": "away",
                "player_id": 4.0,
            },
            {
                "game_pk": 102.0,
                "game_date": pd.to_datetime("2023-04-02"),
                "team_side": "home",
                "player_id": 3.0,
            },
            {
                "game_pk": 102.0,
                "game_date": pd.to_datetime("2023-04-02"),
                "team_side": "away",
                "player_id": 4.0,
            },
        ]
    )
    batter_gold = pd.DataFrame(
        [
            {
                "game_date": pd.to_datetime("2023-04-01"),
                "player_id": 3.0,
                "roll_pas": 100,
            },
            {
                "game_date": pd.to_datetime("2023-04-01"),
                "player_id": 4.0,
                "roll_pas": 150,
            },
            {
                "game_date": pd.to_datetime("2023-04-02"),
                "player_id": 3.0,
                "roll_pas": 200,
            },
            {
                "game_date": pd.to_datetime("2023-04-02"),
                "player_id": 4.0,
                "roll_pas": 250,
            },
        ]
    )
    elo_df = pd.DataFrame(
        [
            {"game_pk": 101.0, "is_home": True, "elo_pre": 1500, "elo_post": 1510},
            {"game_pk": 101.0, "is_home": False, "elo_pre": 1450, "elo_post": 1440},
            {"game_pk": 102.0, "is_home": True, "elo_pre": 1510, "elo_post": 1500},
            {"game_pk": 102.0, "is_home": False, "elo_pre": 1440, "elo_post": 1450},
        ]
    )
    pythag_df = pd.DataFrame(
        [
            {
                "game_pk": 101.0,
                "is_home": True,
                "pythag_win_pct": 0.6,
                "roll_run_diff": 1,
                "roll_rs_per_game": 5,
                "roll_ra_per_game": 4,
            },
            {
                "game_pk": 101.0,
                "is_home": False,
                "pythag_win_pct": 0.4,
                "roll_run_diff": -1,
                "roll_rs_per_game": 4,
                "roll_ra_per_game": 5,
            },
            {
                "game_pk": 102.0,
                "is_home": True,
                "pythag_win_pct": 0.5,
                "roll_run_diff": 0,
                "roll_rs_per_game": 4,
                "roll_ra_per_game": 4,
            },
            {
                "game_pk": 102.0,
                "is_home": False,
                "pythag_win_pct": 0.5,
                "roll_run_diff": 0,
                "roll_rs_per_game": 4,
                "roll_ra_per_game": 4,
            },
        ]
    )
    re24_df = pd.DataFrame(
        [
            {
                "player_id": 1.0,
                "game_date": pd.to_datetime("2023-04-01"),
                "role": "PITCHER",
                "roll_re24": 5.0,
            },
            {
                "player_id": 2.0,
                "game_date": pd.to_datetime("2023-04-01"),
                "role": "PITCHER",
                "roll_re24": 3.0,
            },
            {
                "player_id": 1.0,
                "game_date": "2023-04-02",
                "role": "PITCHER",
                "roll_re24": 4.0,
            },
            {
                "player_id": 2.0,
                "game_date": "2023-04-02",
                "role": "PITCHER",
                "roll_re24": 4.0,
            },
            {
                "player_id": 3.0,
                "game_date": "2023-04-01",
                "role": "BATTER",
                "roll_re24": 1.0,
            },
            {
                "player_id": 4.0,
                "game_date": "2023-04-01",
                "role": "BATTER",
                "roll_re24": 0.5,
            },
            {
                "player_id": 3.0,
                "game_date": pd.to_datetime("2023-04-02"),
                "role": "BATTER",
                "roll_re24": 0.8,
            },
            {
                "player_id": 4.0,
                "game_date": pd.to_datetime("2023-04-02"),
                "role": "BATTER",
                "roll_re24": 0.9,
            },
        ]
    )
    # Alignment: Force primitives
    for df in [games_df, pitcher_gold, lineups, batter_gold, re24_df]:
        df["game_date"] = pd.to_datetime(df["game_date"])
        for col in ["player_id", "game_pk", "home_pitcher_id", "away_pitcher_id"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

    X, y = pipeline.build_uranium_matrix(
        games_df, pitcher_gold, lineups, batter_gold, elo_df, pythag_df, re24_df
    )

    assert not X.empty
    assert len(X) == 2
    assert "h_bat_roll_pas" in X.columns
    assert "a_bat_roll_pas" in X.columns
    assert "elo_diff" in X.columns
    assert "pythag_diff" in X.columns
    assert "h_sp_roll_re24" in X.columns
    assert "a_sp_roll_re24" in X.columns
    assert "re24_sp_diff" in X.columns
    assert "h_bat_roll_re24" in X.columns
    assert "a_bat_roll_re24" in X.columns
    assert len(y) == 2


def test_finalize_features_no_label(pipeline):
    # Cover lines 340-341
    df = pd.DataFrame([{"h_sp_stat": 1.0, "a_sp_stat": 2.0}])
    X, y = pipeline._finalize_features(df)
    assert X.empty
    assert y.empty


def test_prepare_data_for_merge_branches(pipeline):
    # Cover lines 147-151, 154
    games = pd.DataFrame([{"game_pk": "100", "game_date": "2023-01-01"}])
    gold = pd.DataFrame([{"player_id": "1", "game_date": "2023-01-01"}])
    lineups = pd.DataFrame([{"game_pk": "100", "game_date": "2023-01-01"}])
    batter = pd.DataFrame([{"player_id": "1", "game_date": "2023-01-01"}])

    g, p, lineups_prep, b = pipeline._prepare_data_for_merge(
        games, gold, lineups, batter
    )
    assert isinstance(g.loc[0, "game_pk"], float)
    assert lineups_prep is not None
    assert b is not None


def test_build_uranium_matrix_empty(pipeline):
    # Cover lines 98-99
    X, y = pipeline.build_uranium_matrix(pd.DataFrame(), pd.DataFrame())
    assert X.empty
    assert y.empty
