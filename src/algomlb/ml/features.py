import pandas as pd


class FeaturePipeline:
    """Handles data transformation and training matrix construction."""

    def build_training_matrix(
        self,
        games_df: pd.DataFrame,
        pitching_df: pd.DataFrame,
        batting_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Merge performance stats onto historical games for modeling.

        Args:
            games_df: DataFrame with historical outcomes [team_h, team_a, home_win]
            pitching_df: DataFrame with pitching stats per team [team, era, so, etc.]
            batting_df: DataFrame with batting stats per team [team, avg, ops, etc.]

        Returns:
            X: Feature matrix
            y: Target series (home_win)
        """
        # Ensure consistent column naming for merging
        # Assuming pitching_df and batting_df have a 'team' column

        # 1. Merge Home Team Batting
        df = games_df.merge(
            batting_df.add_prefix("h_"), left_on="team_h", right_on="h_team", how="left"
        )

        # 2. Merge Away Team Pitching
        df = df.merge(
            pitching_df.add_prefix("a_"),
            left_on="team_a",
            right_on="a_team",
            how="left",
        )

        # y is the home_win target
        y = df["home_win"].astype(int)

        # X is the numeric feature set
        # Drop non-numeric/ID columns
        cols_to_drop = [
            "team_h",
            "team_a",
            "home_win",
            "h_team",
            "a_team",
            "game_id",
            "date",
        ]
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # Ensure only numeric features remain
        X = X.select_dtypes(include=["number"]).fillna(0)

        return X, y
