import pandas as pd


class FeaturePipeline:
    """Handles data transformation and training matrix construction."""

    def build_training_matrix(
        self,
        games_df: pd.DataFrame,
        historical_stats_df: pd.DataFrame,
        pitch_events_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Merge performance stats onto historical games for modeling.

        Args:
            games_df: DataFrame with historical outcomes and IDs.
            historical_stats_df: Pivoted DataFrame of HistoricalDataORM.
            pitch_events_df: (Optional) Aggregated Statcast metrics per pitcher.

        Returns:
            X: Feature matrix
            y: Target series (home_win)
        """
        if games_df.empty:
            return pd.DataFrame(), pd.Series()

        # 1. Enrich games with Home Pitcher stats
        df = games_df.merge(
            historical_stats_df.add_prefix("h_p_"),
            left_on=["home_pitcher_id"],
            right_on=["h_p_player_id"],
            how="left",
        )

        # 2. Enrich with Away Pitcher stats
        df = df.merge(
            historical_stats_df.add_prefix("a_p_"),
            left_on=["away_pitcher_id"],
            right_on=["a_p_player_id"],
            how="left",
        )

        # 3. Enrich with Statcast aggregates if available (complexity 1 -> real logic)
        if pitch_events_df is not None and not pitch_events_df.empty:
            # Join by pitcher/game to get point-in-time performance or Season-to-date
            # For simplicity, we just join by pitcher ID for now
            df = df.merge(
                pitch_events_df.add_prefix("sc_"),
                left_on=["home_pitcher_id"],
                right_on=["sc_pitcher_id"],
                how="left",
            )

        # Target Label: Home Win (Binary classification)
        if "home_score" in df.columns and "away_score" in df.columns:
            df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

        if "home_win" not in df.columns:
            return pd.DataFrame(), pd.Series()

        y = df["home_win"]

        # Final Feature Cleanup
        # Drop IDs, dates, and non-numeric labels
        drop_patterns = [
            "_id",
            "date",
            "team",
            "score",
            "status",
            "pitcher",
            "home_win",
        ]
        cols_to_drop = [
            c for c in df.columns if any(p in c.lower() for p in drop_patterns)
        ]

        X = df.drop(columns=cols_to_drop)

        # Ensure only numeric features remain and handle NaNs
        X = X.select_dtypes(include=["number"]).fillna(X.median()).fillna(0)

        # Drop any constant columns that might break the model
        X = X.loc[:, (X != X.iloc[0]).any()]

        return X, y
