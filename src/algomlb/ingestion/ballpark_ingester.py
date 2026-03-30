import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session
from algomlb.db.models import BallparkORM


class BallparkIngester:
    """Ingests ballpark data from CSV into the database."""

    def __init__(self, session: Session):
        self.session = session

    def ingest_from_csv(self, csv_path: str):
        """Load ballpark CSV and merge into DB."""
        logger.info(f"Ingesting ballparks from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Standardize column names if needed (the CSV already matches our ORM)
        count = 0
        for _, row in df.iterrows():
            # Create or merge ORM object
            orm = BallparkORM(
                team_name=row["team_name"],
                ballpark=row["ballpark"],
                left_field=int(row["left_field"])
                if not pd.isna(row["left_field"])
                else None,
                center_field=int(row["center_field"])
                if not pd.isna(row["center_field"])
                else None,
                right_field=int(row["right_field"])
                if not pd.isna(row["right_field"])
                else None,
                min_wall_height=float(row["min_wall_height"])
                if not pd.isna(row["min_wall_height"])
                else None,
                max_wall_height=float(row["max_wall_height"])
                if not pd.isna(row["max_wall_height"])
                else None,
                hr_park_effects=float(row["hr_park_effects"])
                if not pd.isna(row["hr_park_effects"])
                else None,
                extra_distance=float(row["extra_distance"])
                if not pd.isna(row["extra_distance"])
                else None,
                avg_temp=float(row["avg_temp"])
                if not pd.isna(row["avg_temp"])
                else None,
                elevation=int(row["elevation"])
                if not pd.isna(row["elevation"])
                else None,
                roof=float(row["roof"]) if not pd.isna(row["roof"]) else None,
                daytime=float(row["daytime"]) if not pd.isna(row["daytime"]) else None,
            )
            self.session.merge(orm)
            count += 1

        self.session.commit()
        logger.success(f"Successfully ingested {count} ballparks.")
