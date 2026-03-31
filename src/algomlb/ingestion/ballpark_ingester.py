import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session
from algomlb.db.models import BallparkORM


class BallparkIngester:
    """Ingests ballpark data from CSV into the database."""

    def __init__(self, session: Session):
        self.session = session

    def ingest_from_csv(self, csv_path: str):
        """Load ballpark CSV and merge with geographic JSON into DB."""
        logger.info(f"Ingesting ballparks from {csv_path}...")

        synonym_map = self._load_geo_map()
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        existing_ballparks = {
            b.ballpark.strip().lower(): b for b in self.session.query(BallparkORM).all()
        }

        count = 0
        for _, row in df.iterrows():
            csv_park_name = str(
                row["ballpark " if "ballpark " in row else "ballpark"]
            ).strip()

            orm = self._find_or_create_orm(
                csv_park_name, existing_ballparks, synonym_map
            )

            # 1. Update structural data from CSV
            self._update_structural_data(orm, row)

            # 2. Update geographic data from JSON
            self._update_geographic_data(orm, csv_park_name, synonym_map)

            count += 1

        self.session.commit()
        logger.success(f"Successfully processed {count} ballparks.")

    def _load_geo_map(self) -> dict:
        """Load geographic mapping and build synonym lookup."""
        import json
        import os

        json_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "ballpark_locations.json"
        )
        with open(json_path, "r") as f:
            geo_map = json.load(f)

        synonym_map = {}
        for _, info in geo_map.items():
            for syn in info["synonyms"]:
                synonym_map[syn.lower().strip()] = info
        return synonym_map

    def _find_or_create_orm(
        self, park_name: str, existing_map: dict, synonym_map: dict
    ) -> BallparkORM:
        """Match existing ballpark or create new record."""
        orm = existing_map.get(park_name.lower())
        if not orm:
            geo_info = synonym_map.get(park_name.lower())
            if geo_info:
                orm = existing_map.get(geo_info["ballpark"].lower())

            if not orm:
                logger.info(f"Creating new ballpark record for: {park_name}")
                orm = BallparkORM(ballpark=park_name)
                self.session.add(orm)
        return orm

    def _update_structural_data(self, orm: BallparkORM, row: pd.Series):
        """Update ballpark dimensions and environmental factors from CSV."""
        orm.team_name = str(row["team_name"]).strip()
        orm.left_field = int(row["left_field"]) if pd.notna(row["left_field"]) else None
        orm.center_field = (
            int(row["center_field"]) if pd.notna(row["center_field"]) else None
        )
        orm.right_field = (
            int(row["right_field"]) if pd.notna(row["right_field"]) else None
        )
        orm.min_wall_height = (
            float(row["min_wall_height"]) if pd.notna(row["min_wall_height"]) else None
        )
        orm.max_wall_height = (
            float(row["max_wall_height"]) if pd.notna(row["max_wall_height"]) else None
        )
        orm.hr_park_effects = (
            float(row["hr_park_effects"]) if pd.notna(row["hr_park_effects"]) else None
        )
        orm.extra_distance = (
            float(row["extra_distance"]) if pd.notna(row["extra_distance"]) else None
        )
        orm.avg_temp = float(row["avg_temp"]) if pd.notna(row["avg_temp"]) else None
        orm.elevation = int(row["elevation"]) if pd.notna(row["elevation"]) else None
        orm.roof = float(row["roof"]) if pd.notna(row["roof"]) else None
        orm.daytime = float(row["daytime"]) if pd.notna(row["daytime"]) else None

    def _update_geographic_data(
        self, orm: BallparkORM, park_name: str, synonym_map: dict
    ):
        """Update location data from geographic source of truth."""
        geo_info = synonym_map.get(park_name.lower())
        if geo_info:
            orm.city = geo_info["city"]
            orm.state = geo_info["state"]
            orm.latitude = geo_info["lat"]
            orm.longitude = geo_info["long"]
        else:
            logger.warning(f"No geographic metadata found for: {park_name}")
