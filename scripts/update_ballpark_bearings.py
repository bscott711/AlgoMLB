from algomlb.db.session import get_session_factory
from algomlb.db.models import BallparkORM
from algomlb.domain.stadium_bearings import STADIUM_HP_BEARINGS
from loguru import logger


def update_ballpark_bearings():
    session = get_session_factory()()
    ballparks = session.query(BallparkORM).all()

    count = 0
    updated = 0

    # Naming map from BallparkIngester for consistency
    naming_map = {
        "oriole_park_at_camden_yards": "camden_yards",
        "oracle_park": "at_t_park",
        "t-mobile_park": "safeco_field",
        "american_family_field": "miller_park",
        "guaranteed_rate_field": "guaranteed_rate_field",
        "oakland-alameda_county_coliseum": "oakland_coliseum",
        "t_mobile_park": "safeco_field",  # extra variant
    }

    # Case-insensitive lookup
    bearing_lookup = {k.lower(): v for k, v in STADIUM_HP_BEARINGS.items()}

    for bp in ballparks:
        count += 1
        # Try to resolve hp_bearing_deg
        norm_name = bp.ballpark.lower().strip().replace(" ", "_").replace("-", "_")
        mapped_name = naming_map.get(norm_name, norm_name)

        if mapped_name in bearing_lookup:
            bp.hp_bearing_deg = bearing_lookup[mapped_name]
            updated += 1
            logger.info(f"Updated {bp.ballpark}: {bp.hp_bearing_deg} deg")
        else:
            # Try synonym fallback
            logger.warning(
                f"Could not resolve bearing for: {bp.ballpark} ({norm_name})"
            )

    session.commit()
    session.close()
    logger.success(f"Processed {count} ballparks, updated {updated} with bearing data.")


if __name__ == "__main__":
    update_ballpark_bearings()
