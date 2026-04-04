import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sqlalchemy import select
from algomlb.db.session import get_session_factory
from algomlb.db.models import BallparkORM

SessionLocal = get_session_factory()

# Official 2024-2025 Stadium Dimensions (LF, LC, CF, RC, RF) + Heights
# Note: LC/RC are Power Alleys (approx 22.5 deg from foul lines)
# Heights are in feet.
STADIUM_DATA = {
    "Fenway Park": {
        "lf": 310,
        "lc": 379,
        "cf": 390,
        "rc": 380,
        "rf": 302,
        "h_lf": 37.0,
        "h_lc": 37.0,
        "h_cf": 17.0,
        "h_rc": 3.0,
        "h_rf": 3.0,
    },
    "Oriole Park at Camden Yards": {
        "lf": 384,
        "lc": 405,
        "cf": 400,
        "rc": 373,
        "rf": 318,
        "h_lf": 12.0,
        "h_lc": 12.0,
        "h_cf": 7.0,
        "h_rc": 21.0,
        "h_rf": 21.0,
    },
    "Oracle Park": {
        "lf": 339,
        "lc": 364,
        "cf": 391,
        "rc": 415,
        "rf": 309,
        "h_lf": 8.0,
        "h_lc": 8.0,
        "h_cf": 8.0,
        "h_rc": 24.0,
        "h_rf": 24.0,
    },
    "Coors Field": {
        "lf": 347,
        "lc": 390,
        "cf": 415,
        "rc": 375,
        "rf": 350,
        "h_lf": 8.0,
        "h_lc": 8.0,
        "h_cf": 8.0,
        "h_rc": 14.0,
        "h_rf": 14.0,
    },
    "Dodger Stadium": {
        "lf": 330,
        "lc": 375,
        "cf": 400,
        "rc": 375,
        "rf": 330,
        "h_lf": 8.0,
        "h_lc": 8.0,
        "h_cf": 8.0,
        "h_rc": 8.0,
        "h_rf": 8.0,
    },
    "Sutter Health Park": {  # A's 2025 Temporary Home
        "lf": 330,
        "lc": 375,
        "cf": 403,
        "rc": 375,
        "rf": 325,
        "h_lf": 8.0,
        "h_lc": 8.0,
        "h_cf": 8.0,
        "h_rc": 8.0,
        "h_rf": 8.0,
    },
    "Yankee Stadium": {
        "lf": 318,
        "lc": 399,
        "cf": 408,
        "rc": 385,
        "rf": 314,
        "h_lf": 8.0,
        "h_lc": 8.0,
        "h_cf": 8.0,
        "h_rc": 8.0,
        "h_rf": 8.0,
    },
    "Minute Maid Park": {
        "lf": 315,
        "lc": 362,
        "cf": 409,
        "rc": 373,
        "rf": 326,
        "h_lf": 19.0,
        "h_lc": 7.0,
        "h_cf": 7.0,
        "h_rc": 7.0,
        "h_rf": 7.0,
    },
    # Default fallback for others (approx symmetric)
}


def populate():
    with SessionLocal() as session:
        # Get all current parks
        res = session.execute(select(BallparkORM)).scalars().all()

        for park in res:
            data = STADIUM_DATA.get(park.ballpark)
            if not data:
                # Estimate for missing parks (LF/CF/RF are already in DB)
                lf = park.left_field or 330
                cf = park.center_field or 400
                rf = park.right_field or 330

                # Alleys are approx (pole + center) / 1.95 for curvature
                lc = int((lf + cf) / 1.95)
                rc = int((rf + cf) / 1.95)

                # Standard wall height is 8ft
                data = {
                    "lf": lf,
                    "lc": lc,
                    "cf": cf,
                    "rc": rc,
                    "rf": rf,
                    "h_lf": 8.0,
                    "h_lc": 8.0,
                    "h_cf": 8.0,
                    "h_rc": 8.0,
                    "h_rf": 8.0,
                }

            # Perform update
            park.left_center = data["lc"]
            park.right_center = data["rc"]
            park.lf_wall_height = data["h_lf"]
            park.lc_wall_height = data["h_lc"]
            park.cf_wall_height = data["h_cf"]
            park.rc_wall_height = data["h_rc"]
            park.rf_wall_height = data["h_rf"]

        session.commit()
        print(f"Populated dimensions for {len(res)} ballparks.")


if __name__ == "__main__":
    populate()
