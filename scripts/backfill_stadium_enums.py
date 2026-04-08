import sys
import os

# Ensure we can import from algomlb
sys.path.insert(0, os.path.abspath("src"))

from algomlb.db.session import get_session_factory
from algomlb.db.models import BallparkORM
from algomlb.domain.models import SurfaceType, RoofType


def main():
    session_factory = get_session_factory()
    with session_factory() as session:
        # Define known values based on MLB stadiums (approximate rules)
        turf_parks = [
            "rogers centre",
            "tropicana field",
            "chase field",
            "loandepot park",
            "globe life field",
        ]
        fixed_roof_parks = ["tropicana field"]
        retractable_parks = [
            "rogers centre",
            "chase field",
            "t-mobile park",
            "american family field",
            "minute maid park",
            "loandepot park",
            "globe life field",
        ]

        count = 0
        for park in session.query(BallparkORM).all():
            name = park.ballpark.lower()

            # Map surface type
            if name in turf_parks:
                park.surface_type = SurfaceType.TURF
            else:
                park.surface_type = SurfaceType.GRASS

            # Map roof type
            if name in fixed_roof_parks:
                park.roof_type = RoofType.CLOSED
            elif name in retractable_parks:
                park.roof_type = RoofType.RETRACTABLE
            else:
                park.roof_type = RoofType.OPEN

            count += 1

        session.commit()
        print(f"✅ Successfully mapped SurfaceType and RoofType for {count} ballparks.")


if __name__ == "__main__":
    main()
