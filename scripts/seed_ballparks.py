from algomlb.db.session import get_session_factory
from algomlb.db.models import BallparkORM


def seed_new_ballparks():
    session = get_session_factory()()

    # Check for Sutter Health Park
    sutter = (
        session.query(BallparkORM)
        .filter(BallparkORM.ballpark.ilike("%Sutter Health Park%"))
        .first()
    )
    if not sutter:
        print("Adding Sutter Health Park...")
        session.add(
            BallparkORM(
                ballpark="Sutter Health Park",
                team_name="OAK",
                city="West Sacramento",
                state="CA",
                latitude=38.5804,
                longitude=-121.5138,
                elevation=23,
            )
        )

    # Check for Las Vegas Ballpark
    lv = (
        session.query(BallparkORM)
        .filter(BallparkORM.ballpark.ilike("%Las Vegas Ballpark%"))
        .first()
    )
    if not lv:
        print("Adding Las Vegas Ballpark...")
        session.add(
            BallparkORM(
                ballpark="Las Vegas Ballpark",
                team_name="OAK",
                city="Summerlin",
                state="NV",
                latitude=36.1523,
                longitude=-115.3294,
                elevation=2900,
            )
        )

    session.commit()
    session.close()


if __name__ == "__main__":
    seed_new_ballparks()
