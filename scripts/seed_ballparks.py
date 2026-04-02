from algomlb.db.session import get_session_factory
from algomlb.db.models import BallparkORM


def seed_new_ballparks():
    session = get_session_factory()()

    # Define the new ballparks data
    new_parks = [
        {
            "ballpark": "Sutter Health Park",
            "team_name": "OAK",
            "city": "West Sacramento",
            "state": "CA",
            "latitude": 38.5772,
            "longitude": -121.5222,
            "elevation": 23,
            "left_field": 330,
            "center_field": 403,
            "right_field": 325,
            "min_wall_height": 8.0,
            "max_wall_height": 8.0,
            "hp_bearing_deg": 45.0,
            "extra_distance": 0.1,
            "hr_park_effects": 100.0,
        },
        {
            "ballpark": "Las Vegas Ballpark",
            "team_name": "OAK",
            "city": "Summerlin South",
            "state": "NV",
            "latitude": 36.1523,
            "longitude": -115.3294,
            "elevation": 2900,
            "left_field": 340,
            "center_field": 415,
            "right_field": 340,
            "min_wall_height": 8.0,
            "max_wall_height": 14.0,
            "hp_bearing_deg": 67.5,
            "extra_distance": 17.1,
            "hr_park_effects": 100.0,
        },
    ]

    for park_data in new_parks:
        existing = (
            session.query(BallparkORM)
            .filter(BallparkORM.ballpark.ilike(f"%{park_data['ballpark']}%"))
            .first()
        )
        if existing:
            print(f"Updating {park_data['ballpark']}...")
            for key, value in park_data.items():
                setattr(existing, key, value)
        else:
            print(f"Adding {park_data['ballpark']}...")
            session.add(BallparkORM(**park_data))

    session.commit()
    session.close()


if __name__ == "__main__":
    seed_new_ballparks()
