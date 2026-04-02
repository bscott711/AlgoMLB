import datetime
from sqlalchemy import select, delete, func
from algomlb.db.session import get_session_factory
from algomlb.db.models import (
    GameResultORM,
    UmpireScorecardORM,
    RetrosheetEventORM,
    OpenMeteoWeatherProgressionORM,
    OpenMeteoDailyForecastORM,
    PitchEventORM,
    BankrollLedgerORM,
)


def cleanup_db():
    session_factory = get_session_factory()
    with session_factory() as session:
        # 1. Define Opening Days (approximate or known)
        opening_days = {
            2019: datetime.date(2019, 3, 20),
            2020: datetime.date(2020, 7, 23),
            2021: datetime.date(2021, 4, 1),
            2022: datetime.date(2022, 4, 7),
            2023: datetime.date(2023, 3, 30),
            2024: datetime.date(2024, 3, 28),
            2025: datetime.date(2025, 3, 27),
            2026: datetime.date(2026, 3, 26),
        }

        for year, od_date in opening_days.items():
            print(f"Checking {year} (Opening Day: {od_date})...")
            # Identify target game IDs
            target_ids_stmt = select(GameResultORM.game_id).where(
                GameResultORM.game_date < od_date,
                func.extract("year", GameResultORM.game_date) == year,
            )
            target_ids = session.execute(target_ids_stmt).scalars().all()

            if target_ids:
                print(f"Found {len(target_ids)} games to remove for {year}")
                # Delete dependencies first (ordered by likelihood of FK constraint)
                session.execute(
                    delete(OpenMeteoWeatherProgressionORM).where(
                        OpenMeteoWeatherProgressionORM.game_id.in_(target_ids)
                    )
                )
                session.execute(
                    delete(OpenMeteoDailyForecastORM).where(
                        OpenMeteoDailyForecastORM.game_id.in_(target_ids)
                    )
                )
                session.execute(
                    delete(PitchEventORM).where(PitchEventORM.game_id.in_(target_ids))
                )
                session.execute(
                    delete(BankrollLedgerORM).where(
                        BankrollLedgerORM.game_id.in_(target_ids)
                    )
                )
                session.execute(
                    delete(UmpireScorecardORM).where(
                        UmpireScorecardORM.game_id.in_(target_ids)
                    )
                )
                session.execute(
                    delete(RetrosheetEventORM).where(
                        RetrosheetEventORM.game_id.in_(target_ids)
                    )
                )
                # HistoricalOddsORM game_id is int in model, but let's try to match
                # HistoricalOddsORM.game_id: int - this might be a problem if it's not the same ID

                # Delete main games
                res = session.execute(
                    delete(GameResultORM).where(GameResultORM.game_id.in_(target_ids))
                )
                print(
                    f"Successfully deleted {res.rowcount} pre-opening day games for {year}"
                )

            # 2. All-Star games
            asg_dates = {
                2019: datetime.date(2019, 7, 9),
                2021: datetime.date(2021, 7, 13),
                2022: datetime.date(2022, 7, 19),
                2023: datetime.date(2023, 7, 11),
                2024: datetime.date(2024, 7, 16),
                2025: datetime.date(2025, 7, 15),
            }
            if year in asg_dates:
                asg_id_stmt = select(GameResultORM.game_id).where(
                    GameResultORM.game_date == asg_dates[year]
                )
                asg_ids = session.execute(asg_id_stmt).scalars().all()
                if asg_ids:
                    session.execute(
                        delete(OpenMeteoWeatherProgressionORM).where(
                            OpenMeteoWeatherProgressionORM.game_id.in_(asg_ids)
                        )
                    )
                    session.execute(
                        delete(OpenMeteoDailyForecastORM).where(
                            OpenMeteoDailyForecastORM.game_id.in_(asg_ids)
                        )
                    )
                    session.execute(
                        delete(PitchEventORM).where(PitchEventORM.game_id.in_(asg_ids))
                    )
                    session.execute(
                        delete(BankrollLedgerORM).where(
                            BankrollLedgerORM.game_id.in_(asg_ids)
                        )
                    )
                    session.execute(
                        delete(UmpireScorecardORM).where(
                            UmpireScorecardORM.game_id.in_(asg_ids)
                        )
                    )
                    session.execute(
                        delete(RetrosheetEventORM).where(
                            RetrosheetEventORM.game_id.in_(asg_ids)
                        )
                    )
                    res_asg = session.execute(
                        delete(GameResultORM).where(GameResultORM.game_id.in_(asg_ids))
                    )
                    print(
                        f"Successfully deleted {res_asg.rowcount} All-Star game for {year}"
                    )

        session.commit()
        print("Database cleanup complete.")


if __name__ == "__main__":
    cleanup_db()
