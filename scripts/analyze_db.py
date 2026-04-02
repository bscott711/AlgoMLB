from sqlalchemy import text
from algomlb.db.session import get_engine

engine = get_engine()
with engine.connect() as conn:
    weather_count = conn.execute(
        text("SELECT count(*) FROM openmeteo_weather_progression")
    ).scalar()
    print(f"Total Weather Ingested: {weather_count:,}")

    # Progress per season
    query = """
        SELECT extract(year from g.game_date) as season, count(*) as count 
        FROM openmeteo_weather_progression w
        JOIN game_results g ON w.game_id = g.game_id
        GROUP BY 1 ORDER BY 1
    """
    res = conn.execute(text(query)).fetchall()
    for row in res:
        print(f"Season {row[0]}: {row[1]:,}")
