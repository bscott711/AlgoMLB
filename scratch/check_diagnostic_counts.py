from algomlb.db.session import get_engine
from sqlalchemy import text


def check_counts():
    engine = get_engine()
    tables = ["uranium_eval_history", "uranium_calibration_bins", "uranium_shap_global"]
    with engine.connect() as conn:
        for table in tables:
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                print(f"{table}: {count}")
            except Exception as e:
                print(f"{table}: Error {e}")


if __name__ == "__main__":
    check_counts()
