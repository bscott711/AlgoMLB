from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import cast
from sqlalchemy import Engine, text


@dataclass(frozen=True)
class TableMeta:
    name: str
    row_count: int
    is_empty: bool
    schema: str = "public"


@dataclass(frozen=True)
class ColumnMeta:
    table: str
    name: str
    dtype: str
    nullable: bool
    null_pct: float
    is_all_null: bool


@dataclass(frozen=True)
class FKEdge:
    from_table: str
    from_col: str
    to_table: str
    to_col: str


@dataclass(frozen=True)
class FreshnessCheck:
    name: str
    latest: date | datetime | None
    max_age_days: float
    age_days: float | None
    ok: bool
    detail: str


# (table label, SQL for its most-recent date, allowed staleness in days)
_FRESHNESS_TABLE_CHECKS: list[tuple[str, str, float]] = [
    (
        "game_results (played)",
        "SELECT MAX(game_date) FROM game_results WHERE home_score IS NOT NULL",
        2,
    ),
    ("live_odds", "SELECT MAX(timestamp)::date FROM live_odds", 2),
    ("statcast_raw", "SELECT MAX(game_date) FROM statcast_raw", 3),
    ("player_rolling_features (gold)", "SELECT MAX(game_date) FROM player_rolling_features", 3),
    ("model_predictions", "SELECT MAX(game_date) FROM model_predictions", 2),
    ("clv_results", "SELECT MAX(game_date) FROM clv_results", 5),
]

# Daily job with ~24h cadence; 30h gives slack for RandomizedDelaySec/retries
# without masking a genuinely stuck pipeline.
_HEARTBEAT_MAX_AGE_HOURS = 30.0


def check_freshness(engine: Engine) -> list[FreshnessCheck]:
    """
    Check whether key tables have advanced recently enough, AND whether the
    daily sync job itself is still actually running.

    These are deliberately two different failure modes: a per-table date
    check alone cannot detect "the job never ran" (nothing to be stale, no
    rows touched at all) — that is exactly how the 2026-06-30 incident went
    undetected for three weeks while its systemd timer kept firing. The
    pipeline_heartbeat check catches that; the per-table checks catch a job
    that runs but silently stops making progress on one data source.
    """
    results: list[FreshnessCheck] = []
    today = datetime.now(timezone.utc).date()

    with engine.connect() as conn:
        for name, sql, max_age_days in _FRESHNESS_TABLE_CHECKS:
            try:
                latest = conn.execute(text(sql)).scalar()
            except Exception as e:
                results.append(
                    FreshnessCheck(name, None, max_age_days, None, False, f"query failed: {e}")
                )
                continue

            if latest is None:
                results.append(
                    FreshnessCheck(name, None, max_age_days, None, False, "no data")
                )
                continue

            latest_date = latest if isinstance(latest, date) else latest.date()
            age_days = (today - latest_date).days
            ok = age_days <= max_age_days
            results.append(
                FreshnessCheck(
                    name,
                    latest_date,
                    max_age_days,
                    float(age_days),
                    ok,
                    f"{age_days}d old (max {max_age_days}d)",
                )
            )

        try:
            hb = conn.execute(
                text(
                    "SELECT last_run_at, last_status, details FROM pipeline_heartbeat "
                    "WHERE job_name = 'sync_daily'"
                )
            ).fetchone()
        except Exception as e:
            hb = None
            hb_error = str(e)
        else:
            hb_error = None

        if hb is None:
            detail = f"query failed: {hb_error}" if hb_error else "never recorded"
            results.append(
                FreshnessCheck(
                    "sync_daily heartbeat",
                    None,
                    _HEARTBEAT_MAX_AGE_HOURS / 24,
                    None,
                    False,
                    detail,
                )
            )
        else:
            last_run_at, last_status, details = hb
            age_hours = (
                datetime.now(timezone.utc) - last_run_at
            ).total_seconds() / 3600
            ok = age_hours <= _HEARTBEAT_MAX_AGE_HOURS and last_status == "OK"
            detail = f"{age_hours:.1f}h since last run, status={last_status}"
            if details:
                detail += f" ({details})"
            results.append(
                FreshnessCheck(
                    "sync_daily heartbeat",
                    last_run_at,
                    _HEARTBEAT_MAX_AGE_HOURS / 24,
                    age_hours / 24,
                    ok,
                    detail,
                )
            )

    return results


class SchemaInspector:
    """Service to query live database schema metadata with TTL caching."""

    def __init__(self, engine: Engine, ttl_minutes: int = 5):
        self.engine = engine
        self.ttl = timedelta(minutes=ttl_minutes)
        self._table_cache: tuple[datetime, list[TableMeta]] | None = None
        self._column_cache: dict[str, tuple[datetime, list[ColumnMeta]]] = {}
        self._fk_cache: tuple[datetime, list[FKEdge]] | None = None

    def _is_expired(self, timestamp: datetime) -> bool:
        return datetime.now() - timestamp > self.ttl

    def list_tables(self) -> list[TableMeta]:
        """Returns all tables in the public schema with accurate row counts."""
        if self._table_cache and not self._is_expired(self._table_cache[0]):
            return self._table_cache[1]

        # 1. Get all user table names in the public schema
        name_query = text("""
            SELECT tablename 
            FROM pg_catalog.pg_tables 
            WHERE schemaname = 'public';
        """)

        with self.engine.connect() as conn:
            names = conn.execute(name_query).fetchall()
            tables = []

            for row in names:
                table_name = cast(str, row[0])
                # 2. Perform direct COUNT(*) for "Ground Truth" row counts
                # Using a sub-query or simple select is fine for the small AlgoMLB schema
                count_query = text(f'SELECT COUNT(*) FROM "{table_name}"')
                try:
                    count = conn.execute(count_query).scalar()
                    count = int(count) if count is not None else 0
                except Exception:
                    count = 0

                tables.append(
                    TableMeta(name=table_name, row_count=count, is_empty=(count <= 0))
                )

            tables.sort(key=lambda t: t.name)
            self._table_cache = (datetime.now(), tables)
            return tables

    def column_report(self, table: str) -> list[ColumnMeta]:
        """Generates detailed column report for a table including NULL percentages."""
        if table in self._column_cache:
            ts, cols = self._column_cache[table]
            if not self._is_expired(ts):
                return cols

        # 1. Get basic metadata from information_schema
        meta_query = text("""
            SELECT 
                column_name, 
                data_type, 
                is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' 
              AND table_name = :table_name
            ORDER BY ordinal_position;
        """)

        columns = []
        with self.engine.connect() as conn:
            base_info = conn.execute(meta_query, {"table_name": table}).fetchall()

            if not base_info:
                return []

            # 2. Get null percentages for each column
            null_cols_sql = ", ".join(
                [
                    f"COUNT(*) FILTER (WHERE {row.column_name} IS NULL)::float / NULLIF(COUNT(*), 0) AS {row.column_name}_null_pct"
                    for row in base_info
                ]
            )

            stats_query = text(f'SELECT {null_cols_sql} FROM "{table}"')

            try:
                stats_row = conn.execute(stats_query).fetchone()
            except Exception:
                stats_row = None

            for i, row in enumerate(base_info):
                null_pct = 0.0
                if stats_row and stats_row[i] is not None:
                    null_pct = float(stats_row[i])

                columns.append(
                    ColumnMeta(
                        table=table,
                        name=row.column_name,
                        dtype=row.data_type,
                        nullable=(row.is_nullable == "YES"),
                        null_pct=null_pct,
                        is_all_null=(null_pct >= 1.0),
                    )
                )

        self._column_cache[table] = (datetime.now(), columns)
        return columns

    def foreign_keys(self) -> list[FKEdge]:
        """Returns all foreign key relationships in the schema."""
        if self._fk_cache and not self._is_expired(self._fk_cache[0]):
            return self._fk_cache[1]

        query = text("""
            SELECT
                tc.table_name     AS from_table,
                kcu.column_name   AS from_col,
                ccu.table_name    AS to_table,
                ccu.column_name   AS to_col
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = 'public';
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query)
            fks = [FKEdge(**row._asdict()) for row in result]
            self._fk_cache = (datetime.now(), fks)
            return fks

    def empty_tables(self) -> list[TableMeta]:
        """Returns tables where row_count == 0."""
        return [t for t in self.list_tables() if t.is_empty]

    def all_null_columns(self) -> list[ColumnMeta]:
        """Scan all tables and return columns that are 100% NULL."""
        tables = self.list_tables()
        violations = []
        for table in tables:
            if table.is_empty:
                continue
            cols = self.column_report(table.name)
            violations.extend([c for c in cols if c.is_all_null])
        return violations

    def clear_cache(self):
        self._table_cache = None
        self._column_cache = {}
        self._fk_cache = None
