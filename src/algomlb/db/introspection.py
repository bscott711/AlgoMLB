from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast
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


class SchemaInspector:
    """ Service to query live database schema metadata with TTL caching. """

    def __init__(self, engine: Engine, ttl_minutes: int = 5):
        self.engine = engine
        self.ttl = timedelta(minutes=ttl_minutes)
        self._table_cache: tuple[datetime, list[TableMeta]] | None = None
        self._column_cache: dict[str, tuple[datetime, list[ColumnMeta]]] = {}
        self._fk_cache: tuple[datetime, list[FKEdge]] | None = None

    def _is_expired(self, timestamp: datetime) -> bool:
        return datetime.now() - timestamp > self.ttl

    def list_tables(self) -> list[TableMeta]:
        """Returns all tables in the public schema with row counts."""
        if self._table_cache and not self._is_expired(self._table_cache[0]):
            return self._table_cache[1]

        query = text("""
            SELECT 
                relname AS table_name, 
                n_live_tup AS row_count
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
            ORDER BY relname;
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query)
            tables = []
            for row in result:
                table_name = cast(str, row[0])
                count = int(row[1])
                tables.append(TableMeta(
                    name=table_name,
                    row_count=count,
                    is_empty=(count <= 0)
                ))
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
            null_cols_sql = ", ".join([
                f"COUNT(*) FILTER (WHERE {row.column_name} IS NULL)::float / NULLIF(COUNT(*), 0) AS {row.column_name}_null_pct"
                for row in base_info
            ])
            
            stats_query = text(f"SELECT {null_cols_sql} FROM \"{table}\"")
            
            try:
                stats_row = conn.execute(stats_query).fetchone()
            except Exception:
                stats_row = None

            for i, row in enumerate(base_info):
                null_pct = 0.0
                if stats_row and stats_row[i] is not None:
                    null_pct = float(stats_row[i])
                
                columns.append(ColumnMeta(
                    table=table,
                    name=row.column_name,
                    dtype=row.data_type,
                    nullable=(row.is_nullable == "YES"),
                    null_pct=null_pct,
                    is_all_null=(null_pct >= 1.0)
                ))

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
