"""repartition statcast_raw across two tablespaces

Revision ID: 79fdb13f70c3
Revises: 863659fd3bcf
Create Date: 2026-07-23 15:02:49.950939

Recovery note (2026-07-23): statcast_raw previously had its storage manually
split across two disks via raw symlinks (base/16384/24796.{1,2,3,4} pointing at
a bind-mounted host directory) -- invisible to Postgres's own tablespace
catalog. That invisibility is exactly why the symlinked segments were mistaken
for orphaned files and deleted, corrupting the table. This migration replaces
that hack with a properly partitioned table using two real, catalog-visible
tablespaces (`pg_tablespace` / `pg_tblspc` will show both), so a future
investigation always sees the true picture instead of unexplained files.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '79fdb13f70c3'
down_revision: Union[str, Sequence[str], None] = '863659fd3bcf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ARCHIVE_TABLESPACE = "statcast_archive"
ARCHIVE_LOCATION = "/var/lib/postgresql/data/custom_tablespace"

# Older seasons live on the archive tablespace (root fs); recent seasons plus
# the default partition stay on the default tablespace (the /var/oled-backed
# Docker volume, where the rest of the DB already lives). This is a starting
# split -- partitions are normal relations, so rebalance later with
# ALTER TABLE statcast_raw_20XX SET TABLESPACE ... if one side fills up
# disproportionately once real per-partition sizes are known.
ARCHIVE_YEARS = range(2019, 2023)  # 2019-2022
DEFAULT_YEARS = range(2023, 2027)  # 2023-2026


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()

    # CREATE TABLESPACE cannot run inside a transaction block, so it's run on
    # a separate autocommit connection rather than through the migration's
    # normal transactional `op.execute`. Postgres has no CREATE TABLESPACE IF
    # NOT EXISTS, so guard it with an explicit check (this migration step is
    # otherwise safe to retry).
    existing_tablespaces = {
        row[0] for row in bind.exec_driver_sql("SELECT spcname FROM pg_tablespace").fetchall()
    }
    if ARCHIVE_TABLESPACE not in existing_tablespaces:
        autocommit_conn = bind.engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        )
        try:
            autocommit_conn.exec_driver_sql(
                f"CREATE TABLESPACE {ARCHIVE_TABLESPACE} LOCATION '{ARCHIVE_LOCATION}'"
            )
        finally:
            autocommit_conn.close()

    # statcast_raw's segment files were deleted (corrupted beyond in-place
    # repair); its data is fully re-ingested from the public Statcast source
    # after this migration runs, so dropping it here is safe. CASCADE also
    # drops the single dependent FK constraint on statcast_quant_features,
    # which is recreated below against the new (wider) primary key.
    op.execute("DROP TABLE IF EXISTS statcast_raw CASCADE")

    # Recreate from the ORM model's own metadata rather than hand-transcribing
    # 100+ columns -- StatcastRawORM (db/models.py) is the single source of
    # truth for this schema. Its __table_args__ now declares
    # postgresql_partition_by="RANGE (game_date)" and includes game_date in
    # the primary key, which Postgres requires for the partition key.
    from algomlb.db.models import StatcastRawORM

    StatcastRawORM.__table__.create(bind)

    for year in ARCHIVE_YEARS:
        op.execute(
            f"CREATE TABLE statcast_raw_{year} PARTITION OF statcast_raw "
            f"FOR VALUES FROM ('{year}-01-01') TO ('{year + 1}-01-01') "
            f"TABLESPACE {ARCHIVE_TABLESPACE}"
        )

    for year in DEFAULT_YEARS:
        op.execute(
            f"CREATE TABLE statcast_raw_{year} PARTITION OF statcast_raw "
            f"FOR VALUES FROM ('{year}-01-01') TO ('{year + 1}-01-01')"
        )

    # Catches anything outside the declared ranges (required so ingestion
    # never fails outright on an unexpected date) instead of erroring.
    op.execute("CREATE TABLE statcast_raw_default PARTITION OF statcast_raw DEFAULT")

    # Recreate the FK from statcast_quant_features that CASCADE dropped above,
    # widened to match statcast_raw's new (game_pk, at_bat_number,
    # pitch_number, game_date) primary key -- see the matching
    # ForeignKeyConstraint update on StatcastQuantFeatures in db/models.py.
    # Added NOT VALID because statcast_quant_features already has existing
    # rows referencing the old (now-empty, pre-reingestion) statcast_raw;
    # this still enforces the constraint for all new writes immediately.
    # Run `ALTER TABLE statcast_quant_features VALIDATE CONSTRAINT
    # statcast_quant_features_game_pk_at_bat_number_pitch_number_fkey;`
    # once re-ingestion has repopulated statcast_raw, to confirm/enforce it
    # retroactively against the existing rows too.
    op.execute(
        """
        ALTER TABLE statcast_quant_features
        ADD CONSTRAINT statcast_quant_features_game_pk_at_bat_number_pitch_number_fkey
        FOREIGN KEY (game_pk, at_bat_number, pitch_number, game_date)
        REFERENCES statcast_raw (game_pk, at_bat_number, pitch_number, game_date)
        ON DELETE CASCADE NOT VALID
        """
    )


def downgrade() -> None:
    """Downgrade schema.

    Note: this does not and cannot restore lost data -- the original
    statcast_raw table was already corrupted (and dropped by this migration's
    upgrade()) before any of this ran. This only reverts the structural
    change back to a plain, empty, non-partitioned table.
    """
    bind = op.get_bind()
    op.execute("DROP TABLE IF EXISTS statcast_raw")

    # DROP TABLESPACE also cannot run inside a transaction block.
    autocommit_conn = bind.engine.connect().execution_options(
        isolation_level="AUTOCOMMIT"
    )
    try:
        autocommit_conn.exec_driver_sql(
            f"DROP TABLESPACE IF EXISTS {ARCHIVE_TABLESPACE}"
        )
    finally:
        autocommit_conn.close()

    from algomlb.db.models import StatcastRawORM

    StatcastRawORM.__table__.create(bind)
