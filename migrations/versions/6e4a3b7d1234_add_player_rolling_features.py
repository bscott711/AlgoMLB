"""add_player_rolling_features

Revision ID: 6e4a3b7d1234
Revises: b2e290580bc8
Create Date: 2026-04-03 18:54:02.506884

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '6e4a3b7d1234'
down_revision: Union[str, Sequence[str], None] = 'b2e290580bc8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Handle PostgreSQL Enums
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for enum_name, values in [
            ("playerrole", ("PITCHER", "BATTER")),
            ("baselinequality", ("COLD_START", "PARTIAL", "FULL")),
        ]:
            # Refined check using pg_type and pg_namespace (public schema)
            exists = bind.execute(
                sa.text(
                    f"SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace "
                    f"WHERE t.typname = '{enum_name}' AND n.nspname = 'public'"
                )
            ).fetchone()
            if not exists:
                sa.Enum(*values, name=enum_name).create(bind)

    # 2. Drop the old placeholder table (if it exists)
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "player_rolling_features" in inspector.get_table_names():
        op.drop_table("player_rolling_features")

    # 3. Create the new wide-column Gold table
    op.create_table(
        "player_rolling_features",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("game_date", sa.Date(), nullable=False),
        sa.Column("season", sa.SmallInteger(), nullable=False),
        sa.Column(
            "role",
            postgresql.ENUM("PITCHER", "BATTER", name="playerrole", create_type=False),
            nullable=False,
        ),
        # Window Metadata
        sa.Column("window_games", sa.SmallInteger(), nullable=False),
        sa.Column("n_games_used", sa.SmallInteger(), nullable=True),
        sa.Column("days_since_last_game", sa.SmallInteger(), nullable=True),
        sa.Column(
            "baseline_quality",
            postgresql.ENUM(
                "cold_start", "partial", "full", name="baselinequality", create_type=False
            ),
            nullable=False,
        ),
        sa.Column(
            "shrinkage_applied", sa.Boolean(), server_default="false", nullable=False
        ),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # PITCHER Features
        sa.Column("roll_pitches", sa.Float(), nullable=True),
        sa.Column("roll_strikes_pct", sa.Float(), nullable=True),
        sa.Column("roll_whiff_pct", sa.Float(), nullable=True),
        sa.Column("roll_k_pct", sa.Float(), nullable=True),
        sa.Column("roll_bb_pct", sa.Float(), nullable=True),
        sa.Column("roll_avg_release_speed", sa.Float(), nullable=True),
        sa.Column("roll_avg_pfx_x", sa.Float(), nullable=True),
        sa.Column("roll_avg_pfx_z", sa.Float(), nullable=True),
        sa.Column("roll_avg_pitcher_xwoba", sa.Float(), nullable=True),
        sa.Column("roll_pitcher_xwoba_shrunk", sa.Float(), nullable=True),
        # BATTER Features
        sa.Column("roll_pas", sa.Float(), nullable=True),
        sa.Column("roll_hits_per_pa", sa.Float(), nullable=True),
        sa.Column("roll_k_pct_batter", sa.Float(), nullable=True),
        sa.Column("roll_bb_pct_batter", sa.Float(), nullable=True),
        sa.Column("roll_barrel_pct", sa.Float(), nullable=True),
        sa.Column("roll_avg_launch_speed", sa.Float(), nullable=True),
        sa.Column("roll_avg_launch_angle", sa.Float(), nullable=True),
        sa.Column("roll_avg_batter_xwoba", sa.Float(), nullable=True),
        sa.Column("roll_batter_xwoba_shrunk", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "player_id", "game_date", "role", name="uq_player_rolling_features"
        ),
    )

    # 4. Create Indexes
    op.create_index(
        "ix_prf_player_id", "player_rolling_features", ["player_id"], unique=False
    )
    op.create_index(
        "ix_prf_game_date", "player_rolling_features", ["game_date"], unique=False
    )
    op.create_index(
        "ix_prf_season", "player_rolling_features", ["season"], unique=False
    )
    op.create_index(
        "ix_prf_date_role", "player_rolling_features", ["game_date", "role"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("player_rolling_features")

    # Recreate the old placeholder table
    op.create_table(
        "player_rolling_features",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("feature_name", sa.String(length=50), nullable=False),
        sa.Column("feature_value", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_player_rolling_features_date",
        "player_rolling_features",
        ["date"],
        unique=False,
    )
    op.create_index(
        "ix_player_rolling_features_player_id",
        "player_rolling_features",
        ["player_id"],
        unique=False,
    )
