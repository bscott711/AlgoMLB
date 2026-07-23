"""drop_historical_data_table

Revision ID: 7d0e646bb164
Revises: c2cd87f58126
Create Date: 2026-04-11 17:26:23.275839

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7d0e646bb164'
down_revision: Union[str, Sequence[str], None] = 'c2cd87f58126'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop legacy historical_data table."""
    op.drop_table("historical_data")


def downgrade() -> None:
    """Recreate legacy historical_data table."""
    op.create_table(
        "historical_data",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("metric_name", sa.String(length=50), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_historical_data_date", "historical_data", ["date"], unique=False)
    op.create_index(
        "ix_historical_data_player_id", "historical_data", ["player_id"], unique=False
    )
