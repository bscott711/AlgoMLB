"""add clv_results table

Revision ID: cca8107f5621
Revises: 79fdb13f70c3
Create Date: 2026-07-23 23:33:58.668764

Note: autogenerate also detected a large amount of pre-existing, unrelated
schema drift (statcast_raw_* partition tables, fadegoblin_slips, etc. showing
as "removed" because they aren't declared in SQLAlchemy metadata as partitions
/ are managed outside Alembic). That drift was deliberately excluded from this
migration — it is not something we want a `clv_results` migration to touch.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'cca8107f5621'
down_revision: Union[str, Sequence[str], None] = '79fdb13f70c3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'clv_results',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('game_id', sa.String(length=50), nullable=False),
        sa.Column('game_date', sa.Date(), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False),
        sa.Column('model_prob', sa.Float(), nullable=False),
        sa.Column('entry_implied', sa.Float(), nullable=False),
        sa.Column('closing_implied', sa.Float(), nullable=False),
        sa.Column('num_books_at_close', sa.Integer(), nullable=False),
        sa.Column('closing_snapshot_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('clv', sa.Float(), nullable=False),
        sa.Column('closing_edge', sa.Float(), nullable=False),
        sa.Column('computed_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('game_id', 'model_version', name='uq_clv_results_game_model'),
    )
    op.create_index(
        op.f('ix_clv_results_game_date'), 'clv_results', ['game_date'], unique=False
    )
    op.create_index(
        op.f('ix_clv_results_game_id'), 'clv_results', ['game_id'], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_clv_results_game_id'), table_name='clv_results')
    op.drop_index(op.f('ix_clv_results_game_date'), table_name='clv_results')
    op.drop_table('clv_results')
