"""add missing sabermetrics and historical tables

Revision ID: fd59cba47450
Revises: 338106f037a3
Create Date: 2026-04-13 15:59:09.203477

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'fd59cba47450'
down_revision: Union[str, Sequence[str], None] = '338106f037a3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add the tables that were previously missing or stripped
    op.create_table('historical_player_stats',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('player_id', sa.Integer(), nullable=False),
    sa.Column('date', sa.Date(), nullable=False),
    sa.Column('metric_name', sa.String(length=50), nullable=False),
    sa.Column('metric_value', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_historical_player_stats_date'), 'historical_player_stats', ['date'], unique=False)
    op.create_index(op.f('ix_historical_player_stats_player_id'), 'historical_player_stats', ['player_id'], unique=False)
    
    op.create_table('team_sabermetrics_history',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('game_pk', sa.BigInteger(), nullable=False),
    sa.Column('game_date', sa.Date(), nullable=False),
    sa.Column('team_id', sa.String(length=50), nullable=False),
    sa.Column('is_home', sa.Boolean(), nullable=False),
    sa.Column('pythag_win_pct', sa.Float(), nullable=False),
    sa.Column('roll_run_diff', sa.Float(), nullable=False),
    sa.Column('roll_rs_per_game', sa.Float(), nullable=False),
    sa.Column('roll_ra_per_game', sa.Float(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('game_pk', 'team_id', 'is_home', name='uq_team_sabermetrics_history_row')
    )
    op.create_index(op.f('ix_team_sabermetrics_history_game_date'), 'team_sabermetrics_history', ['game_date'], unique=False)
    op.create_index(op.f('ix_team_sabermetrics_history_game_pk'), 'team_sabermetrics_history', ['game_pk'], unique=False)
    op.create_index(op.f('ix_team_sabermetrics_history_team_id'), 'team_sabermetrics_history', ['team_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_team_sabermetrics_history_team_id'), table_name='team_sabermetrics_history')
    op.drop_index(op.f('ix_team_sabermetrics_history_game_pk'), table_name='team_sabermetrics_history')
    op.drop_index(op.f('ix_team_sabermetrics_history_game_date'), table_name='team_sabermetrics_history')
    op.drop_table('team_sabermetrics_history')
    op.drop_index(op.f('ix_historical_player_stats_player_id'), table_name='historical_player_stats')
    op.drop_index(op.f('ix_historical_player_stats_date'), table_name='historical_player_stats')
    op.drop_table('historical_player_stats')
    # ### end Alembic commands ###
