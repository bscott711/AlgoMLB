"""expand_statcast_schema

Revision ID: d952e9a5a9ca
Revises: ab4376df88e8
Create Date: 2026-04-03 03:05:52.972037

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd952e9a5a9ca'
down_revision: Union[str, Sequence[str], None] = 'ab4376df88e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('statcast_raw', sa.Column('at_bat_number_1', sa.SmallInteger(), nullable=True))
    op.add_column('statcast_raw', sa.Column('bat_speed', sa.Numeric(precision=5, scale=2), nullable=True))
    op.add_column('statcast_raw', sa.Column('blue_color', sa.String(length=20), nullable=True))
    op.add_column('statcast_raw', sa.Column('delta_home_win_exp', sa.Numeric(precision=5, scale=3), nullable=True))
    op.add_column('statcast_raw', sa.Column('delta_run_exp', sa.Numeric(precision=5, scale=3), nullable=True))
    op.add_column('statcast_raw', sa.Column('des', sa.Text(), nullable=True))
    op.add_column('statcast_raw', sa.Column('home_team_1', sa.String(length=3), nullable=True))
    op.add_column('statcast_raw', sa.Column('industrial_color', sa.String(length=20), nullable=True))
    op.add_column('statcast_raw', sa.Column('pitch_name', sa.String(length=50), nullable=True))
    op.add_column('statcast_raw', sa.Column('pitcher_1', sa.Integer(), nullable=True))
    op.add_column('statcast_raw', sa.Column('post_away_score', sa.SmallInteger(), nullable=True))
    op.add_column('statcast_raw', sa.Column('post_bat_score', sa.SmallInteger(), nullable=True))
    op.add_column('statcast_raw', sa.Column('post_fld_score', sa.SmallInteger(), nullable=True))
    op.add_column('statcast_raw', sa.Column('post_home_score', sa.SmallInteger(), nullable=True))
    op.add_column('statcast_raw', sa.Column('spin_rate_deprecated', sa.Integer(), nullable=True))
    op.add_column('statcast_raw', sa.Column('swing_length', sa.Numeric(precision=5, scale=2), nullable=True))
    op.add_column('statcast_raw', sa.Column('hit_distance_sc', sa.Numeric(precision=5, scale=1), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('statcast_raw', 'hit_distance_sc')
    op.drop_column('statcast_raw', 'swing_length')
    op.drop_column('statcast_raw', 'spin_rate_deprecated')
    op.drop_column('statcast_raw', 'post_home_score')
    op.drop_column('statcast_raw', 'post_fld_score')
    op.drop_column('statcast_raw', 'post_bat_score')
    op.drop_column('statcast_raw', 'post_away_score')
    op.drop_column('statcast_raw', 'pitcher_1')
    op.drop_column('statcast_raw', 'pitch_name')
    op.drop_column('statcast_raw', 'industrial_color')
    op.drop_column('statcast_raw', 'home_team_1')
    op.drop_column('statcast_raw', 'des')
    op.drop_column('statcast_raw', 'delta_run_exp')
    op.drop_column('statcast_raw', 'delta_home_win_exp')
    op.drop_column('statcast_raw', 'blue_color')
    op.drop_column('statcast_raw', 'bat_speed')
    op.drop_column('statcast_raw', 'at_bat_number_1')
