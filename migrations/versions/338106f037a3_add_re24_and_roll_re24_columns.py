"""add re24 and roll_re24 columns

Revision ID: 338106f037a3
Revises: 4fe0de4e2cb0
Create Date: 2026-04-13 07:21:26.568073

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '338106f037a3'
down_revision: Union[str, Sequence[str], None] = '4fe0de4e2cb0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Only keep the intended RE24 columns for this task to avoid conflicts with existing dirty DB state
    op.add_column('player_rolling_features', sa.Column('roll_re24', sa.Float(), nullable=True, comment='Trailing 30-game sum/avg of Run Expectancy Added'))
    op.add_column('statcast_player_game_logs', sa.Column('re24', sa.Float(), nullable=True, comment='Atomic situational added-value for this game'))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('statcast_player_game_logs', 're24')
    op.drop_column('player_rolling_features', 'roll_re24')
