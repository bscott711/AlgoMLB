"""add_game_type_to_statcast_raw

Revision ID: 1e9e29e72de1
Revises: d952e9a5a9ca
Create Date: 2026-04-03 03:14:58.193610

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1e9e29e72de1'
down_revision: Union[str, Sequence[str], None] = 'd952e9a5a9ca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('statcast_raw', sa.Column('game_type', sa.String(length=5), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('statcast_raw', 'game_type')
