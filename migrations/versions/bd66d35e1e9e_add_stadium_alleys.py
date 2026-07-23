"""add_stadium_alleys

Revision ID: bd66d35e1e9e
Revises: 6e4a3b7d1234
Create Date: 2026-04-03 21:12:22.090053

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bd66d35e1e9e'
down_revision: Union[str, Sequence[str], None] = '6e4a3b7d1234'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Only adding the new ballpark columns to avoid drift/enum issues
    op.add_column('ballparks', sa.Column('left_center', sa.Integer(), nullable=True))
    op.add_column('ballparks', sa.Column('right_center', sa.Integer(), nullable=True))
    op.add_column('ballparks', sa.Column('lf_wall_height', sa.Float(), nullable=True))
    op.add_column('ballparks', sa.Column('lc_wall_height', sa.Float(), nullable=True))
    op.add_column('ballparks', sa.Column('cf_wall_height', sa.Float(), nullable=True))
    op.add_column('ballparks', sa.Column('rc_wall_height', sa.Float(), nullable=True))
    op.add_column('ballparks', sa.Column('rf_wall_height', sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('ballparks', 'rf_wall_height')
    op.drop_column('ballparks', 'rc_wall_height')
    op.drop_column('ballparks', 'cf_wall_height')
    op.drop_column('ballparks', 'lc_wall_height')
    op.drop_column('ballparks', 'lf_wall_height')
    op.drop_column('ballparks', 'right_center')
    op.drop_column('ballparks', 'left_center')
