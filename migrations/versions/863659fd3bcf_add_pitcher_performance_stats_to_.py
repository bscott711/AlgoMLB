"""add pitcher performance stats to manager_hook_events

Revision ID: 863659fd3bcf
Revises: fd59cba47450
Create Date: 2026-04-15 01:40:20.980593

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '863659fd3bcf'
down_revision: Union[str, Sequence[str], None] = 'fd59cba47450'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add pitcher performance columns to manager_hook_events."""
    op.add_column('manager_hook_events', sa.Column('runs_allowed', sa.Integer(), nullable=True))
    op.add_column('manager_hook_events', sa.Column('hits_allowed', sa.Integer(), nullable=True))
    op.add_column('manager_hook_events', sa.Column('walks_allowed', sa.Integer(), nullable=True))
    op.add_column('manager_hook_events', sa.Column('strikeouts', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Remove pitcher performance columns from manager_hook_events."""
    op.drop_column('manager_hook_events', 'strikeouts')
    op.drop_column('manager_hook_events', 'walks_allowed')
    op.drop_column('manager_hook_events', 'hits_allowed')
    op.drop_column('manager_hook_events', 'runs_allowed')
