"""add_player_transactions_table

Revision ID: 490bc14c8971
Revises: bc91727ec276
Create Date: 2026-03-31 21:42:46.713025

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '490bc14c8971'
down_revision: Union[str, Sequence[str], None] = 'bc91727ec276'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add team_id columns to game_results if they don't exist
    # Note: we check for existence to avoid redundant additions
    bind = op.get_bind()
    inspect = sa.inspect(bind)
    columns = [c['name'] for c in inspect.get_columns('game_results')]
    
    for col in ['home_team_id', 'away_team_id']:
        if col not in columns:
            op.add_column('game_results', sa.Column(col, sa.Integer(), nullable=True))

    op.create_table(
        'player_transactions',
        sa.Column('transaction_id', sa.String(), nullable=False),
        sa.Column('player_id', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('transaction_date', sa.Date(), nullable=False),
        sa.Column('effective_date', sa.Date(), nullable=True),
        sa.Column('resolution_date', sa.Date(), nullable=True),
        sa.Column('type_desc', sa.String(), nullable=False),
        sa.Column('il_type', sa.String(), nullable=True),
        sa.Column('injury_body_part', sa.String(), nullable=True),
        sa.Column('injury_descriptor', sa.String(), nullable=True),
        sa.Column('raw_description', sa.Text(), nullable=True),
        sa.Column('days_on_il', sa.Integer(), sa.Computed('resolution_date - effective_date', persisted=True), nullable=True),
        sa.PrimaryKeyConstraint('transaction_id')
    )
    op.create_index('idx_player_transactions_player_date', 'player_transactions', ['player_id', 'effective_date'], unique=False)
    op.create_index('idx_player_transactions_team_date', 'player_transactions', ['team_id', 'transaction_date'], unique=False)

    # Create the view exactly as specified, mapped to game_results
    op.execute("""
    CREATE OR REPLACE VIEW v_game_il_features AS
    SELECT
        g.game_id AS game_pk,
        g.game_date,
        g.home_team_id,
        g.away_team_id,

        -- Starting pitcher IL history (as of game_date)
        (SELECT COUNT(*) FROM player_transactions pt
         WHERE pt.player_id = g.home_pitcher_id
           AND pt.il_type IS NOT NULL
           AND EXTRACT(YEAR FROM pt.effective_date) = EXTRACT(YEAR FROM g.game_date)
           AND pt.effective_date < g.game_date
        ) AS home_sp_il_stints_ytd,

        (SELECT (g.game_date - pt.resolution_date)
         FROM player_transactions pt
         WHERE pt.player_id = g.home_pitcher_id
           AND pt.resolution_date IS NOT NULL
           AND pt.resolution_date < g.game_date
         ORDER BY pt.resolution_date DESC LIMIT 1
        ) AS home_sp_days_since_il_return,

        -- Team roster health (active IL count as of game_date)
        (SELECT COUNT(*) FROM player_transactions pt
         WHERE pt.team_id = g.home_team_id
           AND pt.effective_date <= g.game_date
           AND (pt.resolution_date IS NULL OR pt.resolution_date > g.game_date)
        ) AS home_team_active_il_count

    FROM game_results g;
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP VIEW IF EXISTS v_game_il_features")
    op.drop_table('player_transactions')
    # Remove team_id columns from game_results
    for col in ['home_team_id', 'away_team_id']:
        op.drop_column('game_results', col)
