"""align diagnostics schema

Revision ID: c2cd87f58126
Revises: eddaef7af050
Create Date: 2026-04-11 05:40:33.622173

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c2cd87f58126'
down_revision: Union[str, Sequence[str], None] = 'eddaef7af050'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Standardize diagnostics tables."""
    # ── 1. uranium_eval_history ──────────────────
    op.add_column('uranium_eval_history', sa.Column('model_target', sa.String(length=50), nullable=True))
    op.add_column('uranium_eval_history', sa.Column('fold_date', sa.Date(), nullable=True))
    op.add_column('uranium_eval_history', sa.Column('n_samples', sa.Integer(), nullable=True))
    op.add_column('uranium_eval_history', sa.Column('ece', sa.Float(), nullable=True))

    # Backfill
    op.execute("UPDATE uranium_eval_history SET model_target = 'pa_outcome'")
    op.execute("UPDATE uranium_eval_history SET fold_date = TO_DATE(test_year::text || '-01-01', 'YYYY-MM-DD')")
    op.execute("UPDATE uranium_eval_history SET n_samples = n_games")

    # Make non-nullable
    op.alter_column('uranium_eval_history', 'model_target', nullable=False)
    op.alter_column('uranium_eval_history', 'n_samples', nullable=False)

    # Add indices and update unique constraint
    op.create_index(op.f('ix_uranium_eval_history_fold_date'), 'uranium_eval_history', ['fold_date'], unique=False)
    op.create_index(op.f('ix_uranium_eval_history_model_target'), 'uranium_eval_history', ['model_target'], unique=False)
    op.drop_constraint('uq_uranium_eval_model_year', 'uranium_eval_history', type_='unique')
    op.create_unique_constraint('uq_uranium_eval_target_version_fold', 'uranium_eval_history', ['model_target', 'model_version', 'fold_date'])

    # Drop old column
    op.drop_column('uranium_eval_history', 'n_games')
    op.drop_column('uranium_eval_history', 'test_year') # Standardizing on fold_date

    # ── 2. uranium_calibration_bins ──────────────
    op.add_column('uranium_calibration_bins', sa.Column('model_target', sa.String(length=50), nullable=True))
    op.add_column('uranium_calibration_bins', sa.Column('fold_date', sa.Date(), nullable=True))

    # Rename existing columns
    op.alter_column('uranium_calibration_bins', 'bin_lower', new_column_name='bin_start')
    op.alter_column('uranium_calibration_bins', 'bin_upper', new_column_name='bin_end')
    op.alter_column('uranium_calibration_bins', 'pred_mean', new_column_name='predicted_prob_mean')
    op.alter_column('uranium_calibration_bins', 'obs_rate', new_column_name='actual_prob_mean')
    op.alter_column('uranium_calibration_bins', 'n_samples', new_column_name='sample_count')

    # Backfill
    op.execute("UPDATE uranium_calibration_bins SET model_target = 'pa_outcome'")
    op.execute("UPDATE uranium_calibration_bins SET fold_date = TO_DATE(test_year::text || '-01-01', 'YYYY-MM-DD')")

    # Make non-nullable
    op.alter_column('uranium_calibration_bins', 'model_target', nullable=False)
    op.alter_column('uranium_calibration_bins', 'fold_date', nullable=False)

    # Indices and Constraints
    op.create_index(op.f('ix_uranium_calibration_bins_fold_date'), 'uranium_calibration_bins', ['fold_date'], unique=False)
    op.create_index(op.f('ix_uranium_calibration_bins_model_target'), 'uranium_calibration_bins', ['model_target'], unique=False)
    op.drop_constraint('uq_uranium_calibration_bin', 'uranium_calibration_bins', type_='unique')
    op.create_unique_constraint('uq_uranium_calibration_bin_target_version_fold', 'uranium_calibration_bins', ['model_target', 'model_version', 'fold_date', 'bin_index'])

    # Drop old
    op.drop_column('uranium_calibration_bins', 'test_year')

    # ── 3. uranium_shap_global ──────────────────
    op.add_column('uranium_shap_global', sa.Column('model_target', sa.String(length=50), nullable=True))
    op.add_column('uranium_shap_global', sa.Column('fold_date', sa.Date(), nullable=True))

    # Backfill
    op.execute("UPDATE uranium_shap_global SET model_target = 'pa_outcome'")
    op.execute("UPDATE uranium_shap_global SET fold_date = TO_DATE(SUBSTRING(dataset_label FROM 6)::text || '-01-01', 'YYYY-MM-DD') WHERE dataset_label LIKE 'test_%'")
    op.execute("UPDATE uranium_shap_global SET fold_date = '2025-01-01' WHERE fold_date IS NULL")

    # Make non-nullable
    op.alter_column('uranium_shap_global', 'model_target', nullable=False)
    op.alter_column('uranium_shap_global', 'fold_date', nullable=False)

    # Index and Constraint
    op.create_index(op.f('ix_uranium_shap_global_fold_date'), 'uranium_shap_global', ['fold_date'], unique=False)
    op.create_index(op.f('ix_uranium_shap_global_model_target'), 'uranium_shap_global', ['model_target'], unique=False)
    op.drop_constraint('uq_uranium_shap_global_row', 'uranium_shap_global', type_='unique')
    op.create_unique_constraint('uq_uranium_shap_global_target_version_fold', 'uranium_shap_global', ['model_target', 'model_version', 'fold_date', 'feature_name'])

    # Drop old
    op.drop_column('uranium_shap_global', 'dataset_label')


def downgrade() -> None:
    """Downgrade diagnostics schema."""
    # Diagnostic tables are non-critical and complex to reverse-backfill.
    # We provide a simple cleanup logic or leave it to manual intervention.
    pass
