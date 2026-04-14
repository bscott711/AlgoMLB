"""Uranium Model Performance & Explainability dashboard."""

import streamlit as st
import pandas as pd
import datetime
from typing import cast
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text

from algomlb.db.session import get_engine
from algomlb.ui.styles import apply_premium_styles, get_plotly_template, COLORS

# --- Page setup ---
st.set_page_config(page_title="Model Performance & Explainability", layout="wide")
apply_premium_styles()

st.title("📊 Uranium Model Performance & Explainability")
st.markdown("---")

engine = get_engine()
TEMPLATE = get_plotly_template()


# ── Data loaders ──────────────────────────────────────────────────────────


@st.cache_data(ttl=600)
def load_eval_history(model_target: str = "pa_outcome"):
    query = """
        SELECT model_target, model_version, fold_date, train_start_year, train_end_year,
               n_samples, accuracy, auc, log_loss_val AS log_loss, brier
        FROM uranium_eval_history
        WHERE model_target = :target
        ORDER BY model_version, fold_date
    """
    try:
        return pd.read_sql(text(query), engine, params={"target": model_target})
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_calibration(
    model_version: str,
    fold_date: datetime.date | None = None,
    model_target: str = "pa_outcome",
):
    base = """
        SELECT model_version, fold_date, bin_index, bin_start, bin_end,
               predicted_prob_mean, actual_prob_mean, sample_count
        FROM uranium_calibration_bins
        WHERE model_version = :mv AND model_target = :target
    """
    params: dict = {"mv": model_version, "target": model_target}
    if fold_date is not None:
        base += " AND fold_date = :fd"
        params["fd"] = fold_date
    base += " ORDER BY fold_date, bin_index"
    try:
        return pd.read_sql(text(base), engine, params=params)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_global_shap(
    model_version: str,
    fold_date: datetime.date | None = None,
    model_target: str = "pa_outcome",
) -> pd.DataFrame:
    base = """
        SELECT model_version, fold_date, feature_name,
               mean_abs_shap, mean_shap
        FROM uranium_shap_global
        WHERE model_version = :mv AND model_target = :target
    """
    params: dict = {"mv": model_version, "target": model_target}
    if fold_date:
        base += " AND fold_date = :fd"
        params["fd"] = fold_date
    base += " ORDER BY mean_abs_shap DESC"
    try:
        return pd.read_sql(text(base), engine, params=params)
    except Exception:
        return pd.DataFrame()


# ── Load eval history ─────────────────────────────────────────────────────

eval_df = load_eval_history()

if eval_df.empty:
    st.warning(
        "No evaluation history found. Run `algomlb ml train` or "
        "`algomlb ml walk-forward` after creating the `uranium_eval_history` table."
    )
    st.info("Tip: Run `alembic upgrade head` to create the new diagnostic tables.")
    st.stop()


# ── Sidebar controls ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔧 Evaluation Controls")

    model_versions = sorted(eval_df["model_version"].unique().tolist())
    selected_model = st.selectbox("Model Version", model_versions)

    df_mv = eval_df[eval_df["model_version"] == selected_model]
    dates = sorted(df_mv["fold_date"].unique().tolist())
    selected_date = st.selectbox(
        "Fold Date",
        options=dates,
        index=len(dates) - 1,
    )

    # Ensure selected_model is a valid string for typed calls
    model_version_str = str(selected_model) if selected_model else ""

    # SHAP dataset selector
    shap_df_all = load_global_shap(model_version_str)
    shap_dates = (
        sorted(shap_df_all["fold_date"].unique().tolist())
        if not shap_df_all.empty
        else []
    )
    selected_shap_date = None
    if shap_dates:
        default_idx = (
            shap_dates.index(selected_date) if selected_date in shap_dates else 0
        )
        selected_shap_date = st.selectbox(
            "SHAP Fold Date", options=shap_dates, index=default_idx
        )


# ── Hero metrics ──────────────────────────────────────────────────────────

st.subheader(f"Model: {selected_model}")

df_mv = eval_df[eval_df["model_version"] == selected_model].copy()
latest = df_mv[df_mv["fold_date"] == selected_date]

if not latest.empty:
    row = latest.iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{row['accuracy']:.3f}")
    c2.metric("ROC AUC", f"{row['auc']:.3f}")
    c3.metric("Log Loss", f"{row['log_loss']:.3f}")
    c4.metric("Brier Score", f"{row['brier']:.3f}")
    c5.metric("Samples", f"{int(row['n_samples']):,}")

st.markdown("---")


# ── Section 1: Walk-Forward Metrics Over Time ─────────────────────────────

st.markdown("### 📈 Walk-Forward Metrics by Fold Date")

metric_cols = ["accuracy", "auc", "log_loss", "brier"]
df_long = df_mv.melt(
    id_vars=["fold_date"],
    value_vars=metric_cols,
    var_name="metric",
    value_name="value",
)

color_map = {
    "accuracy": COLORS["success"],
    "auc": COLORS["secondary"],
    "log_loss": COLORS["warning"],
    "brier": COLORS["danger"],
}

fig_metrics = px.line(
    df_long,
    x="fold_date",
    y="value",
    color="metric",
    markers=True,
    color_discrete_map=color_map,
    labels={"fold_date": "Fold Date", "value": "", "metric": "Metric"},
    template=TEMPLATE,
)
fig_metrics.update_layout(
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_metrics, width="stretch")

# Also show the raw table
with st.expander("Raw Walk-Forward Data"):
    st.dataframe(df_mv, width="stretch")

st.markdown("---")


# ── Section 2: Calibration / Reliability Curve ────────────────────────────

st.markdown(f"### 🎯 Calibration Curve — Fold {selected_date}")

cal_df = load_calibration(model_version_str, selected_date)

if cal_df.empty:
    st.info("No calibration bins found for this model/date. Run training to populate.")
else:
    # Filter out empty bins for cleaner display
    cal_plot = cal_df[cal_df["sample_count"] > 0].copy()

    fig_cal = go.Figure()

    # Perfect calibration diagonal
    fig_cal.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1),
            showlegend=False,
        )
    )

    # Actual calibration curve
    fig_cal.add_trace(
        go.Scatter(
            x=cal_plot["predicted_prob_mean"],
            y=cal_plot["actual_prob_mean"],
            mode="lines+markers",
            marker=dict(
                size=cal_plot["sample_count"].clip(upper=300) / 10 + 4,
                color=COLORS["secondary"],
            ),
            line=dict(color=COLORS["secondary"], width=2),
            name="Uranium",
            hovertemplate=(
                "Predicted: %{x:.2f}<br>"
                "Observed: %{y:.2f}<br>"
                "N=%{customdata}<extra></extra>"
            ),
            customdata=cal_plot["sample_count"],
        )
    )

    fig_cal.update_layout(
        template=TEMPLATE,
        xaxis=dict(title="Predicted Win Probability", range=[0, 1]),
        yaxis=dict(title="Observed Win Rate", range=[0, 1]),
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_cal, width="stretch")

    # Bin detail table
    with st.expander("Calibration Bin Details"):
        st.dataframe(
            cal_df[
                [
                    "bin_index",
                    "bin_start",
                    "bin_end",
                    "predicted_prob_mean",
                    "actual_prob_mean",
                    "sample_count",
                ]
            ],
            width="stretch",
        )

st.markdown("---")


# ── Section 3: Global Feature Importance (SHAP) ──────────────────────────

st.markdown("### 🔬 Global Feature Importance (SHAP)")

if selected_shap_date is None or shap_df_all.empty:
    st.info(
        "No SHAP results found. Install `shap` (`uv add shap`) and re-run training."
    )
else:
    shap_df = shap_df_all[shap_df_all["fold_date"] == selected_shap_date].copy()
    if shap_df.empty:
        st.info(f"No SHAP rows for date '{selected_shap_date}'.")
    else:
        top_n = st.slider("Top N Features", min_value=5, max_value=50, value=20)
        # Force DataFrame type for pyright to resolve nlargest
        shap_df_df = cast(pd.DataFrame, shap_df)
        shap_top = shap_df_df.nlargest(n=top_n, columns="mean_abs_shap")

        fig_shap = px.bar(
            shap_top.sort_values(by="mean_abs_shap"),
            x="mean_abs_shap",
            y="feature_name",
            orientation="h",
            color="mean_shap",
            color_continuous_scale=["#FF1744", "#424242", "#00E676"],
            color_continuous_midpoint=0,
            labels={
                "mean_abs_shap": "Mean |SHAP|",
                "feature_name": "",
                "mean_shap": "Direction",
            },
            template=TEMPLATE,
        )
        fig_shap.update_layout(
            height=max(400, top_n * 22),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_shap, width="stretch")

        with st.expander("Raw SHAP Table"):
            st.dataframe(
                shap_top[["feature_name", "mean_abs_shap", "mean_shap"]].reset_index(
                    drop=True
                ),
                width="stretch",
            )

st.markdown("---")
st.success("Uranium Diagnostics Engine: Online")
