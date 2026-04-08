"""Uranium Model Performance & Explainability dashboard."""

import streamlit as st
import pandas as pd
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
def load_eval_history():
    query = """
        SELECT model_version, test_year, train_start_year, train_end_year,
               n_games, accuracy, auc, log_loss_val AS log_loss, brier
        FROM uranium_eval_history
        ORDER BY model_version, test_year
    """
    try:
        return pd.read_sql(text(query), engine)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_calibration(model_version: str, test_year: int | None = None):
    base = """
        SELECT model_version, test_year, bin_index, bin_lower, bin_upper,
               pred_mean, obs_rate, n_samples
        FROM uranium_calibration_bins
        WHERE model_version = :mv
    """
    params: dict = {"mv": model_version}
    if test_year is not None:
        base += " AND test_year = :ty"
        params["ty"] = test_year
    base += " ORDER BY test_year, bin_index"
    try:
        return pd.read_sql(text(base), engine, params=params)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_global_shap(
    model_version: str, dataset_label: str | None = None
) -> pd.DataFrame:
    base = """
        SELECT model_version, dataset_label, feature_name,
               mean_abs_shap, mean_shap
        FROM uranium_shap_global
        WHERE model_version = :mv
    """
    params: dict = {"mv": model_version}
    if dataset_label:
        base += " AND dataset_label = :dl"
        params["dl"] = dataset_label
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
    years = sorted(df_mv["test_year"].unique().tolist())
    selected_year = st.selectbox(
        "Test Year (detail view)",
        options=years,
        index=len(years) - 1,
    )

    # Ensure selected_model is a valid string for typed calls
    model_version_str = str(selected_model) if selected_model else ""

    # SHAP dataset selector
    shap_df_all = load_global_shap(model_version_str)
    shap_labels = (
        sorted(shap_df_all["dataset_label"].unique().tolist())
        if not shap_df_all.empty
        else []
    )
    preferred_label = f"test_{selected_year}"
    selected_shap_label = None
    if shap_labels:
        default_idx = (
            shap_labels.index(preferred_label) if preferred_label in shap_labels else 0
        )
        selected_shap_label = st.selectbox(
            "SHAP Dataset", options=shap_labels, index=default_idx
        )


# ── Hero metrics ──────────────────────────────────────────────────────────

st.subheader(f"Model: {selected_model}")

df_mv = eval_df[eval_df["model_version"] == selected_model].copy()
latest = df_mv[df_mv["test_year"] == selected_year]

if not latest.empty:
    row = latest.iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{row['accuracy']:.3f}")
    c2.metric("ROC AUC", f"{row['auc']:.3f}")
    c3.metric("Log Loss", f"{row['log_loss']:.3f}")
    c4.metric("Brier Score", f"{row['brier']:.3f}")
    c5.metric("Games", f"{int(row['n_games']):,}")

st.markdown("---")


# ── Section 1: Walk-Forward Metrics Over Time ─────────────────────────────

st.markdown("### 📈 Walk-Forward Metrics by Test Year")

metric_cols = ["accuracy", "auc", "log_loss", "brier"]
df_long = df_mv.melt(
    id_vars=["test_year"],
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
    x="test_year",
    y="value",
    color="metric",
    markers=True,
    color_discrete_map=color_map,
    labels={"test_year": "Test Year", "value": "", "metric": "Metric"},
    template=TEMPLATE,
)
fig_metrics.update_layout(
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(dtick=1),
)
st.plotly_chart(fig_metrics, use_container_width=True)

# Also show the raw table
with st.expander("Raw Walk-Forward Data"):
    st.dataframe(df_mv, use_container_width=True)

st.markdown("---")


# ── Section 2: Calibration / Reliability Curve ────────────────────────────

st.markdown(f"### 🎯 Calibration Curve — Test Year {selected_year}")

cal_df = load_calibration(model_version_str, selected_year)

if cal_df.empty:
    st.info("No calibration bins found for this model/year. Run training to populate.")
else:
    # Filter out empty bins for cleaner display
    cal_plot = cal_df[cal_df["n_samples"] > 0].copy()

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
            x=cal_plot["pred_mean"],
            y=cal_plot["obs_rate"],
            mode="lines+markers",
            marker=dict(
                size=cal_plot["n_samples"].clip(upper=300) / 10 + 4,
                color=COLORS["secondary"],
            ),
            line=dict(color=COLORS["secondary"], width=2),
            name="Uranium",
            hovertemplate=(
                "Predicted: %{x:.2f}<br>"
                "Observed: %{y:.2f}<br>"
                "N=%{customdata}<extra></extra>"
            ),
            customdata=cal_plot["n_samples"],
        )
    )

    fig_cal.update_layout(
        template=TEMPLATE,
        xaxis=dict(title="Predicted Win Probability", range=[0, 1]),
        yaxis=dict(title="Observed Win Rate", range=[0, 1]),
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # Bin detail table
    with st.expander("Calibration Bin Details"):
        st.dataframe(
            cal_df[
                [
                    "bin_index",
                    "bin_lower",
                    "bin_upper",
                    "pred_mean",
                    "obs_rate",
                    "n_samples",
                ]
            ],
            use_container_width=True,
        )

st.markdown("---")


# ── Section 3: Global Feature Importance (SHAP) ──────────────────────────

st.markdown("### 🔬 Global Feature Importance (SHAP)")

if selected_shap_label is None or shap_df_all.empty:
    st.info(
        "No SHAP results found. Install `shap` (`uv add shap`) and re-run training."
    )
else:
    shap_df = shap_df_all[shap_df_all["dataset_label"] == selected_shap_label].copy()
    if shap_df.empty:
        st.info(f"No SHAP rows for dataset '{selected_shap_label}'.")
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
        st.plotly_chart(fig_shap, use_container_width=True)

        with st.expander("Raw SHAP Table"):
            st.dataframe(
                shap_top[["feature_name", "mean_abs_shap", "mean_shap"]].reset_index(
                    drop=True
                ),
                use_container_width=True,
            )

st.markdown("---")
st.success("Uranium Diagnostics Engine: Online")
