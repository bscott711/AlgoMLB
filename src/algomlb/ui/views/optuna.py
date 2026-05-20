import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import optuna
from sqlalchemy import text
from algomlb.db.session import get_engine


def render_optuna_view():
    st.header("🧪 Model Tuning & Diagnostics Lab")
    st.markdown("---")

    # Sidebar for filters
    st.sidebar.title("Filters")
    target_filter = st.sidebar.selectbox(
        "Model Target",
        options=["pa_outcome", "home_win", "total_runs_actual", "is_strike"],
        index=0,
    )

    engine = get_engine()
    # Get available versions for this target
    versions_query = text(
        "SELECT DISTINCT model_version FROM uranium_eval_history WHERE model_target = :target"
    )
    try:
        versions_df = pd.read_sql(
            versions_query, engine, params={"target": target_filter}
        )
        available_versions = sorted(versions_df["model_version"].tolist())
    except Exception:
        available_versions = []

    try:
        storage_url = "sqlite:///models/optuna_history.db"
        study_summaries = optuna.get_all_study_summaries(storage=storage_url)
        for s in study_summaries:
            if s.study_name.startswith(target_filter + "_"):
                version = s.study_name[len(target_filter) + 1 :]
                if version not in available_versions:
                    available_versions.append(version)
    except Exception:
        pass

    available_versions = sorted(list(set(available_versions)))
    if not available_versions:
        available_versions = ["v1.0"]

    version_filter = st.sidebar.selectbox(
        "Model Version",
        options=available_versions,
        index=len(available_versions) - 1,
    )

    tabs = st.tabs(
        [
            "Hyperparameter Analysis",
            "Calibration & Reliability",
            "Feature Importance Drift",
        ]
    )

    # ── Tab 1: Optuna Studies ──────────────────────────────────────────
    with tabs[0]:
        st.subheader("Optuna Optimization History")

        try:
            storage_url = "sqlite:///models/optuna_history.db"
            study_name = f"{target_filter}_{version_filter}"

            # Load study from SQLite
            study_summaries = optuna.get_all_study_summaries(storage=storage_url)
            study_names = [s.study_name for s in study_summaries]

            if not study_names:
                st.warning(
                    "No Optuna studies found in sqlite:///models/optuna_history.db"
                )
            else:
                selected_study = st.selectbox(
                    "Select Optuna Study",
                    study_names,
                    index=study_names.index(study_name)
                    if study_name in study_names
                    else 0,
                )

                study = optuna.load_study(
                    study_name=selected_study, storage=storage_url
                )

                # Plot optimization history
                fig_hist = optuna.visualization.plot_optimization_history(study)
                st.plotly_chart(fig_hist, width="stretch")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Best Hyperparameters**")
                    st.json(study.best_params)
                with col2:
                    st.markdown("**Best Value (Loss)**")
                    st.metric("Objective Value", f"{study.best_value:.4f}")

                # Param Importances
                st.markdown("---")
                st.markdown("**Parameter Importances**")
                fig_imp = optuna.visualization.plot_param_importances(study)
                st.plotly_chart(fig_imp, width="stretch")

        except Exception as e:
            st.error(f"Error loading Optuna study: {e}")

    # ── Tab 2: Calibration ─────────────────────────────────────────────
    with tabs[1]:
        st.subheader("20-Bin Reliability Diagram")

        engine = get_engine()
        query = text("""
            SELECT fold_date, bin_index, bin_start, bin_end, predicted_prob_mean, actual_prob_mean, sample_count
            FROM uranium_calibration_bins
            WHERE model_target = :target AND model_version = :version
            ORDER BY fold_date DESC, bin_index ASC
        """)

        try:
            cal_df = pd.read_sql(
                query,
                engine,
                params={"target": target_filter, "version": version_filter},
            )

            if cal_df.empty:
                st.info(
                    f"No calibration data found in Postgres for {target_filter}/{version_filter}"
                )
            else:
                latest_fold = cal_df["fold_date"].max()
                latest_df = cal_df[cal_df["fold_date"] == latest_fold]

                st.markdown(f"**Latest Fold: {latest_fold}**")

                fig_cal = go.Figure()

                # Perfect Calibration line
                fig_cal.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        line=dict(color="gray", dash="dash"),
                        name="Perfectly Calibrated",
                    )
                )

                # Model Calibration
                fig_cal.add_trace(
                    go.Scatter(
                        x=latest_df["predicted_prob_mean"],
                        y=latest_df["actual_prob_mean"],
                        mode="lines+markers",
                        marker=dict(
                            size=latest_df["sample_count"]
                            / latest_df["sample_count"].max()
                            * 20
                        ),
                        line=dict(color="#00d1b2"),
                        name="Model Output",
                    )
                )

                fig_cal.update_layout(
                    xaxis_title="Predicted Probability",
                    yaxis_title="Actual Win Rate",
                    template="plotly_dark",
                    height=600,
                )

                st.plotly_chart(fig_cal, width="stretch")

                # Metrics History
                st.markdown("---")
                st.markdown("**ECE History**")
                ece_query = text("""
                    SELECT fold_date, ece, n_samples
                    FROM uranium_eval_history
                    WHERE model_target = :target AND model_version = :version
                    ORDER BY fold_date ASC
                """)
                ece_df = pd.read_sql(
                    ece_query,
                    engine,
                    params={"target": target_filter, "version": version_filter},
                )
                fig_ece = px.line(
                    ece_df,
                    x="fold_date",
                    y="ece",
                    markers=True,
                    title="Expected Calibration Error over Folds",
                )
                st.plotly_chart(fig_ece, width="stretch")

        except Exception as e:
            st.error(f"Error loading calibration data: {e}")

    # ── Tab 3: SHAP Feature Importance ──────────────────────────────────
    with tabs[2]:
        st.subheader("Feature Importance Over Time (SHAP Drift)")

        shap_query = text("""
            SELECT fold_date, feature_name, mean_abs_shap
            FROM uranium_shap_global
            WHERE model_target = :target AND model_version = :version
            ORDER BY fold_date ASC, mean_abs_shap DESC
        """)

        try:
            shap_df = pd.read_sql(
                shap_query,
                engine,
                params={"target": target_filter, "version": version_filter},
            )

            if shap_df.empty:
                st.info(f"No SHAP data found for {target_filter}/{version_filter}")
            else:
                # Get Top 15 features by latest fold importance
                latest_fold = shap_df["fold_date"].max()
                top_features = (
                    shap_df[shap_df["fold_date"] == latest_fold]
                    .head(15)["feature_name"]
                    .tolist()
                )

                plot_df = shap_df[shap_df["feature_name"].isin(top_features)]

                fig_shap = px.bar(
                    plot_df,
                    x="fold_date",
                    y="mean_abs_shap",
                    color="feature_name",
                    title="Top 15 Features: SHAP Magnitude Drift",
                    barmode="stack",
                    template="plotly_dark",
                    height=600,
                )
                st.plotly_chart(fig_shap, width="stretch")

                st.markdown(
                    "**Why this matters:** A sudden spike in a single feature's importance over time can indicate data leakage or a shift in the underlying MLB environment (e.g. pitch clock effects)."
                )

        except Exception as e:
            st.error(f"Error loading SHAP data: {e}")


if __name__ == "__main__":
    render_optuna_view()
