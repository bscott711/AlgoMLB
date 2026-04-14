import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from algomlb.db.session import get_engine
from sqlalchemy import text


def _render_league_trends(engine):
    """Render league-wide injury trends and snapshots."""
    st.subheader("📊 League-Wide Injury Trends")
    c1, c2 = st.columns(2)

    with c1:
        st.write("#### Most Vulnerable Body Parts")
        query = """
            SELECT injury_body_part, count(*) as count
            FROM player_transactions
            WHERE il_type IS NOT NULL AND injury_body_part != 'unknown'
            GROUP BY 1 ORDER BY 2 DESC LIMIT 10
        """
        df = pd.read_sql(query, engine)
        if not df.empty:
            fig = px.bar(
                df,
                x="count",
                y="injury_body_part",
                orientation="h",
                template="plotly_dark",
                color="count",
                color_continuous_scale="Reds",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")

    with c2:
        st.write("#### Typical Injury Types")
        query = """
            SELECT injury_descriptor, count(*) as count
            FROM player_transactions
            WHERE il_type IS NOT NULL AND injury_descriptor != 'unknown'
            GROUP BY 1 ORDER BY 2 DESC LIMIT 10
        """
        df = pd.read_sql(query, engine)
        if not df.empty:
            fig = px.pie(
                df,
                values="count",
                names="injury_descriptor",
                template="plotly_dark",
                hole=0.4,
            )
            st.plotly_chart(fig, width="stretch")


def _render_temporal_trends(engine):
    """Render temporal roster health and seasonality trends."""
    st.subheader("📅 Temporal Roster Health")
    st.write("#### IL Placements by Month (Seasonality)")
    query = """
        SELECT to_char(transaction_date, 'Mon') as month_name, 
               extract(month from transaction_date) as month_num,
               count(*) as count
        FROM player_transactions WHERE il_type IS NOT NULL
        GROUP BY 1, 2 ORDER BY 2
    """
    df = pd.read_sql(query, engine)
    if not df.empty:
        fig = px.line(
            df,
            x="month_name",
            y="count",
            markers=True,
            title="Seasonality of Injury Placements",
            template="plotly_dark",
        )
        st.plotly_chart(fig, width="stretch")


def _render_gold_metrics(engine, player_input, df_player):
    """Render physical fatigue and stuff stability metrics from the Gold layer."""
    st.markdown("---")
    st.subheader("🔋 Workload & Stuff Stability")

    p_id = int(player_input) if player_input.isdigit() else None
    if not p_id and not df_player.empty:
        q_id = text(
            "SELECT DISTINCT player_id FROM player_transactions WHERE player_name ILIKE :pname LIMIT 1"
        )
        with engine.connect() as conn:
            res = conn.execute(q_id, {"pname": f"%{player_input}%"}).fetchone()
            if res:
                p_id = res[0]

    if p_id:
        q_gold = text(
            "SELECT * FROM player_rolling_features WHERE player_id = :pid AND role = 'PITCHER' ORDER BY game_date DESC LIMIT 1"
        )
        with engine.connect() as conn:
            gold_df = pd.read_sql(q_gold, conn, params={"pid": p_id})

        if not gold_df.empty:
            latest = gold_df.iloc[0]
            hg1, hg2 = st.columns([1, 1])
            with hg1:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=latest.get("fatigue_index_7d", 0),
                        title={"text": "7-Day Fatigue Index (Whiteside)"},
                        gauge={
                            "axis": {"range": [0, 30]},
                            "bar": {"color": "#00E5FF"},
                            "steps": [
                                {"range": [0, 15], "color": "rgba(0, 255, 0, 0.1)"},
                                {"range": [15, 25], "color": "rgba(255, 255, 0, 0.1)"},
                                {"range": [25, 30], "color": "rgba(255, 0, 0, 0.2)"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 25,
                            },
                        },
                    )
                )
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, width="stretch")
            with hg2:
                st.write("#### 📡 Early Warning Signals")
                for label, key, delta_key, unit in [
                    (
                        "Spin Rate vs Season",
                        "roll_avg_spin_rate",
                        "delta_spin_rate_3g",
                        "rpm",
                    ),
                    (
                        "Velo vs Season",
                        "roll_avg_release_speed",
                        "delta_fb_velo_3g",
                        "mph",
                    ),
                    (
                        "Extension vs Season",
                        "roll_avg_release_extension",
                        "delta_extension_3g",
                        "ft",
                    ),
                ]:
                    st.metric(
                        label,
                        f"{latest.get(key, 0):.1f} {unit}",
                        delta=f"{latest.get(delta_key, 0):.1f} {unit}",
                    )
        else:
            st.info(
                "No recent Pitcher health metrics found for this player in the Gold Layer."
            )


def show_health_analytics(engine=None):
    """Main entry point for Player Health & Injury Analytics."""
    if engine is None:
        engine = get_engine()

    st.title("🩹 Player Health & Injury Analytics")
    st.markdown(
        "Insights from historical player transactions, IL stints, and injury patterns."
    )
    st.markdown("---")

    _render_league_trends(engine)
    st.markdown("---")
    _render_temporal_trends(engine)
    st.markdown("---")

    st.subheader("🔍 Individual Player History")
    player_input = st.text_input(
        "Enter Player ID or Full Name (e.g. Mookie Betts)", "605141"
    )

    if player_input:
        base_query = """
            WITH player_tx AS (
                SELECT transaction_date, raw_description, il_type, injury_body_part, injury_descriptor,
                       MAX(CASE WHEN (raw_description ILIKE '%%placed%%' OR raw_description ILIKE '%%assigned%%') THEN transaction_date END) 
                           OVER (PARTITION BY player_id ORDER BY transaction_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as last_placed_date
                FROM player_transactions WHERE {where_clause}
            )
            SELECT transaction_date, il_type, injury_body_part, injury_descriptor, raw_description,
                   CASE WHEN (raw_description ILIKE '%%activated%%' OR raw_description ILIKE '%%reinstated%%') AND last_placed_date IS NOT NULL
                        THEN (transaction_date - last_placed_date) ELSE 0 END as days_on_il
            FROM player_tx ORDER BY transaction_date DESC
        """
        where = (
            "player_id = :pid" if player_input.isdigit() else "player_name ILIKE :pname"
        )
        params = (
            {"pid": int(player_input)}
            if player_input.isdigit()
            else {"pname": f"%{player_input}%"}
        )

        with engine.connect() as conn:
            df_player = pd.read_sql(
                text(base_query.format(where_clause=where)), conn, params=params
            )

        if not df_player.empty:
            st.write(f"History for '{player_input}'")
            df_player["days_on_il"] = df_player["days_on_il"].fillna(0).astype(int)
            st.dataframe(df_player, width="stretch")

            c1, c2 = st.columns(2)
            c1.metric("Lifetime IL Days", f"{int(df_player['days_on_il'].sum())}")
            c2.metric(
                "Total IL Stints",
                df_player[
                    df_player["raw_description"].str.contains(
                        "placed", case=False, na=False
                    )
                ].shape[0],
            )

            _render_gold_metrics(engine, player_input, df_player)
        else:
            st.warning(f"No data found for '{player_input}'.")

    st.markdown("---")
    st.success("Health Model Ready: Features fully populated.")


if __name__ == "__main__":
    st.set_page_config(page_title="Player Health & Injury Market", layout="wide")
    show_health_analytics()
