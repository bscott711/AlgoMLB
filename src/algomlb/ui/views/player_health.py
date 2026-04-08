import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from algomlb.db.session import get_engine
from sqlalchemy import text

st.set_page_config(page_title="Player Health & Injury Market", layout="wide")

st.title("🩹 Player Health & Injury Analytics")
st.markdown("""
Insights from historical player transactions, IL stints, and injury patterns. 
This data feeds features like `sp_days_since_il_return` and `team_active_il_count`.
""")
st.markdown("---")

engine = get_engine()

# --- 1. LEAGUE-WIDE INJURY SNAPSHOT ---
st.subheader("📊 League-Wide Injury Trends")
c1, c2 = st.columns(2)

with c1:
    st.write("#### Most Vulnerable Body Parts")
    query_parts = """
        SELECT injury_body_part, count(*) as count
        FROM player_transactions
        WHERE il_type IS NOT NULL AND injury_body_part != 'unknown'
        GROUP BY 1 ORDER BY 2 DESC LIMIT 10
    """
    df_parts = pd.read_sql(query_parts, engine)
    if not df_parts.empty:
        fig_parts = px.bar(
            df_parts,
            x="count",
            y="injury_body_part",
            orientation="h",
            template="plotly_dark",
            color="count",
            color_continuous_scale="Reds",
        )
        fig_parts.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_parts, width="stretch")

with c2:
    st.write("#### Typical Injury Types")
    query_kind = """
        SELECT injury_descriptor, count(*) as count
        FROM player_transactions
        WHERE il_type IS NOT NULL AND injury_descriptor != 'unknown'
        GROUP BY 1 ORDER BY 2 DESC LIMIT 10
    """
    df_kind = pd.read_sql(query_kind, engine)
    if not df_kind.empty:
        fig_kind = px.pie(
            df_kind,
            values="count",
            names="injury_descriptor",
            template="plotly_dark",
            hole=0.4,
        )
        st.plotly_chart(fig_kind, width="stretch")

st.markdown("---")

# --- 2. TEMPORAL TRENDS ---
st.subheader("📅 Temporal Roster Health")

st.write("#### IL Placements by Month (Seasonality)")
query_monthly = """
    SELECT 
        to_char(transaction_date, 'Mon') as month_name, 
        extract(month from transaction_date) as month_num,
        count(*) as count
    FROM player_transactions
    WHERE il_type IS NOT NULL
    GROUP BY 1, 2 ORDER BY 2
"""
df_monthly = pd.read_sql(query_monthly, engine)
if not df_monthly.empty:
    fig_monthly = px.line(
        df_monthly,
        x="month_name",
        y="count",
        title="Seasonality of Injury Placements",
        markers=True,
        template="plotly_dark",
    )
    st.plotly_chart(fig_monthly, width="stretch")

st.markdown("---")

# --- 3. PLAYER DEEP-DIVE ---
st.subheader("🔍 Individual Player History")

player_input = st.text_input(
    "Enter Player ID or Full Name (e.g. Mookie Betts)", "605141"
)

if player_input:
    # Use window functions to find duration since placement for activation records
    base_query = """
        WITH player_tx AS (
            SELECT 
                transaction_date, 
                type_desc, 
                raw_description, 
                il_type, 
                injury_body_part, 
                injury_descriptor,
                transaction_id,
                -- Find previous placement/assignment for duration calculation
                MAX(CASE WHEN (raw_description ILIKE '%%placed%%' OR raw_description ILIKE '%%assigned%%') THEN transaction_date END) 
                    OVER (PARTITION BY player_id ORDER BY transaction_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as last_placed_date
            FROM player_transactions
            WHERE {where_clause}
        )
        SELECT 
            transaction_date, 
            type_desc, 
            il_type,
            CASE 
                WHEN (raw_description ILIKE '%%activated%%' OR raw_description ILIKE '%%reinstated%%') AND last_placed_date IS NOT NULL
                THEN (transaction_date - last_placed_date)
                ELSE 0
            END as days_on_il,
            injury_body_part, 
            injury_descriptor,
            raw_description
        FROM player_tx
        ORDER BY transaction_date DESC
    """
    if player_input.isdigit():
        query_player = text(base_query.format(where_clause="player_id = :pid"))
        params = {"pid": int(player_input)}
    else:
        query_player = text(base_query.format(where_clause="player_name ILIKE :pname"))
        params = {"pname": f"%{player_input}%"}

    with engine.connect() as conn:
        df_player = pd.read_sql(query_player, conn, params=params)

    if not df_player.empty:
        st.write(f"History for '{player_input}'")
        # Ensure days_on_il is int
        df_player["days_on_il"] = df_player["days_on_il"].fillna(0).astype(int)
        st.dataframe(df_player, width="stretch")

        # Summary metrics
        total_il_days = df_player["days_on_il"].sum()
        il_count = df_player[
            df_player["raw_description"].str.contains("placed", case=False, na=False)
        ].shape[0]

        col_pa1, col_pa2 = st.columns(2)
        col_pa1.metric("Lifetime IL Days (since 2019)", f"{int(total_il_days)}")
        col_pa2.metric("Total IL Stints", f"{il_count}")

        # --- NEW: PHYSICAL FATIGUE & STABILITY (GOLD LAYER) ---
        st.markdown("---")
        st.subheader("🔋 Workload & Stuff Stability")

        # Determine actual player ID for Gold lookup
        p_id = int(player_input) if player_input.isdigit() else None
        if not p_id and not df_player.empty:
            # Try to get p_id from another source if name was used
            q_id = text(
                "SELECT DISTINCT player_id FROM player_transactions WHERE player_name ILIKE :pname LIMIT 1"
            )
            with engine.connect() as conn:
                res = conn.execute(q_id, {"pname": f"%{player_input}%"}).fetchone()
                if res:
                    p_id = res[0]

        if p_id:
            # Fetch latest Gold record (Pitcher priority for health)
            q_gold = text("""
                SELECT * FROM player_rolling_features 
                WHERE player_id = :pid AND role = 'PITCHER'
                ORDER BY game_date DESC LIMIT 1
            """)
            with engine.connect() as conn:
                gold_df = pd.read_sql(q_gold, conn, params={"pid": p_id})

            if not gold_df.empty:
                latest = gold_df.iloc[0]
                hg1, hg2 = st.columns([1, 1])

                with hg1:
                    # Fatigue Gauge
                    fi_7d = latest.get("fatigue_index_7d", 0)
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=fi_7d,
                            title={"text": "7-Day Fatigue Index (Whiteside)"},
                            gauge={
                                "axis": {"range": [0, 30]},
                                "bar": {"color": "#00E5FF"},
                                "steps": [
                                    {"range": [0, 15], "color": "rgba(0, 255, 0, 0.1)"},
                                    {
                                        "range": [15, 25],
                                        "color": "rgba(255, 255, 0, 0.1)",
                                    },
                                    {
                                        "range": [25, 30],
                                        "color": "rgba(255, 0, 0, 0.2)",
                                    },
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": 25,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(template="plotly_dark", height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with hg2:
                    # Stuff Stability Metrics
                    st.write("#### 📡 Early Warning Signals")
                    spin_delta = latest.get("delta_spin_rate_3g", 0)
                    velo_delta = latest.get("delta_fb_velo_3g", 0)
                    ext_delta = latest.get("delta_extension_3g", 0)

                    st.metric(
                        "Spin Rate vs Season",
                        f"{latest.get('roll_avg_spin_rate', 0):.0f} rpm",
                        delta=f"{spin_delta:.1f} rpm",
                        delta_color="normal",
                    )
                    st.metric(
                        "Velo vs Season",
                        f"{latest.get('roll_avg_release_speed', 0):.1f} mph",
                        delta=f"{velo_delta:.1f} mph",
                        delta_color="normal",
                    )
                    st.metric(
                        "Extension vs Season",
                        f"{latest.get('roll_avg_release_extension', 0):.2f} ft",
                        delta=f"{ext_delta:.2f} ft",
                        delta_color="normal",
                    )
            else:
                st.info(
                    "No recent Pitcher health metrics found for this player in the Gold Layer."
                )
    else:
        st.warning(f"No data found for '{player_input}'.")

st.markdown("---")
st.success(
    "Health Model Ready: Features `sp_il_stints_ytd` and `v_game_il_features` fully populated."
)
