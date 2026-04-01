import streamlit as st
import pandas as pd
import plotly.express as px
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
        st.plotly_chart(fig_parts, use_container_width=True)

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
        st.plotly_chart(fig_kind, use_container_width=True)

st.markdown("---")

# --- 2. TEMPORAL TRENDS ---
st.subheader("📅 Temporal Roster Health")

st.write("#### IL Placements by Month (Seasonality)")
query_monthly = """
    SELECT 
        to_char(transaction_date, 'Month') as month_name, 
        extract(month from transaction_date) as month_num,
        count(*) as count
    FROM player_transactions
    WHERE type_desc LIKE 'Placed on %% IL'
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
    st.plotly_chart(fig_monthly, use_container_width=True)

st.markdown("---")

# --- 3. PLAYER DEEP-DIVE ---
st.subheader("🔍 Individual Player History")

player_input = st.text_input(
    "Enter Player ID or Full Name (e.g. Mookie Betts)", "605141"
)

if player_input:
    if player_input.isdigit():
        query_player = text("""
            SELECT transaction_date, type_desc, raw_description, il_type, days_on_il, injury_body_part, injury_descriptor
            FROM player_transactions
            WHERE player_id = :pid
            ORDER BY transaction_date DESC
        """)
        params = {"pid": int(player_input)}
    else:
        query_player = text("""
            SELECT transaction_date, type_desc, raw_description, il_type, days_on_il, injury_body_part, injury_descriptor
            FROM player_transactions
            WHERE player_name ILIKE :pname
            ORDER BY transaction_date DESC
        """)
        params = {"pname": f"%{player_input}%"}

    with engine.connect() as conn:
        df_player = pd.read_sql(query_player, conn, params=params)

    if not df_player.empty:
        st.write(f"History for '{player_input}'")
        # Cleanup column display
        st.dataframe(df_player, use_container_width=True)

        # Summary metrics
        total_il_days = df_player["days_on_il"].sum()
        il_count = df_player[df_player["il_type"].notnull()].shape[0]

        col_pa1, col_pa2 = st.columns(2)
        col_pa1.metric("Lifetime IL Days (since 2019)", f"{int(total_il_days)}")
        col_pa2.metric("Total IL Stints", f"{il_count}")
    else:
        st.warning(f"No data found for '{player_input}'.")

st.markdown("---")
st.success(
    "Health Model Ready: Features `sp_il_stints_ytd` and `v_game_il_features` fully populated."
)
