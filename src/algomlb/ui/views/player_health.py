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
        st.plotly_chart(fig_parts, width='stretch')

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
        st.plotly_chart(fig_kind, width='stretch')

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
    st.plotly_chart(fig_monthly, width='stretch')

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
        st.dataframe(df_player, width='stretch')

        # Summary metrics
        total_il_days = df_player["days_on_il"].sum()
        il_count = df_player[
            df_player["raw_description"].str.contains("placed", case=False, na=False)
        ].shape[0]

        col_pa1, col_pa2 = st.columns(2)
        col_pa1.metric("Lifetime IL Days (since 2019)", f"{int(total_il_days)}")
        col_pa2.metric("Total IL Stints", f"{il_count}")
    else:
        st.warning(f"No data found for '{player_input}'.")

st.markdown("---")
st.success(
    "Health Model Ready: Features `sp_il_stints_ytd` and `v_game_il_features` fully populated."
)
