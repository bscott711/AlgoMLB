import streamlit as st
import pandas as pd
from datetime import datetime, date
from algomlb.db.session import get_engine
from algomlb.ui.styles import apply_premium_styles
from algomlb.ui.components.spray_charts import plot_spray_chart

# --- Configuration & Styling ---
st.set_page_config(page_title="Game Analytics Hub", layout="wide")
apply_premium_styles()
engine = get_engine()

TEAM_MAP = {
    "ATH": "Oakland Athletics",
    "OAK": "Oakland Athletics",
    "ATL": "Atlanta Braves",
    "AZ": "Arizona Diamondbacks",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "CWS": "Chicago White Sox",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants",
    "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
}

st.title("🏟️ Game Analytics Hub")
st.markdown("---")

# --- Optimized Sidebar (Date & Game Selection) ---
with st.sidebar:
    st.header("🔍 Matchup Controllers")

    # 1. Date Selection
    selected_date = st.date_input("Select Date", value=datetime(2025, 9, 28))

    # 2. Game Selection for Date
    @st.cache_data(ttl=600)
    def get_games_for_date(date: date, _engine):
        query = f"""
            SELECT game_id, away_team, home_team, away_score, home_score, ballpark_id
            FROM game_results 
            WHERE game_date = '{date}'
        """
        return pd.read_sql(query, _engine)

    games_df = get_games_for_date(selected_date, engine)

    if not games_df.empty:
        games_df["display"] = games_df.apply(
            lambda r: (
                f"{r['away_team']} @ {r['home_team']} ({r['away_score']}-{r['home_score']})"
            ),
            axis=1,
        )
        selected_game_display = st.selectbox(
            "Select Game", options=games_df["display"].tolist()
        )
        game_row = games_df[games_df["display"] == selected_game_display].iloc[0]
        selected_game_id = game_row["game_id"]
    else:
        st.warning(f"No games found for {selected_date}.")
        st.stop()

    st.divider()

    # 3. Visualization Settings
    color_mode = st.radio("Color By", ["Outcome", "Exit Velo"], horizontal=True)
    color_col = "events" if color_mode == "Outcome" else "launch_speed"

    # --- 4. Native Stadium Logic (RESTORED) ---
    @st.cache_data(ttl=3600)
    def get_stadium_dims(ballpark_id: int, _engine):
        query_bp = f"""
            SELECT ballpark, left_field, left_center, center_field, right_center, right_field, 
                   lf_wall_height, lc_wall_height, cf_wall_height, rc_wall_height, rf_wall_height 
            FROM ballparks WHERE id = {ballpark_id}
        """
        bp_df = pd.read_sql(query_bp, _engine)
        if bp_df.empty:
            return None
        row = bp_df.iloc[0]
        return {
            "lf": row["left_field"],
            "lc": row["left_center"],
            "cf": row["center_field"],
            "rc": row["right_center"],
            "rf": row["right_field"],
            "h_lf": row["lf_wall_height"],
            "h_lc": row["lc_wall_height"],
            "h_cf": row["cf_wall_height"],
            "h_rc": row["rc_wall_height"],
            "h_rf": row["rf_wall_height"],
            "name": row["ballpark"],
        }

    # Fetch native stadium by default
    native_dims = get_stadium_dims(int(game_row["ballpark_id"]), engine)

    st.subheader("🧪 Stadium Simulator")
    simulate_stadium = st.checkbox("Swap Ballpark Fences", value=False)

    if simulate_stadium:

        @st.cache_data(ttl=3600)
        def get_all_ballparks(_engine):
            return pd.read_sql("SELECT ballpark, id FROM ballparks", _engine)

        all_bp_df = get_all_ballparks(engine)
        target_ballpark = st.selectbox(
            "Target Ballpark", all_bp_df["ballpark"].sort_values().unique()
        )
        selected_bp_id = all_bp_df[all_bp_df["ballpark"] == target_ballpark].iloc[0][
            "id"
        ]
        ballpark_dims = get_stadium_dims(int(selected_bp_id), engine)
    else:
        ballpark_dims = native_dims


# --- Main Game Analysis ---
@st.cache_data(ttl=300)
def load_game_events(game_id: str, _engine):
    # Join with weather if possible
    query = f"""
        SELECT s.*, g.home_team as h_team, g.away_team as a_team
        FROM statcast_raw s
        JOIN game_results g ON CAST(s.game_pk AS TEXT) = g.game_id
        WHERE g.game_id = '{game_id}'
        AND s.events IS NOT NULL
    """
    return pd.read_sql(query, _engine)


df_events = load_game_events(selected_game_id, engine)

if not df_events.empty:
    home_team = game_row["home_team"]
    away_team = game_row["away_team"]

    st.header(
        f"🎽 Matchup Analysis: {TEAM_MAP.get(away_team, away_team)} @ {TEAM_MAP.get(home_team, home_team)}"
    )
    st.info(
        f"Final Score: {game_row['away_team']} {game_row['away_score']} - {game_row['home_score']} {game_row['home_team']} | Venue: {native_dims['name'] if native_dims else 'Unknown'}"
    )

    col_away, col_home = st.columns(2)

    with col_away:
        st.subheader(f"🏹 {TEAM_MAP.get(away_team, away_team)} Hits")
        # For away team, check inning_topbot? Or just team?
        # statcast_raw.inning_topbot: 'Top' for away team hits
        df_away = df_events[df_events["inning_topbot"].str.lower() == "top"]

        fig_away = plot_spray_chart(
            df_away,
            title=f"{away_team} Hits",
            color_col=color_col,
            ballpark_dims=ballpark_dims,
        )
        st.plotly_chart(fig_away, use_container_width=True)

    with col_home:
        st.subheader(f"🏹 {TEAM_MAP.get(home_team, home_team)} Hits")
        df_home = df_events[df_events["inning_topbot"].str.lower() == "bot"]

        fig_home = plot_spray_chart(
            df_home,
            title=f"{home_team} Hits",
            color_col=color_col,
            ballpark_dims=ballpark_dims,
        )
        st.plotly_chart(fig_home, use_container_width=True)

    st.markdown("### 📊 Game Event Log")
    st.dataframe(
        df_events[
            [
                "at_bat_number",
                "inning",
                "inning_topbot",
                "player_name",
                "launch_speed",
                "events",
            ]
        ].sort_values("at_bat_number"),
        use_container_width=True,
    )

else:
    st.warning("No Statcast hit data available for this game PK.")

st.success("Visual Data Engine: 100% Online")
