import streamlit as st
import pandas as pd
from datetime import datetime, date
from algomlb.db.session import get_engine
from algomlb.ui.styles import apply_premium_styles
from algomlb.ui.components.spray_charts import (
    plot_spray_chart,
    get_ballpark_selection_ui,
)

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

    # 4. Ballpark Selection (Centralized SOLID HELPER)
    ballpark_dims = get_ballpark_selection_ui(
        engine, native_id=int(game_row["ballpark_id"]), key_prefix="matchup"
    )


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
        f"Final Score: {game_row['away_team']} {game_row['away_score']} - {game_row['home_score']} {game_row['home_team']} | Venue: {ballpark_dims['name'] if ballpark_dims else 'Unknown'}"
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
        st.plotly_chart(fig_away, width="stretch")

    with col_home:
        st.subheader(f"🏹 {TEAM_MAP.get(home_team, home_team)} Hits")
        df_home = df_events[df_events["inning_topbot"].str.lower() == "bot"]

        fig_home = plot_spray_chart(
            df_home,
            title=f"{home_team} Hits",
            color_col=color_col,
            ballpark_dims=ballpark_dims,
        )
        st.plotly_chart(fig_home, width="stretch")

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
        width="stretch",
    )

else:
    st.warning("No Statcast hit data available for this game PK.")

st.success("Visual Data Engine: 100% Online")
