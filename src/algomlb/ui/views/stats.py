import streamlit as st
import pandas as pd
from algomlb.db.session import get_engine
from algomlb.config.settings import get_settings
from algomlb.ui.styles import apply_premium_styles
from algomlb.ui.components.spray_charts import plot_spray_chart
from algomlb.ui.components.strike_zone import plot_strike_zone
from algomlb.ui.components.rolling_trends import (
    load_rolling_features,
    plot_rolling_trend,
)

# --- Configuration & Styling ---
st.set_page_config(page_title="Player Performance Lab", layout="wide")
apply_premium_styles()
engine = get_engine()
settings = get_settings()

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

st.title("⚾ Player Performance Lab")
st.markdown("---")

# --- Optimized Sidebar (Team-First Selection) ---
with st.sidebar:
    st.header("🔍 Lab Controllers")

    # 1. Team Selection
    @st.cache_data(ttl=3600)
    def get_teams(_engine):
        query = "SELECT DISTINCT home_team FROM statcast_raw ORDER BY 1"
        return pd.read_sql(query, _engine)["home_team"].tolist()

    team_codes = get_teams(engine)
    team_display = {code: TEAM_MAP.get(code, code) for code in team_codes}

    selected_team_display = st.selectbox(
        "Select Team",
        options=list(team_display.values()),
        index=list(team_display.values()).index(
            team_display.get("LAD", list(team_display.values())[0])
        ),
    )
    # Reverse map back to code
    selected_team = [
        code
        for code, display in team_display.items()
        if display == selected_team_display
    ][0]

    # 2. Season/Date Filter
    season = st.slider("Season", 2019, 2026, 2026)  # Expanded to 2026

    # 3. Hit Simulator (The "What-If" Engine)
    st.divider()
    st.subheader("🧪 Hit Simulator")
    simulate_stadium = st.checkbox("Swap Ballpark Fences", value=False)
    target_ballpark = "Dodger Stadium"
    if simulate_stadium:
        # Cache ballpark dims
        @st.cache_data(ttl=3600)
        def get_ballparks(_engine):
            query_bp = """
                SELECT ballpark, left_field, left_center, center_field, right_center, right_field,
                       lf_wall_height, lc_wall_height, cf_wall_height, rc_wall_height, rf_wall_height
                FROM ballparks
            """
            return pd.read_sql(query_bp, _engine)

        ballparks_df = get_ballparks(engine)
        target_ballpark = st.selectbox(
            "Target Ballpark", ballparks_df["ballpark"].sort_values().unique()
        )
        bp_row = ballparks_df[ballparks_df["ballpark"] == target_ballpark].iloc[0]
        ballpark_dims = {
            "lf": bp_row["left_field"],
            "lc": bp_row["left_center"],
            "cf": bp_row["center_field"],
            "rc": bp_row["right_center"],
            "rf": bp_row["right_field"],
            "h_lf": bp_row["lf_wall_height"],
            "h_lc": bp_row["lc_wall_height"],
            "h_cf": bp_row["cf_wall_height"],
            "h_rc": bp_row["rc_wall_height"],
            "h_rf": bp_row["rf_wall_height"],
        }
    else:
        ballpark_dims = None


# --- Master Player Metadata Map ---
@st.cache_data(ttl=3600)
def get_player_name_map(_engine):
    """
    Building a robust ID -> Name mapping by combining multiple sources.
    This resolves the duplication where pitcher names appear for batter IDs.
    """
    # Source A: Pitcher names from Statcast (ID-to-Name is guaranteed here)
    q_pit = "SELECT DISTINCT pitcher as id, player_name FROM statcast_raw WHERE player_name IS NOT NULL"
    df_pit = pd.read_sql(q_pit, _engine)

    # Source B: Transaction history (High coverage for both batters/pitchers)
    q_tx = "SELECT DISTINCT player_id as id, player_name FROM player_transactions WHERE player_name IS NOT NULL"
    df_tx = pd.read_sql(q_tx, _engine)

    # Merge and prioritize TX names (often more 'official' full names)
    full_map = pd.concat([df_tx, df_pit]).drop_duplicates(subset=["id"], keep="first")
    return full_map.set_index("id")["player_name"].to_dict()


player_map = get_player_name_map(engine)


# --- Main Player Selection Hub ---
@st.cache_data(ttl=600)
def get_players_by_team(team: str, season: int, _engine):
    """Retrieves roster for a team in a given season with usage counts."""
    # We query IDs from statcast_raw (the subjects of the performance views)
    # We ignore the 'player_name' column here as it's correctly mapped in get_player_name_map
    query = f"""
        SELECT 
            player_id,
            sum(is_batter) as batted_balls,
            sum(is_pitcher) as pitches_thrown
        FROM (
            SELECT batter as player_id, 1 as is_batter, 0 as is_pitcher 
            FROM statcast_raw 
            WHERE (home_team = '{team}' OR away_team = '{team}') 
            AND EXTRACT(YEAR FROM game_date) = {season}
            AND events IS NOT NULL
            UNION ALL
            SELECT pitcher as player_id, 0 as is_batter, 1 as is_pitcher 
            FROM statcast_raw 
            WHERE (home_team = '{team}' OR away_team = '{team}') 
            AND EXTRACT(YEAR FROM game_date) = {season}
        ) sub
        GROUP BY 1
        ORDER BY batted_balls DESC, pitches_thrown DESC
    """
    return pd.read_sql(query, _engine)


st.subheader(f"👥 {TEAM_MAP.get(selected_team, selected_team)} Roster ({season})")
roster_ids_df = get_players_by_team(selected_team, season, engine)

if not roster_ids_df.empty:
    # Map IDs to Names from our Master Map
    roster_ids_df["id"] = roster_ids_df["player_id"]
    roster_ids_df["Player Name"] = (
        roster_ids_df["player_id"]
        .map(player_map)
        .fillna(roster_ids_df["player_id"].apply(lambda x: f"Player {x}"))
    )

    # Reorder columns for display
    display_df = roster_ids_df[["Player Name", "id", "batted_balls", "pitches_thrown"]]

    st.info("Select a player from the table below to load the Performance Lab.")
    selection = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Get selected ID from selection state
    selection_data = selection.get("selection", {})
    selected_rows = selection_data.get("rows", [])
    if selection and selected_rows:
        selected_idx = int(selected_rows[0])
        selected_player_id = int(display_df.iloc[selected_idx]["id"])
        selected_player_name = display_df.iloc[selected_idx]["Player Name"]
    else:
        st.warning("Please select a player row in the table above.")
        st.stop()
else:
    st.error(
        f"No Statcast data found for {selected_team} in {season}. Try another season or team."
    )
    st.stop()


# --- Data Loading ---
@st.cache_data(ttl=300)
def load_player_events(player_id: int, season: int, _engine):
    query = f"""
        SELECT * FROM statcast_raw 
        WHERE (batter = {player_id} OR pitcher = {player_id})
        AND EXTRACT(YEAR FROM game_date) = {season}
    """
    return pd.read_sql(query, _engine)


df_events = load_player_events(selected_player_id, season, engine)

# --- Dashboard Layout (Only if player selected) ---
st.markdown(f"## 🧬 Lab Analysis: {selected_player_name}")

tab_overview, tab_spray, tab_trends = st.tabs(
    ["📊 Performance Overview", "🏹 Spray & Zone Lab", "📈 Rolling Trends"]
)

with tab_overview:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    is_pitcher = len(df_events[df_events["pitcher"] == selected_player_id]) > len(
        df_events[df_events["batter"] == selected_player_id]
    )

    if is_pitcher:
        avg_velo = df_events["release_speed"].mean()
        whiff_rate = (
            len(
                df_events[
                    df_events["description"].str.contains("swinging_strike", na=False)
                ]
            )
            / len(df_events)
            if len(df_events) > 0
            else 0
        )
        col_m1.metric("Avg Velocity", f"{avg_velo:.1f} mph" if avg_velo else "N/A")
        col_m2.metric("Whiff %", f"{whiff_rate:.1%}")
    else:
        max_ev = df_events["launch_speed"].max()
        avg_la = df_events["launch_angle"].mean()
        col_m1.metric("Max Exit Velo", f"{max_ev:.1f} mph" if max_ev else "N/A")
        col_m2.metric("Avg Launch Angle", f"{avg_la:.1f}°" if avg_la else "N/A")

    st.dataframe(
        df_events[
            [
                "game_date",
                "at_bat_number",
                "pitch_number",
                "pitch_type",
                "launch_speed",
                "events",
            ]
        ].head(100),
        use_container_width=True,
    )

with tab_spray:
    col_left, col_right = st.columns([1.2, 0.8])

    with col_left:
        st.subheader("🏹 Spray Chart")

        c1, c2 = st.columns([1, 1])
        with c1:
            color_mode = st.radio(
                "Color By",
                ["Exit Velo", "Outcome"],
                horizontal=True,
                key="spray_color_mode",
            )

        color_col = "launch_speed" if color_mode == "Exit Velo" else "events"

        if simulate_stadium:
            st.caption(f"Simulating outcomes for: **{target_ballpark}**")

        # Spray Chart Component
        fig_spray = plot_spray_chart(
            df_events[df_events["batter"] == selected_player_id],
            title=f"{selected_player_name} - {season} Hits",
            color_col=color_col,
            ballpark_dims=ballpark_dims,
        )
        st.plotly_chart(fig_spray, use_container_width=True)

    with col_right:
        st.subheader("🎯 Strike Zone Lab")
        fig_zone = plot_strike_zone(df_events, title="Pitch Locations & Quality")
        st.plotly_chart(fig_zone, use_container_width=True)

with tab_trends:
    st.subheader("📈 Rolling Feature Workbench")

    role = "PITCHER" if is_pitcher else "BATTER"
    df_rolling = load_rolling_features(selected_player_id, role, engine)

    if not df_rolling.empty:
        metric_to_plot = (
            "roll_avg_pitcher_xwoba" if is_pitcher else "roll_avg_batter_xwoba"
        )
        league_mean = (
            getattr(settings.ml, "league_mean_pitcher_xwoba", 0.320)
            if is_pitcher
            else getattr(settings.ml, "league_mean_batter_xwoba", 0.320)
        )

        fig_trend = plot_rolling_trend(
            df_rolling,
            metric=metric_to_plot,
            league_mean=league_mean,
            title=f"Rolling xwOBA Trend (Shrinkage Baseline: {league_mean})",
        )
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info(
            "No rolling Gold layer features found for this player. Ensure the backfill has processed this player_id."
        )

st.success("Visual Data Engine: 100% Online")
