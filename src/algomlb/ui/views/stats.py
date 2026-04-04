import streamlit as st
import pandas as pd
from algomlb.db.session import get_engine
from algomlb.config.settings import get_settings
from algomlb.ui.styles import apply_premium_styles
from algomlb.ui.components.spray_charts import plot_spray_chart, get_ballpark_selection_ui
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

    # 0. Lab Mode
    lab_mode = st.radio(
        "Analysis Mode",
        ["Batter Analysis", "Pitcher Analysis"],
        index=0,
        horizontal=True,
    )
    st.divider()

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
    
    # NEW: Fetch home ballpark if player is selected in session state
    home_bp_id = None
    if "last_selected_player_id" in st.session_state:
        @st.cache_data(ttl=3600)
        def get_home_field_id(_player_id, _engine):
            # 1. Recent team code
            query = f"SELECT home_team FROM statcast_raw WHERE batter = {_player_id} OR pitcher = {_player_id} ORDER BY game_date DESC LIMIT 1"
            df = pd.read_sql(query, _engine)
            if df.empty: return None
            t_code = df.iloc[0]["home_team"]
            t_full = TEAM_MAP.get(t_code, t_code)
            # 2. Map to ballpark id (escape single quotes safely)
            safe_team_name = t_full.replace("'", "''")
            q_bp = f"SELECT id FROM ballparks WHERE team_name = '{safe_team_name}' LIMIT 1"
            df_bp = pd.read_sql(q_bp, _engine)
            return int(df_bp.iloc[0]["id"]) if not df_bp.empty else None
            
        home_bp_id = get_home_field_id(st.session_state.last_selected_player_id, engine)

    # 3. Ballpark Simulation (Centralized SOLID HELPER)
    ballpark_dims = get_ballpark_selection_ui(engine, native_id=home_bp_id, key_prefix="player_stats")


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
    # Filter by Lab Mode
    if lab_mode == "Pitcher Analysis":
        roster_ids_df = roster_ids_df[roster_ids_df["pitches_thrown"] > 0]
    else:
        roster_ids_df = roster_ids_df[roster_ids_df["batted_balls"] > 0]

    # Map IDs to Names from our Master Map
    roster_ids_df["id"] = roster_ids_df["player_id"]
    roster_ids_df["Player Name"] = (
        roster_ids_df["player_id"]
        .map(player_map)
        .fillna(roster_ids_df["player_id"].apply(lambda x: f"Player {x}"))
    )

    # Reorder columns for display
    display_df = roster_ids_df[["Player Name", "batted_balls", "pitches_thrown", "id"]]

    st.info("Select a player from the table below to load the Performance Lab.")
    selection = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # --- Sticky Selection Logic ---
    selection_data = selection.get("selection", {})
    selected_rows = selection_data.get("rows", [])
    
    selected_player_id = None
    selected_player_name = None

    if selection and selected_rows:
        # A. Priority 1: Direct User Click
        selected_idx = int(selected_rows[0])
        selected_player_id = int(display_df.iloc[selected_idx]["id"])
        selected_player_name = display_df.iloc[selected_idx]["Player Name"]
        st.session_state.last_selected_player_id = selected_player_id
    elif "last_selected_player_id" in st.session_state:
        # B. Priority 2: Sticky Session State (Auto-re-select in new season/team roster)
        prev_id = st.session_state.last_selected_player_id
        if prev_id in display_df["id"].values:
            selected_player_id = prev_id
            selected_player_name = display_df[display_df["id"] == prev_id]["Player Name"].iloc[0]
            
    # Exit if no player identified
    if not selected_player_id:
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

        if ballpark_dims and ballpark_dims.get("name"):
            st.caption(f"Visualizing Context: **{ballpark_dims['name']}**")

        # Spray Chart Component
        spray_df = (
            df_events[df_events["pitcher"] == selected_player_id]
            if lab_mode == "Pitcher Analysis"
            else df_events[df_events["batter"] == selected_player_id]
        )

        spray_title = (
            f"Hits Allowed by {selected_player_name} ({season})"
            if lab_mode == "Pitcher Analysis"
            else f"{selected_player_name} - {season} Hits"
        )

        fig_spray = plot_spray_chart(
            spray_df,
            title=spray_title,
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
