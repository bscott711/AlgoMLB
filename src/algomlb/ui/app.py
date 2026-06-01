import streamlit as st
import hmac
from algomlb.ui.styles import apply_premium_styles

# Setup navigation and page structure (Streamlit 1.29+)
st.set_page_config(page_title="AlgoMLB Command Center", page_icon="📡", layout="wide")
apply_premium_styles()

def check_password_sidebar():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets.get("password", "")):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    with st.sidebar.expander("Admin Login", expanded=False):
        st.text_input(
            "Enter Admin Password", type="password", on_change=password_entered, key="password"
        )
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😕 Password incorrect")
    return False

is_admin = check_password_sidebar()

# Hero Header
st.title("📡 AlgoMLB Command Center")
st.markdown("---")

pages = {
    "Live Analytics": [
        st.Page("views/picks.py", title="Live Model Picks", icon="🔮"),
        st.Page("views/bankroll.py", title="Bankroll & Performance", icon="💰"),
        st.Page("views/market.py", title="Market Analytics", icon="📈"),
    ],
    "Player & Game Deep-Dive": [
        st.Page("views/stats.py", title="Player Performance", icon="⚾"),
        st.Page("views/games.py", title="Game Analytics Hub", icon="🏟️"),
        st.Page("views/player_health.py", title="Player Health & IL", icon="🩹"),
        st.Page("views/umpires.py", title="Umpire Analytics", icon="👨‍⚖️"),
        st.Page("views/ballparks.py", title="Ballpark Context", icon="🏟️"),
    ],
    "Simulation Lab": [
        st.Page("views/simulation_lab.py", title="Monte Carlo Engine", icon="🏗️"),
    ]
}

if is_admin:
    pages["ML Engineering"] = [
        st.Page("views/optuna.py", title="Optuna Studies", icon="🧪"),
        st.Page("views/model_performance.py", title="Model Performance", icon="📊"),
    ]
    pages["System Health"] = [
        st.Page("views/backfill_status.py", title="Backfill & Orchestration", icon="🔋"),
        st.Page("views/data.py", title="Data & Ingest Health", icon="📡"),
        st.Page("views/weather.py", title="Weather Insights", icon="🌦️"),
        st.Page("views/db_explorer.py", title="Database Explorer", icon="🔍"),
    ]
    pages["Engineering Docs"] = [
        st.Page("views/database_relationships.py", title="Database Architecture", icon="🏗️"),
    ]

pg = st.navigation(pages)
pg.run()
