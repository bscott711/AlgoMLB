import streamlit as st

# Setup navigation and page structure (Streamlit 1.29+)
st.set_page_config(page_title="AlgoMLB Dashboard", layout="wide")

pages = {
    "Live Analytics": [
        st.Page("views/picks.py", title="Live Model Picks", icon="🔮"),
        st.Page("views/bankroll.py", title="Bankroll & Performance", icon="💰"),
        st.Page("views/market.py", title="Market Analytics", icon="📈"),
    ],
    "Player & Game Deep-Dive": [
        st.Page("views/stats.py", title="Player Performance", icon="⚾"),
        st.Page("views/umpires.py", title="Umpire Analytics", icon="👨‍⚖️"),
        st.Page("views/ballparks.py", title="Ballpark Context", icon="🏟️"),
    ],
    "ML Engineering": [
        st.Page("views/optuna.py", title="Optuna Studies", icon="🧪"),
    ],
    "System Health": [
        st.Page("views/data.py", title="Data & Ingest Health", icon="📡"),
    ],
}

pg = st.navigation(pages)
pg.run()
