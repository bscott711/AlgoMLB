import streamlit as st

# Setup navigation and page structure (Streamlit 1.29+)
st.set_page_config(page_title="AlgoMLB Dashboard", layout="wide")

pages = {
    "Live Analytics": [
        st.Page("pages/picks.py", title="Live Model Picks", icon="🔮"),
        st.Page("pages/bankroll.py", title="Bankroll & Performance", icon="💰"),
    ],
    "ML Engineering": [
        st.Page("pages/optuna.py", title="Optuna Studies", icon="🧪"),
    ],
    "System Health": [
        st.Page("pages/data.py", title="Data & Ingest Health", icon="📡"),
    ],
}

pg = st.navigation(pages)
pg.run()
