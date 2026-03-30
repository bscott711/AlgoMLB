import streamlit as st

st.title("🧪 Optuna Hyperparameter Studies")
st.markdown("---")
market = st.selectbox("Select Betting Market", ["moneyline", "total", "runline"])
st.info(
    f"Visualizing studies for market: {market}. TODO: Load real studies via Postgres storage."
)
