import streamlit as st

st.title("🔮 Live Model Predictions")
st.markdown("---")
st.write("Today's scheduled games and model forecasts.")
st.table(
    {"Game": "ATL vs NYM", "Model": "xG-V1", "Prediction": "ATL ML", "EV %": "4.2%"}
)
