import streamlit as st

st.set_page_config(page_title="EV Simulation", layout="wide")

st.title("⚡ EV Load Forecasting")
st.markdown("Welcome to the EV Load Forecasting application!")

st.write("""
This application allows you to:
- Configure EV and charger parameters
- Set up time-based arrival patterns
- Apply Portugal EV scenarios
- Use optimization strategies (Smart Charging, PV + Battery, Grid-Charged Batteries)
- Run simulations and view results
""")

st.info("Navigate to the Forecasting page to start using the application.") 