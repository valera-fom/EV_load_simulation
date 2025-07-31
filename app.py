import streamlit as st

st.set_page_config(
    page_title="EV Load Simulation", 
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ EV Load Simulation Tool")
st.markdown("### Advanced Electric Vehicle Charging Load Analysis & Optimization")

st.write("""
**🔬 Comprehensive EV Load Simulation Platform**

This application simulates electric vehicle charging patterns and their impact on electrical grid load. 
It helps utilities, researchers, and planners understand how EV adoption affects power demand and optimize charging strategies.
""")

st.write("**🎯 Key Capabilities:**")
col1, col2 = st.columns(2)

with col1:
    st.write("• **Real-time Simulation:** 48-hour EV charging scenarios")
    st.write("• **Multiple Data Sources:** Real historical data or AI-generated synthetic curves")
    st.write("• **Advanced Optimization:** Time-of-use pricing, PV+battery, V2G, grid storage")
    st.write("• **Capacity Analysis:** Find optimal EV count for your infrastructure")

with col2:
    st.write("• **AI-Powered Optimization:** Gradient-based algorithms for peak demand reduction")
    st.write("• **Professional Reports:** Generate detailed PDF analysis reports")
    st.write("• **Interactive Visualization:** Real-time charts and load curve analysis")
    st.write("• **Scenario Management:** Pre-configured setups for quick testing")

st.write("**⚡ Optimization Strategies:**")
st.write("• **Smart Charging:** Shift charging to off-peak hours based on TOU rates")
st.write("• **PV + Battery:** Solar charging during day, discharge during peak demand")
st.write("• **Grid Battery:** Utility-scale storage for load balancing")
st.write("• **Vehicle-to-Grid (V2G):** Bidirectional charging for grid support")

st.info("🚀 **Get Started:** Navigate to the 'Main Simulation' page to begin your analysis.")

st.write("---")

st.write("**📊 Analysis Features:**")
st.write("• Grid load curve visualization with EV charging overlay")
st.write("• Peak demand analysis and optimization effects")
st.write("• Time-of-use charging pattern simulation")
st.write("• Maximum EV capacity calculation for given infrastructure")
st.write("• Strategy impact comparison (TOU, PV, V2G, grid battery)")

st.write("**🔧 Technical Capabilities:**")
st.write("• **Simulation Engine:** EV arrival/departure patterns with charging algorithms")
st.write("• **Data Processing:** Excel file import or AI-generated synthetic load curves")
st.write("• **Optimization:** Capacity analysis and gradient-based TOU optimization")
st.write("• **Output:** Interactive plots and downloadable PDF reports") 