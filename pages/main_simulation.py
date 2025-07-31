import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import warnings
import sys
import os
import math
import time
import streamlit.components.v1 as components
warnings.filterwarnings('ignore')

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize random seed in session state if not exists
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42

try:
    from sim_setup import SimulationSetup
except ImportError as e:
    st.error(f"Failed to import SimulationSetup: {e}")
    st.stop()

try:
    from EV import EV, EV_MODELS
except ImportError as e:
    st.error(f"Failed to import EV: {e}")
    st.stop()

try:
    from charger import CHARGER_MODELS
except ImportError as e:
    st.error(f"Failed to import charger: {e}")
    st.stop()
try:
    from pages.components.capacity_analyzer import find_max_cars_capacity
except ImportError as e:
    st.error(f"Failed to import capacity_analyzer: {e}")
    st.stop()

# Import the new TOU optimizer
try:
    from pages.components.tou_optimizer import (
        optimize_tou_periods_24h, 
        convert_to_simulation_format, 
        validate_periods
    )
except ImportError as e:
    st.error(f"Failed to import tou_optimizer: {e}")
    st.stop()
# Import optimization components

def _is_light_color(hex_color):
    """Helper function to determine if a hex color is light or dark for text contrast."""
    # Handle transparent and non-hex colors
    if hex_color == 'transparent' or not hex_color.startswith('#'):
        return True  # Default to black text for transparent/non-hex colors
    
    # Remove # if present
    hex_color = hex_color.lstrip('#')
    
    # Convert to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Calculate luminance
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    return luminance > 0.5

# Portugal EV Scenarios dictionary
portugal_ev_scenarios = {
    "substation_count": 69000,
    
    "2027": {
        "total_cars_million": 6.2,
        "total_evs_million": 5.6,
        "home_charging_percent": 0.90,  # 90%
        "home_charged_evs_million": 5.0,
        "smart_charging_percent": 30,  # 30% (limited smart charging)
        "pv_adoption_percent": 15,  # 15% (early adoption)
        "battery_capacity": 12.5,
        "discharge_rate": 5.0,
        "solar_energy_percent": 70,
        "notes": "Most EVs are still early adopters with home access",
        "scenarios": {
            "conservative": {
                "ev_penetration": 0.20,  # 20%
                "avg_battery_kWh": 70,
                "charging_power_kW": 7.4
            },
            "realistic": {
                "ev_penetration": 0.25,
                "avg_battery_kWh": 75,
                "charging_power_kW": 9
            },
            "aggressive": {
                "ev_penetration": 0.30,
                "avg_battery_kWh": 80,
                "charging_power_kW": 11
            }
        }
    },

    "2030": {
        "total_cars_million": 6.3,
        "total_evs_million": 5.5,
        "home_charging_percent": 0.88,  # 88%
        "home_charged_evs_million": 4.8,
        "smart_charging_percent": 40,  # 40% (moderate smart charging adoption)
        "pv_adoption_percent": 25,  # 25% (moderate adoption)
        "battery_capacity": 17.5,
        "discharge_rate": 7.0,
        "solar_energy_percent": 70,
        "notes": "Growth of fleet electrification; small shift to public charging",
        "scenarios": {
            "conservative": {
                "ev_penetration": 0.35,
                "avg_battery_kWh": 80,
                "charging_power_kW": 9
            },
            "realistic": {
                "ev_penetration": 0.50,
                "avg_battery_kWh": 85,
                "charging_power_kW": 11
            },
            "aggressive": {
                "ev_penetration": 0.65,
                "avg_battery_kWh": 95,
                "charging_power_kW": 15
            }
        }
    },

    "2035": {
        "total_cars_million": 5.9,
        "total_evs_million": 5.0,
        "home_charging_percent": 0.85,  # 85%
        "home_charged_evs_million": 4.3,
        "smart_charging_percent": 60,  # 60% (widespread smart charging deployment)
        "pv_adoption_percent": 35,  # 35% (widespread adoption)
        "battery_capacity": 25.0,
        "discharge_rate": 10.0,
        "solar_energy_percent": 70,
        "notes": "Urban densification, more flats; workplace & public charging expand",
        "scenarios": {
            "conservative": {
                "ev_penetration": 0.55,
                "avg_battery_kWh": 90,
                "charging_power_kW": 11
            },
            "realistic": {
                "ev_penetration": 0.70,
                "avg_battery_kWh": 105,
                "charging_power_kW": 15
            },
            "aggressive": {
                "ev_penetration": 0.85,
                "avg_battery_kWh": 120,
                "charging_power_kW": 22
            }
        }
    },

    "2040": {
        "total_cars_million": 5.4,
        "total_evs_million": 4.4,
        "home_charging_percent": 0.82,  # 82%
        "home_charged_evs_million": 3.6,
        "smart_charging_percent": 60,  # 60% (widespread smart charging deployment)
        "pv_adoption_percent": 45,  # 45% (high adoption)
        "battery_capacity": 30.0,
        "discharge_rate": 12.0,
        "solar_energy_percent": 70,
        "notes": "Urban mobility plans reduce private ownership",
        "scenarios": {
            "conservative": {
                "ev_penetration": 0.70,
                "avg_battery_kWh": 100,
                "charging_power_kW": 22
            },
            "realistic": {
                "ev_penetration": 0.85,
                "avg_battery_kWh": 130,
                "charging_power_kW": 33
            },
            "aggressive": {
                "ev_penetration": 1.00,
                "avg_battery_kWh": 170,
                "charging_power_kW": 44
            }
        }
    },

    "2045": {
        "total_cars_million": 4.9,
        "total_evs_million": 3.9,
        "home_charging_percent": 0.80,  # 80%
        "home_charged_evs_million": 3.1,
        "smart_charging_percent": 60,  # 60% (widespread smart charging deployment)
        "pv_adoption_percent": 55,  # 55% (very high adoption)
        "battery_capacity": 35.0,
        "discharge_rate": 15.0,
        "solar_energy_percent": 70,
        "notes": "Shared mobility, better public transport, bidirectional grid",
        "scenarios": {
            "conservative": {
                "ev_penetration": 0.80,
                "avg_battery_kWh": 110,
                "charging_power_kW": 22
            },
            "realistic": {
                "ev_penetration": 0.95,
                "avg_battery_kWh": 140,
                "charging_power_kW": 33
            },
            "aggressive": {
                "ev_penetration": 1.00,
                "avg_battery_kWh": 170,
                "charging_power_kW": 44
            }
        }
    }
}

# Initialize session state variables
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'simulation_just_run' not in st.session_state:
    st.session_state.simulation_just_run = False

# Initialize session state for single dynamic EV
if 'dynamic_ev' not in st.session_state:
    st.session_state.dynamic_ev = {
        'capacity': 50.0,
        'AC': 11.0,
        'DC': 50.0,
        'quantity': 10
    }

st.set_page_config(page_title="Forecasting", layout="wide")
st.title("⚡ EV Load Forecasting")
st.markdown("Forecast EV charging load patterns with configurable parameters")

# Reset button
if st.sidebar.button("⭮ Reset", type="primary", help="Reset all configuration to default values"):
    # Reset dynamic EV
    st.session_state.dynamic_ev = {
        'capacity': 50.0,
        'AC': 11.0
    }
    
    # Reset time peaks
    st.session_state.time_peaks = [
        {
            'name': 'Evening Peak',
            'time': 19,
            'span': 1.5,
            'quantity': 5,
            'enabled': True
        }
    ]
    
    # Reset charger config (this is what the UI actually uses)
    st.session_state.charger_config = {
        'ac_rate': 11.0,
        'ac_count': 4
    }
    
    # Clear simulation run flag
    if 'simulation_run' in st.session_state:
        del st.session_state.simulation_run
    
    st.rerun()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # EV Configuration (single dynamic EV)
    with st.expander("🚗 EV Configuration", expanded=False):
        st.write("**Dynamic EV Parameters:**")
        ev_capacity = st.number_input("Battery Capacity (kWh)", min_value=1.0, max_value=1000.0, 
                                     value=float(st.session_state.dynamic_ev['capacity']), step=1.0)
        ev_ac = st.number_input("AC Charging Rate (kW)", min_value=0.0, max_value=100.0, 
                               value=float(st.session_state.dynamic_ev['AC']), step=1.0)
        
        # Initialize SOC in session state if not exists
        if 'ev_soc' not in st.session_state:
            st.session_state.ev_soc = 0.2  # Default 20% SOC
        
        ev_soc = st.slider("Initial State of Charge (%)", min_value=5, max_value=95, 
                           value=int(st.session_state.ev_soc * 100), step=5,
                           help="Initial SOC of arriving EVs (affects how much they need to charge)")
        
        # Clear simulation results if parameters changed
        if (ev_capacity != st.session_state.dynamic_ev['capacity'] or 
            ev_ac != st.session_state.dynamic_ev['AC'] or
            ev_soc/100 != st.session_state.ev_soc):
            st.session_state.simulation_just_run = False
            # Don't clear simulation_results - let them persist

    # Charger Configuration
    with st.expander("🔌 Charger Configuration", expanded=False):
        # Initialize charger config in session state if not exists
        if 'charger_config' not in st.session_state:
            st.session_state.charger_config = {
                'ac_rate': 11.0,
                'ac_count': 4,
                'dc_rate': 50.0,
                'dc_count': 0,
                'fast_dc_rate': 200.0,
                'fast_dc_count': 0
            }
        
        st.write("**AC Chargers:**")
        ac_rate = st.number_input("AC Charging Rate (kW)", min_value=1.0, max_value=100.0, 
                                 value=float(st.session_state.charger_config['ac_rate']), step=1.0)
        ac_count = st.number_input("AC Charger Count", min_value=0, 
                                  value=int(st.session_state.charger_config['ac_count']))
        
        # Update session state with new charger values
        st.session_state.charger_config['ac_rate'] = ac_rate
        st.session_state.charger_config['ac_count'] = ac_count
        
        # Clear simulation results if charger parameters changed
        if (ac_rate != st.session_state.charger_config.get('ac_rate', ac_rate) or 
            ac_count != st.session_state.charger_config.get('ac_count', ac_count)):
            st.session_state.simulation_just_run = False
            # Don't clear simulation_results - let them persist

    # Time Control Menu
    with st.expander("⏰ Time Control", expanded=False):
        st.write("**Simulation Duration:**")
        sim_duration = st.slider("Duration (hours)", min_value=1, max_value=48, value=36)
        st.session_state.sim_duration = sim_duration  # Store for plotting
        
        st.write("**Peak Configuration:**")
        
        # Initialize peaks in session state if not exists
        if 'time_peaks' not in st.session_state:
            st.session_state.time_peaks = [
                {
                    'name': 'Evening Peak',
                    'time': 19,
                    'span': 1.5,
                    'quantity': 5,
                    'enabled': True
                }
            ]
        
        # Add Peak button and Total EVs at the top
        col_add_peak, col_total_evs = st.columns([1, 1])
        
        with col_add_peak:
            if st.button("➕ Add Peak", type="secondary"):
                # Add a new peak with 0 cars and same parameters as first peak
                if st.session_state.time_peaks:
                    first_peak = st.session_state.time_peaks[0]
                    new_peak = {
                        'name': f'Peak {len(st.session_state.time_peaks) + 1}',
                        'time': first_peak.get('time', 19),
                        'span': first_peak.get('span', 1.5),
                        'quantity': 0,  # Start with 0 cars
                        'enabled': True
                    }
                    st.session_state.time_peaks.append(new_peak)
                    st.rerun()
        
        with col_total_evs:
            total_evs = sum(peak.get('quantity', 0) for peak in st.session_state.time_peaks)
            st.metric("Total EVs", f"{total_evs} EVs")
        
        # Display all peaks
        for i, peak in enumerate(st.session_state.time_peaks):
            # Get current peak data for immediate title update
            current_peak = st.session_state.time_peaks[i]
            peak_name = current_peak.get('name', f'Peak {i+1}')
            
            # Create header with delete button for non-first peaks
            if i == 0:
                st.write(f"**📊 {peak_name}**")
            else:
                col_header, col_delete = st.columns([3, 1])
                with col_header:
                    st.write(f"**📊 {peak_name}**")
                with col_delete:
                    if st.button("🗑️ Delete", key=f"delete_peak_{i}", type="secondary"):
                        st.session_state.time_peaks.pop(i)
                        st.rerun()
            
            with st.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    def update_peak_name(i=i):
                        # Check if the session state key exists before accessing it
                        key = f"peak_name_{i}"
                        if key in st.session_state:
                            st.session_state.time_peaks[i]['name'] = st.session_state[key]
                        
                    
                    new_name = st.text_input("Peak Name", value=str(peak.get('name', f'Peak {i+1}')), key=f"peak_name_{i}", on_change=update_peak_name)
                    new_time = st.slider("Peak Time (hours)", min_value=0, max_value=48, value=int(peak.get('time', 19)), key=f"peak_time_{i}")
                
                with col2:
                    # Ensure span is a float
                    span_value = float(peak.get('span', 1.5)) if peak.get('span') is not None else 1.5
                    new_span = st.slider("Time Span (1σ in hours)", min_value=0.5, max_value=12.0, value=span_value, step=0.5, key=f"peak_span_{i}", help="Spread of arrival times (1 standard deviation)")
                    
                    def update_peak_quantity(i=i):
                        # Check if the session state key exists before accessing it
                        key = f"peak_quantity_{i}"
                        if key in st.session_state:
                            st.session_state.time_peaks[i]['quantity'] = st.session_state[key]
                    
                    new_quantity = st.number_input("EV Quantity", min_value=0, max_value=1000, value=int(peak.get('quantity', 5)), key=f"peak_quantity_{i}", on_change=update_peak_quantity)
                
                # Update peak data
                st.session_state.time_peaks[i] = {
                    'name': new_name,
                    'time': new_time,
                    'span': new_span,
                    'quantity': new_quantity,
                    'enabled': True  # Always enabled
                }

    # Portugal EV Scenarios Presets
    with st.expander("🎯 Portugal EV Scenarios", expanded=False):
        st.write("**Select year and scenario to automatically set EV parameters:**")
        st.caption("⚠️ Note: Applying a scenario will change all EV and charger parameters to match the selected scenario.")
        
        # Scenario type toggle
        scenario_type = st.radio(
            "Scenario Type",
            options=["Built-in Scenarios", "Custom Scenarios"],
            horizontal=True,
            help="Choose between built-in Portugal scenarios or your custom scenarios"
        )
        
        # Initialize scenario values in session state if not exists
        if 'portugal_scenario' not in st.session_state:
            st.session_state.portugal_scenario = {
                'year': '2030',
                'scenario': 'realistic',
            }
        
        if scenario_type == "Built-in Scenarios":
            # Built-in scenarios section with dropdowns in first row
            col1, col2 = st.columns([1, 1])
            
            with col1:
                selected_year = st.selectbox(
                    "Year",
                    options=list(portugal_ev_scenarios.keys())[1:],
                    index=list(portugal_ev_scenarios.keys())[1:].index(st.session_state.portugal_scenario['year']) if st.session_state.portugal_scenario['year'] in list(portugal_ev_scenarios.keys())[1:] else 0,
                    help="Select the target year for the scenario"
                )
            
            with col2:
                # Get available scenarios for selected year
                available_scenarios = ['conservative', 'realistic', 'aggressive']
                
                selected_scenario = st.selectbox(
                    "Scenario",
                    options=available_scenarios,
                    index=available_scenarios.index(st.session_state.portugal_scenario['scenario']) if st.session_state.portugal_scenario['scenario'] in available_scenarios else 1,
                    help="Select the scenario type"
                )
            
                        # Checkbox in second row
            show_builtin_summary = st.checkbox("📊 Show Summary", value=False, help="Toggle to show/hide the detailed scenario summary", key="show_builtin_summary")
            
            # Get scenario data
            scenario_data = portugal_ev_scenarios[selected_year]['scenarios'][selected_scenario]
            year_data = portugal_ev_scenarios[selected_year]
            
            # Calculate EVs per substation
            total_cars = year_data['total_cars_million'] * 1_000_000
            substation_count = portugal_ev_scenarios['substation_count']
            evs_per_substation = (total_cars * scenario_data['ev_penetration']) / substation_count
            
            # Calculate scenario year for optimization strategies
            scenario_year_int = int(selected_year)
            
            # Smart charging adoption based on scenario year
            if scenario_year_int <= 2027:
                smart_charging_percent = 20
            elif scenario_year_int <= 2030:
                smart_charging_percent = 40
            else:
                smart_charging_percent = 60
            
            # Grid battery adoption based on scenario year
            if scenario_year_int <= 2027:
                grid_battery_adoption = 5
            elif scenario_year_int <= 2030:
                grid_battery_adoption = 15
            else:
                grid_battery_adoption = 25
            
            # V2G adoption based on scenario year
            if scenario_year_int <= 2027:
                v2g_adoption_percent = 5
            elif scenario_year_int <= 2030:
                v2g_adoption_percent = 15
            else:
                v2g_adoption_percent = 25
            
            v2g_discharge_rate = scenario_data['charging_power_kW']
            
            # Apply scenario button
            if st.button("🚀 Apply Scenario", type="primary"):
                # Update EV configuration
                st.session_state.dynamic_ev = {
                    'capacity': scenario_data['avg_battery_kWh'],
                    'AC': scenario_data['charging_power_kW']
                }
                
                # Set default SOC for scenario (20% for realistic scenarios)
                st.session_state.ev_soc = 0.2
                
                # Update charger configuration
                st.session_state.dynamic_charger = {
                    'ac_rate': scenario_data['charging_power_kW'],
                    'ac_count': math.ceil(evs_per_substation),  # Number of chargers equals number of EVs
                }
                
                # Also update charger_config to match (used by simulation)
                st.session_state.charger_config = {
                    'ac_rate': scenario_data['charging_power_kW'],
                    'ac_count': math.ceil(evs_per_substation),  # Number of chargers equals number of EVs
                }
                
                # Update time peaks configuration
                st.session_state.time_peaks = [
                    {
                        'name': 'Evening Peak',
                        'time': 19,
                        'span': 1.5,
                        'quantity': math.ceil(evs_per_substation),
                        'enabled': True,
                        'penetration_percent': scenario_data['ev_penetration'] * 100,
                    }
                ]
                
                # Update optimization strategies
                st.session_state.optimization_strategy['smart_charging_percent'] = smart_charging_percent
                st.session_state.optimization_strategy['pv_adoption_percent'] = 30
                st.session_state.optimization_strategy['battery_capacity'] = 17.5
                st.session_state.optimization_strategy['max_discharge_rate'] = 7.0
                st.session_state.optimization_strategy['discharge_start_hour'] = 18
                st.session_state.optimization_strategy['solar_energy_percent'] = 70
                st.session_state.optimization_strategy['grid_battery_adoption_percent'] = grid_battery_adoption
                st.session_state.optimization_strategy['grid_battery_capacity'] = 20.0
                st.session_state.optimization_strategy['grid_battery_max_rate'] = 5.0
                st.session_state.optimization_strategy['grid_battery_charge_start_hour'] = 7
                st.session_state.optimization_strategy['grid_battery_charge_duration'] = 8
                st.session_state.optimization_strategy['grid_battery_discharge_start_hour'] = 18
                st.session_state.optimization_strategy['grid_battery_discharge_duration'] = 4
                st.session_state.optimization_strategy['v2g_adoption_percent'] = v2g_adoption_percent
                st.session_state.optimization_strategy['v2g_discharge_duration'] = 3
                st.session_state.optimization_strategy['v2g_max_discharge_rate'] = v2g_discharge_rate
                st.session_state.optimization_strategy['v2g_start_hour'] = 18
                
                # Update session state
                st.session_state.portugal_scenario = {
                    'year': selected_year,
                    'scenario': selected_scenario,
                }
                
                st.session_state.scenario_success_message = f"✅ Scenario applied!"
                st.session_state.scenario_success_timer = 0
                st.rerun()
            
            # Display scenario details after the button
            if show_builtin_summary:
                st.write(f"**📊 Scenario Details:**")
                st.write(f"**Year:** {selected_year}")
                st.write(f"**Scenario:** {selected_scenario.title()}")
                st.write(f"**Total Cars:** {year_data['total_cars_million']:.1f}M")
                st.write(f"**Total EVs:** {year_data['total_cars_million'] * scenario_data['ev_penetration']:.1f}M")
                st.write(f"• **EV Penetration:** {scenario_data['ev_penetration']*100:.0f}%")
                st.write(f"• **Home Charging:** {year_data['home_charging_percent']*100:.0f}% ({year_data['home_charged_evs_million']:.1f}M EVs)")
                st.write(f"• **Battery Capacity:** {scenario_data['avg_battery_kWh']:.0f} kWh")
                st.write(f"• **Charging Power:** {scenario_data['charging_power_kW']:.1f} kW")
                st.write(f"• **Smart Charging:** {smart_charging_percent}%")
                st.write(f"• **Grid Battery:** {grid_battery_adoption}%")
                st.write(f"• **V2G:** {v2g_adoption_percent}%")
                st.write(f"• **AC Chargers:** {math.ceil(evs_per_substation)} chargers ({scenario_data['charging_power_kW']:.1f} kW each)")
                st.write(f"• **Total Charger Capacity:** {math.ceil(evs_per_substation) * scenario_data['charging_power_kW']:.1f} kW")
                
                
                
                # EV Configuration
                st.write(f"**🚗 EV Configuration:**")
                st.write(f"• **Battery Capacity:** {scenario_data['avg_battery_kWh']:.0f} kWh")
                st.write(f"• **Charging Power:** {scenario_data['charging_power_kW']:.1f} kW")
                st.write(f"• **Initial SOC:** 20%")
                
                # Charger Configuration
                st.write(f"**🔌 Charger Configuration:**")
                st.write(f"• **AC Chargers:** {math.ceil(evs_per_substation)} chargers ({scenario_data['charging_power_kW']:.1f} kW each)")
                st.write(f"• **Total Charger Capacity:** {math.ceil(evs_per_substation) * scenario_data['charging_power_kW']:.1f} kW")
                
                # Smart Charging
                st.write(f"**🌙 Smart Charging:**")
                st.write(f"• **Smart Charging:** {smart_charging_percent:.0f}%")
                
                # PV + Battery System
                st.write(f"**☀️ PV + Battery System:**")
                st.write(f"• **PV + Battery Adoption:** 30%")
                st.write(f"• **Battery Capacity:** 17.5 kWh")
                st.write(f"• **Discharge Rate:** 7.0 kW")
                st.write(f"• **Solar Energy Available:** 70%")
                
                # Grid-Charged Battery
                st.write(f"**🔋 Grid-Charged Battery (Normal Battery):**")
                st.write(f"• **Adoption:** {grid_battery_adoption}%")
                st.write(f"• **Capacity:** 20.0 kWh")
                st.write(f"• **Max Rate:** 5.0 kW")
                st.write(f"• **Charge:** 7:00 for 8h")
                st.write(f"• **Discharge:** 18:00 for 4h")
                
                # V2G Section
                st.write(f"**🚗 Vehicle-to-Grid (V2G):**")
                st.write(f"• **Adoption:** {v2g_adoption_percent}%")
                st.write(f"• **Discharge Rate:** {v2g_discharge_rate:.1f} kW (same as charging rate)")
                st.write(f"• **Discharge:** 18:00 for 3h")
                st.write(f"• **Recharge Arrival:** 02:00 next day (26:00)")
                
                st.write(f"**📝 Notes:** {year_data['notes']}")
        
        else:
            # Custom scenarios section
            if 'custom_scenarios' not in st.session_state:
                st.session_state.custom_scenarios = {}
            
            # Get all available custom years
            custom_years = list(st.session_state.custom_scenarios.keys())
            
            if custom_years:
                # Custom scenarios section with dropdowns in first row
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    selected_custom_year = st.selectbox(
                        "Year",
                        options=custom_years,
                        help="Select the year for custom scenarios"
                    )
                
                with col2:
                    # Get available scenarios for selected year
                    available_custom_scenarios = list(st.session_state.custom_scenarios[selected_custom_year]['scenarios'].keys())
                    
                    if available_custom_scenarios:
                        selected_custom_scenario = st.selectbox(
                            "Custom Scenario",
                            options=available_custom_scenarios,
                            help="Select your custom scenario"
                        )
                
                                # Checkbox in second row
                show_custom_summary = st.checkbox("📊 Show Summary", value=False, help="Toggle to show/hide the detailed scenario summary", key="show_custom_summary")
                
                # Get custom scenario data
                scenario_data = st.session_state.custom_scenarios[selected_custom_year]['scenarios'][selected_custom_scenario]
                year_data = st.session_state.custom_scenarios[selected_custom_year]
                total_cars_million = st.session_state.custom_scenarios[selected_custom_year]['total_cars_million']
                    
                    # Calculate EVs per substation
                total_cars = total_cars_million * 1_000_000
                substation_count = portugal_ev_scenarios['substation_count']
                evs_per_substation = (total_cars * scenario_data['ev_penetration']) / substation_count
                    
                # Apply custom scenario button
                if st.button("🚀 Apply Custom Scenario", type="primary"):
                    # Update EV configuration
                    st.session_state.dynamic_ev = {
                        'capacity': scenario_data['avg_battery_kWh'],
                        'AC': scenario_data['charging_power_kW']
                    }
                    
                    # Set default SOC
                    st.session_state.ev_soc = 0.2
                    
                    # Update charger configuration
                    st.session_state.dynamic_charger = {
                        'ac_rate': scenario_data['charging_power_kW'],
                        'ac_count': math.ceil(evs_per_substation),  # Number of chargers equals number of EVs
                    }
                    
                    # Also update charger_config to match (used by simulation)
                    st.session_state.charger_config = {
                        'ac_rate': scenario_data['charging_power_kW'],
                        'ac_count': math.ceil(evs_per_substation),  # Number of chargers equals number of EVs
                    }
                    
                    # Update time peaks configuration
                    st.session_state.time_peaks = [
                        {
                            'name': 'Evening Peak',
                            'time': 19,
                            'span': 1.5,
                            'quantity': math.ceil(evs_per_substation),
                            'enabled': True,
                            'penetration_percent': scenario_data['ev_penetration'] * 100,
                        }
                    ]
                    
                    st.success(f"✅ Custom scenario '{selected_custom_scenario}' applied!")
                    st.rerun()
                
                # Display custom scenario details
                if show_custom_summary:
                    st.write(f"**📊 Custom Scenario Details:**")
                    st.write(f"**Year:** {selected_custom_year}")
                    st.write(f"**Name:** {selected_custom_scenario}")
                    st.write(f"**Total Cars:** {total_cars_million:.1f}M")
                    st.write(f"**Total EVs:** {total_cars_million * scenario_data['ev_penetration']:.1f}M")
                    st.write(f"• **EV Penetration:** {scenario_data['ev_penetration']*100:.0f}%")
                    st.write(f"• **Home Charging:** {year_data.get('home_charging_percent', 0.8)*100:.0f}% ({total_cars_million * scenario_data['ev_penetration'] * year_data.get('home_charging_percent', 0.8):.1f}M EVs)")
                    st.write(f"• **Battery Capacity:** {scenario_data['avg_battery_kWh']:.0f} kWh")
                    st.write(f"• **Charging Power:** {scenario_data['charging_power_kW']:.1f} kW")
                    st.write(f"• **Smart Charging:** {scenario_data.get('optimization_params', {}).get('smart_charging_percent', 0)}%")
                    st.write(f"• **Grid Battery:** {scenario_data.get('optimization_params', {}).get('grid_battery_adoption_percent', 0)}%")
                    st.write(f"• **V2G:** {scenario_data.get('optimization_params', {}).get('v2g_adoption_percent', 0)}%")
                    st.write(f"• **AC Chargers:** {math.ceil(evs_per_substation)} chargers ({scenario_data['charging_power_kW']:.1f} kW each)")
                    st.write(f"• **Total Charger Capacity:** {math.ceil(evs_per_substation) * scenario_data['charging_power_kW']:.1f} kW")
                    
                    
                    
                    # EV Configuration
                    st.write(f"**🚗 EV Configuration:**")
                    st.write(f"• **Battery Capacity:** {scenario_data['avg_battery_kWh']:.0f} kWh")
                    st.write(f"• **Charging Power:** {scenario_data['charging_power_kW']:.1f} kW")
                    st.write(f"• **Initial SOC:** 20%")
                    
                    # Charger Configuration
                    st.write(f"**🔌 Charger Configuration:**")
                    st.write(f"• **AC Chargers:** {math.ceil(evs_per_substation)} chargers ({scenario_data['charging_power_kW']:.1f} kW each)")
                    st.write(f"• **Total Charger Capacity:** {math.ceil(evs_per_substation) * scenario_data['charging_power_kW']:.1f} kW")
                    
                    # Smart Charging
                    st.write(f"**🌙 Smart Charging:**")
                    st.write(f"• **Smart Charging:** {scenario_data.get('optimization_params', {}).get('smart_charging_percent', 0):.0f}%")
                    
                    # PV + Battery System
                    st.write(f"**☀️ PV + Battery System:**")
                    st.write(f"• **PV + Battery Adoption:** 30%")
                    st.write(f"• **Battery Capacity:** 17.5 kWh")
                    st.write(f"• **Discharge Rate:** 7.0 kW")
                    st.write(f"• **Solar Energy Available:** 70%")
                    
                    # Grid-Charged Battery
                    st.write(f"**🔋 Grid-Charged Battery (Normal Battery):**")
                    st.write(f"• **Adoption:** {scenario_data.get('optimization_params', {}).get('grid_battery_adoption_percent', 0)}%")
                    st.write(f"• **Capacity:** 20.0 kWh")
                    st.write(f"• **Max Rate:** 5.0 kW")
                    st.write(f"• **Charge:** 7:00 for 8h")
                    st.write(f"• **Discharge:** 18:00 for 4h")
                    
                    # V2G Section
                    st.write(f"**🚗 Vehicle-to-Grid (V2G):**")
                    st.write(f"• **Adoption:** {scenario_data.get('optimization_params', {}).get('v2g_adoption_percent', 0)}%")
                    st.write(f"• **Discharge Rate:** {scenario_data['charging_power_kW']:.1f} kW (same as charging rate)")
                    st.write(f"• **Discharge:** 18:00 for 3h")
                    st.write(f"• **Recharge Arrival:** 02:00 next day (26:00)")
                    
                    st.write(f"**📝 Notes:** {year_data.get('notes', 'No notes')}")
                else:
                    st.warning("No custom scenarios available for this year.")
            else:
                st.info("No custom scenarios created yet. Use the 'Add Scenario' button below to create your first custom scenario.")
        
        # Success message display
        if 'scenario_success_message' in st.session_state and 'scenario_success_timer' in st.session_state:
            st.success(st.session_state.scenario_success_message)
            st.session_state.scenario_success_timer += 1
            
            # Clear the message after timer expires
            if st.session_state.scenario_success_timer >= 3:
                del st.session_state.scenario_success_message
                del st.session_state.scenario_success_timer

        # Add Scenario button - opens comprehensive scenario editor
        st.divider()
        if st.button("➕ Add Scenario", type="secondary"):
            st.session_state.show_add_scenario = True
        
        if st.session_state.get('show_add_scenario', False):
            st.write("**🔧 Add New Scenario:**")
            st.caption("This will capture all current settings and create a new custom scenario.")
            
            # Scenario inputs in a row
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                new_scenario_name = st.text_input(
                    "Scenario Name",
                    value="my_scenario",
                    help="Name for your new scenario"
                )
            
            with col2:
                new_scenario_year = st.number_input(
                    "Year",
                    min_value=2020,
                    max_value=2050,
                    value=2030,
                    help="Target year for the scenario"
                )
            
            
            show_summary = st.checkbox("📊 Show Summary", value=False, help="Toggle to show/hide the detailed scenario summary", key="show_add_scenario_summary")
            
            new_scenario_notes = st.text_area(
                "Scenario Notes",
                value="Custom scenario created from current settings",
                help="Brief description or notes about this scenario"
            )
            
            # Display current settings that will be captured
            st.write("**📋 Current Settings to be Captured:**")
            
            if show_summary:
                # Get current settings (needed for summary display)
                current_ev = st.session_state.get('dynamic_ev', {})
                current_charger = st.session_state.get('dynamic_charger', {})
                current_strategies = st.session_state.get('active_strategies', [])
                current_optimization = st.session_state.get('optimization_strategy', {})
                
                # Calculate EVs per substation for the scenario
                current_peaks = st.session_state.get('time_peaks', [])
                total_evs = sum(peak.get('quantity', 0) for peak in current_peaks)
                substation_count = portugal_ev_scenarios['substation_count']
                ev_penetration = total_evs / substation_count if substation_count > 0 else 0
                
                # Calculate EVs per substation
                evs_per_substation = ev_penetration
                
                # Comprehensive current settings summary (identical to Portugal scenario format)
                st.write("**📈 General Data:**")
                st.write(f"• **Total Cars:** 6.0M")
                st.write(f"• **Total EVs:** {total_evs}")
                st.write(f"• **EV Penetration:** {ev_penetration*100:.1f}%")
                st.write(f"• **Home Charging:** 80% ({total_evs * 0.8:.1f} EVs)")
                st.write(f"• **EVs per Substation:** {evs_per_substation:.2f}")
                
                # EV Configuration
                st.write("**🚗 EV Configuration:**")
                st.write(f"• **Battery Capacity:** {current_ev.get('capacity', 75)} kWh")
                st.write(f"• **Charging Power:** {current_ev.get('AC', 11)} kW")
                st.write(f"• **Initial SOC:** 20%")
                
                # Charger Configuration
                st.write("**🔌 Charger Configuration:**")
                st.write(f"• **AC Chargers:** {math.ceil(evs_per_substation)} chargers ({current_ev.get('AC', 11)} kW each)")
                st.write(f"• **Total Charger Capacity:** {math.ceil(evs_per_substation) * current_ev.get('AC', 11)} kW")
                
                # Smart Charging
                st.write("**🌙 Smart Charging:**")
                smart_charging_percent = current_optimization.get('smart_charging_percent', 0)
                st.write(f"• **Smart Charging:** {smart_charging_percent:.0f}%")
                
                # PV + Battery System
                st.write("**☀️ PV + Battery System:**")
                st.write(f"• **PV + Battery Adoption:** 30%")
                st.write(f"• **Battery Capacity:** 17.5 kWh")
                st.write(f"• **Discharge Rate:** 7.0 kW")
                st.write(f"• **Solar Energy Available:** 70%")
                
                # Grid-Charged Battery
                st.write("**🔋 Grid-Charged Battery (Normal Battery):**")
                grid_battery_adoption = current_optimization.get('grid_battery_adoption_percent', 0)
                st.write(f"• **Adoption:** {grid_battery_adoption}%")
                st.write(f"• **Capacity:** 20.0 kWh")
                st.write(f"• **Max Rate:** 5.0 kW")
                st.write(f"• **Charge:** 7:00 for 8h")
                st.write(f"• **Discharge:** 18:00 for 4h")
                
                # V2G Section
                st.write("**🚗 Vehicle-to-Grid (V2G):**")
                v2g_adoption_percent = current_optimization.get('v2g_adoption_percent', 0)
                v2g_discharge_rate = current_optimization.get('v2g_max_discharge_rate', current_ev.get('AC', 11))
                st.write(f"• **Adoption:** {v2g_adoption_percent}%")
                st.write(f"• **Discharge Rate:** {v2g_discharge_rate:.1f} kW (same as charging rate)")
                st.write(f"• **Discharge:** 18:00 for 3h")
                st.write(f"• **Recharge Arrival:** 02:00 next day (26:00)")
                
                st.write("**📝 Notes:** Custom scenario created from current settings")
            
            # Save and Cancel buttons
            col_save1, col_save2 = st.columns(2)
            
            with col_save1:
                if st.button("💾 Save Scenario", type="primary"):
                    # Get current settings (needed for save functionality)
                    current_ev = st.session_state.get('dynamic_ev', {})
                    current_charger = st.session_state.get('dynamic_charger', {})
                    current_strategies = st.session_state.get('active_strategies', [])
                    current_optimization = st.session_state.get('optimization_strategy', {})
                    
                    # Initialize custom scenarios in session state if not exists
                    if 'custom_scenarios' not in st.session_state:
                        st.session_state.custom_scenarios = {}
                    
                    # Create year entry if not exists
                    if str(new_scenario_year) not in st.session_state.custom_scenarios:
                        st.session_state.custom_scenarios[str(new_scenario_year)] = {
                            'total_cars_million': 6.0,  # Default value
                            'total_evs_million': 1.5,   # Default value
                            'home_charging_percent': 0.8,  # Default value
                            'home_charged_evs_million': 1.2,  # Default value
                            'notes': new_scenario_notes,
                            'scenarios': {}
                        }
                    
                    # Calculate EVs per substation for the scenario
                    current_peaks = st.session_state.get('time_peaks', [])
                    total_evs = sum(peak.get('quantity', 0) for peak in current_peaks)
                    substation_count = portugal_ev_scenarios['substation_count']
                    ev_penetration = total_evs / substation_count if substation_count > 0 else 0
                    
                    # Save the scenario with current settings
                    st.session_state.custom_scenarios[str(new_scenario_year)]['scenarios'][new_scenario_name] = {
                        'ev_penetration': ev_penetration,
                        'avg_battery_kWh': current_ev.get('capacity', 75),
                        'charging_power_kW': current_ev.get('AC', 11),
                        'initial_soc': st.session_state.get('ev_soc', 0.2),
                        'ac_rate': current_charger.get('ac_rate', 11),
                        'active_strategies': current_strategies.copy(),
                        'optimization_params': current_optimization.copy(),
                        'time_peaks': current_peaks.copy(),
                        'notes': new_scenario_notes
                    }
                    
                    st.success(f"✅ Scenario '{new_scenario_name}' for year {new_scenario_year} saved with all current settings!")
                    st.session_state.show_add_scenario = False
                    st.rerun()
            
            with col_save2:
                if st.button("❌ Cancel", type="secondary"):
                    st.session_state.show_add_scenario = False
                    st.rerun()

    # Optimization Strategies
    with st.expander("⚡ Optimization Strategies", expanded=False):
        st.write("**Select charging optimization strategies to reduce grid stress:**")
        
        # Initialize optimization values in session state if not exists
        if 'optimization_strategy' not in st.session_state:
            st.session_state.optimization_strategy = {
                'smart_charging_percent': 0,
                'shift_start_time': 23,
                'pv_adoption_percent': 30,
                'pv_power': 7,
                'battery_capacity': 17.5,
                'max_discharge_rate': 7.0,
                'discharge_start_hour': 18,  # Default 6pm
                'solar_energy_percent': 70,
                'grid_battery_adoption_percent': 10,
                'grid_battery_capacity': 20.0,
                'grid_battery_max_rate': 5.0,
                'grid_battery_charge_start_hour': 7,
                'grid_battery_charge_duration': 8,
                'grid_battery_discharge_start_hour': 18,  # Default 6pm
                'grid_battery_discharge_duration': 4
            }
        
        # Strategy selection with multiple options
        st.write("**Select optimization strategies (multiple can be selected):**")
        
        # Initialize active strategies if not exists
        if 'active_strategies' not in st.session_state:
            st.session_state.active_strategies = []
        
        # Time of Use checkbox
        smart_charging_enabled = st.checkbox(
            "⚡ Time of Use",
            value='smart_charging' in st.session_state.active_strategies,
            help="Configure different charging periods with varying adoption rates based on time-of-use tariffs"
        )
        
        # PV + Battery checkbox
        pv_battery_enabled = st.checkbox(
            "☀️ PV + Battery System",
            value='pv_battery' in st.session_state.active_strategies,
            help="Solar panels and batteries support the grid"
        )
        
        # Grid-Charged Batteries checkbox
        grid_battery_enabled = st.checkbox(
            "🔋 Grid-Charged Batteries",
            value='grid_battery' in st.session_state.active_strategies,
            help="Batteries charged from grid during day, discharged during peak"
        )
        
        v2g_enabled = st.checkbox(
            "🚗 Vehicle-to-Grid (V2G)",
            value='v2g' in st.session_state.active_strategies,
            help="EVs discharge to grid during peak, recharge later"
        )
        
        # Update active strategies based on selections
        active_strategies = []
        if smart_charging_enabled:
            active_strategies.append('smart_charging')
        if pv_battery_enabled:
            active_strategies.append('pv_battery')
        if grid_battery_enabled:
            active_strategies.append('grid_battery')
        if v2g_enabled:
            active_strategies.append('v2g')
        
        st.session_state.active_strategies = active_strategies
        
        # Time of Use configuration - back in sidebar
        if 'smart_charging' in active_strategies:
            st.write("**⚡ Time of Use Configuration:**")
            st.write("Configure different charging periods with varying adoption rates based on time-of-use tariffs.")
            
            # Add toggles for number of TOU periods
            if 'tou_period_count' not in st.session_state:
                st.session_state.tou_period_count = 4
            
            st.write("**Number of TOU Periods:**")
            
            # Use radio buttons for mutually exclusive selection
            num_periods = st.radio(
                "Select number of periods:",
                options=[2, 3, 4, 5],
                index=[2, 3, 4, 5].index(st.session_state.tou_period_count) if st.session_state.tou_period_count in [2, 3, 4, 5] else 2,
                horizontal=True,
                key="tou_period_selector",
                help="Choose the number of TOU periods for optimization"
            )
            
            # Update session state if changed
            if num_periods != st.session_state.tou_period_count:
                print(f"🔄 TOU Period Count Changed: {st.session_state.tou_period_count} -> {num_periods}")
                st.session_state.tou_period_count = num_periods
                # Reset timeline to force regeneration with new period count
                if 'time_of_use_timeline' in st.session_state:
                    del st.session_state.time_of_use_timeline
                st.rerun()
            
            # Function to merge periods based on selected count
            def merge_periods(periods, target_count):
                if target_count == 5:
                    # For 5 periods, add a new Period 5 with equal distribution
                    # Keep the original 4 periods but add a new one
                    new_periods = periods.copy()
                    # Add Period 5 with some default hours (e.g., late night)
                    new_periods.append({
                        'name': 'Period 5',
                        'color': '#9370DB',  # Purple color for Period 5
                        'adoption': 100.0 / 5,
                        'hours': [23, 24]  # Late night hours
                    })
                    # Apply equal adoption percentages for all 5 periods
                    equal_adoption = 100.0 / 5
                    for period in new_periods:
                        period['adoption'] = equal_adoption
                    return new_periods
                elif target_count == 4:
                    # Apply equal adoption percentages for 4 periods
                    equal_adoption = 100.0 / 4
                    for period in periods:
                        period['adoption'] = equal_adoption
                    return periods
                elif target_count == 3:
                    # Preserve first (left) two periods and merge last (right) two periods
                    merged_periods = periods[:2]  # Keep Period 1 and Period 2
                    merged_hours = periods[2]['hours'] + periods[3]['hours']
                    merged_periods.append({
                        'name': 'Period 3',
                        'color': '#FFD700',  # Keep Period 3 color
                        'adoption': 100.0 / 3,  # Equal adoption for 3 periods
                        'hours': merged_hours
                    })
                    
                    # Apply equal adoption percentages
                    equal_adoption = 100.0 / 3
                    for period in merged_periods:
                        period['adoption'] = equal_adoption
                    
                    return merged_periods
                elif target_count == 2:
                    # Merge first two (left) and last two (right)
                    merged_periods = []
                    
                    # Merge Period 1 and Period 2 (first two)
                    merged_hours_1_2 = periods[0]['hours'] + periods[1]['hours']
                    merged_periods.append({
                        'name': 'Period 1',
                        'color': '#87CEEB',  # Keep Period 1 color
                        'adoption': 100.0 / 2,  # Equal adoption for 2 periods
                        'hours': merged_hours_1_2
                    })
                    
                    # Merge Period 3 and Period 4 (last two)
                    merged_hours_3_4 = periods[2]['hours'] + periods[3]['hours']
                    merged_periods.append({
                        'name': 'Period 2',
                        'color': '#90EE90',  # Use Period 2 color (not Period 3)
                        'adoption': 100.0 / 2,  # Equal adoption for 2 periods
                        'hours': merged_hours_3_4
                    })
                    
                    return merged_periods
                return periods
            
            # Initialize timeline if not in session state
            if 'time_of_use_timeline' not in st.session_state:
                base_periods = [
                    {'name': 'Period 1', 'color': '#87CEEB', 'adoption': 25, 'hours': list(range(2, 6))},
                    {'name': 'Period 2', 'color': '#90EE90', 'adoption': 25, 'hours': list(range(1, 2)) + list(range(6, 8)) + list(range(22, 25))},
                    {'name': 'Period 3', 'color': '#FFD700', 'adoption': 25, 'hours': list(range(8, 9)) + list(range(11, 18)) + list(range(21, 22))},
                    {'name': 'Period 4', 'color': '#FF6B6B', 'adoption': 25, 'hours': list(range(9, 11)) + list(range(18, 21))}
                ]
                
                merged_periods = merge_periods(base_periods, num_periods)
                
                # Apply equal adoption percentages
                equal_adoption = 100.0 / len(merged_periods)
                for period in merged_periods:
                    period['adoption'] = equal_adoption
                
                st.session_state.time_of_use_timeline = {
                    'periods': merged_periods,
                    'selected_period': 0
                }
            
            # Store original timeline if not already stored
            if 'original_timeline' not in st.session_state:
                base_periods = [
                    {'name': 'Period 1', 'color': '#87CEEB', 'adoption': 25, 'hours': list(range(2, 6))},
                    {'name': 'Period 2', 'color': '#90EE90', 'adoption': 25, 'hours': list(range(1, 2)) + list(range(6, 8)) + list(range(22, 25))},
                    {'name': 'Period 3', 'color': '#FFD700', 'adoption': 25, 'hours': list(range(8, 9)) + list(range(11, 18)) + list(range(21, 22))},
                    {'name': 'Period 4', 'color': '#FF6B6B', 'adoption': 25, 'hours': list(range(9, 11)) + list(range(18, 21))}
                ]
                
                st.session_state.original_timeline = {
                    'periods': base_periods
                }
            
            # Update timeline if period count changed
            if len(st.session_state.time_of_use_timeline['periods']) != num_periods:
                base_periods = st.session_state.original_timeline['periods']
                merged_periods = merge_periods(base_periods, num_periods)
                
                # Apply equal adoption percentages
                equal_adoption = 100.0 / len(merged_periods)
                for period in merged_periods:
                    period['adoption'] = equal_adoption
                
                st.session_state.time_of_use_timeline['periods'] = merged_periods
            
            timeline = st.session_state.time_of_use_timeline
            
            # Period adoption percentages at the top
            st.write("**📊 Period Adoption Percentages:**")
            st.write("Specify what percentage of users choose each period:")
            
            # Apply optimized TOU values if available from Bayesian optimization
            if 'optimized_tou_values' in st.session_state:
                optimized_values = st.session_state.optimized_tou_values
                
                # Create a flexible mapping system that works with any number of periods
                # Only map periods that actually exist to avoid conflicts
                period_mapping = {}
                
                # Map based on the actual periods present using period_one, period_two, etc.
                for period in timeline['periods']:
                    period_name = period['name']
                    if period_name == 'Period 1':
                        period_mapping[period_name] = 'period_one'
                    elif period_name == 'Period 2':
                        period_mapping[period_name] = 'period_two'
                    elif period_name == 'Period 3':
                        period_mapping[period_name] = 'period_three'
                    elif period_name == 'Period 4':
                        period_mapping[period_name] = 'period_four'
                    elif period_name == 'Period 5':
                        period_mapping[period_name] = 'period_five'
                
                # Apply optimized values to timeline periods using the mapping
                for period in timeline['periods']:
                    period_name = period['name']
                    if period_name in period_mapping:
                        param_name = period_mapping[period_name]
                        if param_name in optimized_values:
                            period['adoption'] = optimized_values[param_name]
                            print(f"🔍 Applied {param_name}: {optimized_values[param_name]:.2f}% to {period_name}")
                        else:
                            print(f"🔍 Warning: {param_name} not found in optimized_values")
                    else:
                        print(f"🔍 Warning: Unknown period name '{period_name}'")
                
                # Debug output to verify applied values
                print(f"🔍 Main Simulation Debug - {len(timeline['periods'])} periods:")
                for period in timeline['periods']:
                    print(f"  {period['name']}: {period['adoption']:.2f}%")
                
                # Also update the simulation periods in optimization_strategy
                if 'optimization_strategy' not in st.session_state:
                    st.session_state.optimization_strategy = {}
                
                # Convert timeline periods to simulation format and store in optimization_strategy
                simulation_periods = convert_to_simulation_format(timeline['periods'])
                st.session_state.optimization_strategy['time_of_use_periods'] = simulation_periods
                
                # Debug: Show original timeline periods
                st.info(f"🔍 Original Timeline Periods: {[(p['name'], len(p['hours']), p['hours'][:5] if p['hours'] else []) for p in timeline['periods']]}")
                
                # Clear the optimized values after applying them
                del st.session_state.optimized_tou_values
                
                # Rerun to refresh the UI with updated values
                st.rerun()
            
            # Create columns for each period with color blocks
            period_cols = st.columns(len(timeline['periods']))
            for i, period in enumerate(timeline['periods']):
                with period_cols[i]:
                    # Color block above period name with consistent alignment
                    st.markdown(f"""
                    <div style="
                        background-color: {period['color']};
                        width: 100%;
                        height: 30px;
                        border-radius: 5px;
                        margin: 0;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        border: 1px solid #ccc;
                    ">
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Consistent spacing and alignment with fixed height
                    st.markdown(f"""
                    <div style="
                        margin: 8px 0 4px 0;
                        height: 40px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">
                        <strong>{period['name']}:</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    period['adoption'] = st.number_input(
                        "Adoption %", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=float(period['adoption']), 
                        step=1.0, 
                        key=f"adoption_{i}"
                    )
            
            # Show total adoption info
            total_adoption = sum(p['adoption'] for p in timeline['periods'])
            if total_adoption > 100.01:
                st.warning(f"⚠️ Total adoption is {total_adoption:.2f}% (exceeds 100%)")
            elif total_adoption < 100:
                st.info(f"ℹ️ Total adoption is {total_adoption}% ({100 - total_adoption}% of users not using Time of Use)")
            else:
                st.success(f"✅ Total adoption is {total_adoption}%")
            
            # Ensure simulation periods are always updated in optimization_strategy
            if 'optimization_strategy' not in st.session_state:
                st.session_state.optimization_strategy = {}
            
            # Convert timeline periods to simulation format and store in optimization_strategy
            simulation_periods = convert_to_simulation_format(timeline['periods'])
            st.session_state.optimization_strategy['time_of_use_periods'] = simulation_periods
            
            # Add period legend in 2 columns (compressed)
            # legend_col1, legend_col2 = st.columns(2)
            # with legend_col1:
            #     st.markdown("**Period Legend:**")
            #     st.markdown("- **1** = Period 1")
            #     st.markdown("- **2** = Period 2")
            # with legend_col2:
            #     st.markdown("&nbsp;")  # Empty space for alignment
            #     st.markdown("- **3** = Period 3")
            #     st.markdown("- **4** = Period 4")
            
            # Add custom CSS to reduce vertical spacing
            st.markdown("""
            <style>
            /* Reduce spacing in legend */
            .stMarkdown {
                margin-bottom: 0.5rem !important;
            }
            .stMarkdown p {
                margin-bottom: 0.2rem !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create period options for dropdown with single letter names
            period_options = []
            period_name_mapping = {}
            for period in timeline['periods']:
                if period['name'] == "Period 1":
                    short_name = "P1"
                elif period['name'] == "Period 2":
                    short_name = "P2"
                elif period['name'] == "Period 3":
                    short_name = "P3"
                elif period['name'] == "Period 4":
                    short_name = "P4"
                else:
                    short_name = period['name'][0]
                
                period_options.append(short_name)
                period_name_mapping[short_name] = period['name']

   # 24-Hour Timeline with 4-column grid layout
            st.write("**24-Hour Timeline:**")
          

            # Create 4 rows with 6 columns each (inverted layout)
            for row in range(4):
                cols = st.columns(6)
                
                # Row 1: Hours 1-6
                # Row 2: Hours 7-12
                # Row 3: Hours 13-18
                # Row 4: Hours 19-24
                
                for col in range(6):
                    hour = row * 6 + col + 1
                    
                    with cols[col]:
                        # Find which period this hour belongs to
                        period_idx = None
                        for j, period in enumerate(timeline['periods']):
                            if hour in period['hours']:
                                period_idx = j
                                break
                        
                        # Determine current selection and color
                        if period_idx is not None:
                            current_selection = timeline['periods'][period_idx]['name']
                            bg_color = timeline['periods'][period_idx]['color']
                        else:
                            # Default to first period if not assigned
                            current_selection = timeline['periods'][0]['name']
                            bg_color = timeline['periods'][0]['color']
                            # Add to default period
                            timeline['periods'][0]['hours'].append(hour)
                        
                        # Initialize session state for hour assignments if not exists (scenario-independent)
                        if 'hour_assignments' not in st.session_state:
                            st.session_state.hour_assignments = {}
                        
                        # Get the current period assignment for this hour (from session state or timeline)
                        # Use timeline default if no session state assignment exists
                        current_period_name = st.session_state.hour_assignments.get(hour, current_selection)
                        
                        # Find the period index for the current assignment
                        current_period_idx = None
                        for j, period in enumerate(timeline['periods']):
                            if period['name'] == current_period_name:
                                current_period_idx = j
                                break
                        
                        if current_period_idx is not None:
                            current_bg_color = timeline['periods'][current_period_idx]['color']
                        else:
                            # Fallback to first period if assignment not found
                            current_bg_color = timeline['periods'][0]['color']
                            current_period_name = timeline['periods'][0]['name']
                        
                        # Create colored hour block FIRST (above dropdown)
                        st.markdown(f"""
                        <div style="
                            background-color: {current_bg_color};
                            border: 1px solid #ccc;
                            border-radius: 5px;
                            padding: 8px;
                            margin: 4px 0;
                            text-align: center;
                            color: {'black' if _is_light_color(current_bg_color) else 'white'};
                            font-weight: bold;
                            font-size: 14px;
                        ">
                            {hour:02.0f}h
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create dropdown for period selection BELOW the block
                        dropdown_key = f"hour_{hour}_dropdown"
                        
                        # Convert current selection to short name for dropdown
                        current_short_name = None
                        for short_name, full_name in period_name_mapping.items():
                            if full_name == current_period_name:
                                current_short_name = short_name
                                break
                        
                        # Add custom CSS to hide dropdown arrow and prevent text truncation
                        st.markdown(f"""
                        <style>
                        /* Remove dropdown arrow from all selectboxes */
                        .stSelectbox > div > div > div > div {{
                            background-image: none !important;
                        }}
                        .stSelectbox > div > div > div > div::after {{
                            display: none !important;
                        }}
                        .stSelectbox > div > div > div > div::before {{
                            display: none !important;
                        }}
                        /* Alternative selectors */
                        div[data-testid="stSelectbox"] select {{
                            -webkit-appearance: none !important;
                            -moz-appearance: none !important;
                            appearance: none !important;
                            background-image: none !important;
                            background: none !important;
                        }}
                        /* Hide any SVG or icon elements */
                        .stSelectbox svg {{
                            display: none !important;
                        }}
                        .stSelectbox path {{
                            display: none !important;
                        }}
                        /* Allow text to overflow and be cut off in dropdowns */
                        .stSelectbox select {{
                            text-overflow: clip !important;
                            white-space: nowrap !important;
                            overflow: hidden !important;
                        }}
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Use dropdown with custom CSS to hide arrow
                        selected_short_period = st.selectbox(
                            "Period",
                            options=period_options,
                            index=period_options.index(current_short_name) if current_short_name else 0,
                            key=dropdown_key,
                            label_visibility="collapsed"
                        )
                        
                        # Convert back to full period name
                        selected_period = period_name_mapping[selected_short_period]
                        
                        # Update session state and period assignments immediately
                        if selected_period != current_period_name:
                            # Update session state
                            st.session_state.hour_assignments[hour] = selected_period
                            
                            # Remove from current period
                            if period_idx is not None:
                                timeline['periods'][period_idx]['hours'].remove(hour)
                            
                            # Add to new period
                            for j, period in enumerate(timeline['periods']):
                                if period['name'] == selected_period:
                                    period['hours'].append(hour)
                                    break
                            
                            # Force immediate rerun to update the color block
                            st.rerun()
            
            # Add reset button under the timeline
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Reset TOU Timeline", type="secondary"):
                    # Clear session state assignments to restore original timeline defaults
                    st.session_state.pop('hour_assignments', None)
                    st.session_state.pop('initial_timeline', None)
                    
                    # Clear optimized TOU values from dynamic capacity optimizer
                    st.session_state.pop('optimized_tou_values', None)
                    st.session_state.pop('dynamic_optimization_completed', None)
                    st.session_state.pop('dynamic_optimization_results', None)
                    st.session_state.pop('dynamic_optimization_car_count', None)
                    
                    # Restore original timeline
                    if 'original_timeline' in st.session_state:
                        st.session_state.time_of_use_timeline = st.session_state.original_timeline.copy()
                    
                    st.success("✅ Timeline reset to default values!")
                    st.rerun()
            
            with col2:
                if st.button("🎯 Find Optimal TOU", type="secondary"):
                    # Define the function inline to avoid NameError
                    def extract_margin_curve_with_battery_effects():
                        try:
                            # Set random seed for consistent battery effects between TOU optimizer and main simulation
                            np.random.seed(st.session_state.random_seed)
                            
                            # Get current configuration
                            available_load_fraction = st.session_state.get('available_load_fraction', 0.8)
                            active_strategies = st.session_state.get('active_strategies', [])
                            
                            # Get power values
                            if 'synthetic_load_curve' in st.session_state and st.session_state.synthetic_load_curve is not None:
                                power_values = st.session_state.synthetic_load_curve
                            elif 'selected_dataset' in st.session_state and st.session_state.selected_dataset:
                                try:
                                    df = pd.read_csv(f"datasets/{st.session_state.selected_dataset}")
                                    selected_date = st.session_state.get('selected_date', pd.to_datetime('2023-01-18').date())
                                    df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                                    day_data = df[df['date'].dt.date == selected_date]
                                    
                                    if not day_data.empty:
                                        power_values = day_data.iloc[:, 2].values
                                    else:
                                        return None
                                except Exception as e:
                                    st.error(f"Error reading dataset: {e}")
                                    return None
                            else:
                                return None
                            
                            # Get original number of cars for battery effects (existing infrastructure)
                            if 'time_peaks' in st.session_state and st.session_state.time_peaks:
                                original_total_evs = sum(peak.get('quantity', 0) for peak in st.session_state.time_peaks)
                            elif 'ev_calculator' in st.session_state and 'total_evs' in st.session_state.ev_calculator:
                                original_total_evs = st.session_state.ev_calculator['total_evs']
                            else:
                                # Use a reasonable default if no peaks or calculator available
                                original_total_evs = 32  # Default fallback
                            
                            # Upsample 15-minute data to 1-minute intervals (EXACTLY like main simulation)
                            grid_profile_full = np.repeat(power_values, 15).astype(float)
                            
                            # Apply battery effects to grid profile (EXACTLY like main simulation)
                            adjusted_grid_profile = grid_profile_full.copy()
                            
                            # Apply PV + Battery optimization if enabled (charging during day, discharging during evening)
                            if 'pv_battery' in active_strategies:
                                pv_adoption_percent = st.session_state.optimization_strategy.get('pv_adoption_percent', 0)
                                battery_capacity = st.session_state.optimization_strategy.get('battery_capacity', 0)
                                max_charge_rate = st.session_state.optimization_strategy.get('max_charge_rate', 0)
                                max_discharge_rate = st.session_state.optimization_strategy.get('max_discharge_rate', 0)
                                solar_energy_percent = st.session_state.optimization_strategy.get('solar_energy_percent', 70)
                                pv_start_hour = st.session_state.optimization_strategy.get('pv_start_hour', 8)
                                pv_duration = st.session_state.optimization_strategy.get('pv_duration', 8)
                                charge_time = st.session_state.optimization_strategy.get('charge_time', battery_capacity / max_charge_rate)
                                system_support_time = st.session_state.optimization_strategy.get('system_support_time', pv_duration - charge_time)
                                discharge_start_hour = st.session_state.optimization_strategy.get('discharge_start_hour', 18)
                                discharge_duration = st.session_state.optimization_strategy.get('discharge_duration', battery_capacity / max_discharge_rate)
                                actual_discharge_rate = st.session_state.optimization_strategy.get('actual_discharge_rate', max_discharge_rate)
                                
                                if pv_adoption_percent > 0 and battery_capacity > 0 and max_charge_rate > 0 and actual_discharge_rate > 0:
                                    # Use the ORIGINAL number of EVs for battery effects (existing infrastructure)
                                    total_evs_for_pv = original_total_evs
                                    pv_evs = int(total_evs_for_pv * pv_adoption_percent / 100)
                                    total_charge_power = pv_evs * max_charge_rate
                                    total_system_support_power = pv_evs * max_charge_rate  # Same as charge rate for system support
                                    total_discharge_power = pv_evs * actual_discharge_rate * (solar_energy_percent / 100)
                                    
                                    # Generate variable start times with normal distribution or strict boundaries
                                    pv_use_normal_distribution = st.session_state.optimization_strategy.get('pv_use_normal_distribution', True)
                                    pv_sigma_divisor = st.session_state.optimization_strategy.get('pv_sigma_divisor', 8)
                                    
                                    if pv_use_normal_distribution and pv_sigma_divisor:
                                        pv_sigma = pv_duration / pv_sigma_divisor
                                        pv_start_times = np.random.normal(pv_start_hour, pv_sigma, pv_evs)
                                        pv_start_times = np.clip(pv_start_times, 0, 24)  # Clip to valid hours
                                    else:
                                        # Use strict boundaries - all EVs start at the same time
                                        pv_start_times = np.full(pv_evs, pv_start_hour)
                                    
                                    if pv_use_normal_distribution and pv_sigma_divisor:
                                        discharge_sigma = discharge_duration / pv_sigma_divisor
                                        discharge_start_times = np.random.normal(discharge_start_hour, discharge_sigma, pv_evs)
                                        discharge_start_times = np.clip(discharge_start_times, 0, 24)  # Clip to valid hours
                                    else:
                                        # Use strict boundaries - all EVs start at the same time
                                        discharge_start_times = np.full(pv_evs, discharge_start_hour)
                                    
                                    # Calculate time periods for each EV
                                    for ev_idx in range(pv_evs):
                                        # Get individual start times
                                        ev_pv_start = pv_start_times[ev_idx]
                                        ev_discharge_start = discharge_start_times[ev_idx]
                                        
                                        # Convert to minutes
                                        ev_pv_start_minute = int(ev_pv_start * 60)
                                        ev_system_support_end_minute = ev_pv_start_minute + int(pv_duration * 60)
                                        ev_discharge_start_minute = int(ev_discharge_start * 60)
                                        ev_discharge_duration_minutes = int(discharge_duration * 60)
                                        ev_discharge_end_minute = ev_discharge_start_minute + ev_discharge_duration_minutes
                                        
                                        # Calculate required charging rate for this EV
                                        ev_battery_capacity = battery_capacity
                                        ev_charging_time_hours = pv_duration
                                        ev_required_charging_rate = ev_battery_capacity / ev_charging_time_hours
                                        
                                        for minute in range(len(grid_profile_full)):
                                            # Day Phase: Simultaneous battery charging and grid support for this EV
                                            if ev_system_support_end_minute > 24 * 60:
                                                # System support period extends beyond 24 hours, use absolute time
                                                if (minute >= ev_pv_start_minute and minute < ev_system_support_end_minute):
                                                    # Calculate total PV power available for this EV
                                                    ev_total_pv_power = total_system_support_power / pv_evs  # Divide by number of EVs
                                                    
                                                    # Calculate remaining power for grid support
                                                    ev_grid_support_power = max(0, ev_total_pv_power - ev_required_charging_rate)
                                                    
                                                    # Apply grid support (increases available capacity)
                                                    if ev_grid_support_power > 0:
                                                        adjusted_grid_profile[minute] += ev_grid_support_power  # Increase available capacity
                                            else:
                                                # System support period is within same day, use modulo logic
                                                time_of_day = minute % (24 * 60)
                                                if (time_of_day >= ev_pv_start_minute and time_of_day < ev_system_support_end_minute):
                                                    # Calculate total PV power available for this EV
                                                    ev_total_pv_power = total_system_support_power / pv_evs  # Divide by number of EVs
                                                    
                                                    # Calculate remaining power for grid support
                                                    ev_grid_support_power = max(0, ev_total_pv_power - ev_required_charging_rate)
                                                    
                                                    # Apply grid support (increases available capacity)
                                                    if ev_grid_support_power > 0:
                                                        adjusted_grid_profile[minute] += ev_grid_support_power  # Increase available capacity
                                                
                                                # Evening Phase: Battery discharge for this EV
                                                if ev_discharge_end_minute > 24 * 60:
                                                    # Discharging period extends beyond 24 hours, use absolute time
                                                    if (minute >= ev_discharge_start_minute and minute < ev_discharge_end_minute):
                                                        ev_discharge_power = total_discharge_power / pv_evs  # Divide by number of EVs
                                                        adjusted_grid_profile[minute] += ev_discharge_power  # Increase available capacity
                                                else:
                                                    # Discharging period is within same day, use modulo logic
                                                    time_of_day = minute % (24 * 60)
                                                    if (time_of_day >= ev_discharge_start_minute and time_of_day < ev_discharge_end_minute):
                                                        ev_discharge_power = total_discharge_power / pv_evs  # Divide by number of EVs
                                                        adjusted_grid_profile[minute] += ev_discharge_power  # Increase available capacity
                            
                            # Apply Grid-Charged Batteries optimization if enabled
                            if 'grid_battery' in active_strategies:
                                grid_battery_adoption_percent = st.session_state.optimization_strategy.get('grid_battery_adoption_percent', 0)
                                grid_battery_capacity = st.session_state.optimization_strategy.get('grid_battery_capacity', 0)
                                grid_battery_max_rate = st.session_state.optimization_strategy.get('grid_battery_max_rate', 0)
                                grid_battery_charge_start_hour = st.session_state.optimization_strategy.get('grid_battery_charge_start_hour', 7)
                                grid_battery_charge_duration = st.session_state.optimization_strategy.get('grid_battery_charge_duration', 8)
                                grid_battery_discharge_start_hour = st.session_state.optimization_strategy.get('grid_battery_discharge_start_hour', 18)
                                grid_battery_discharge_duration = st.session_state.optimization_strategy.get('grid_battery_discharge_duration', 4)
                                
                                if grid_battery_adoption_percent > 0 and grid_battery_capacity > 0 and grid_battery_max_rate > 0:
                                    # Use the ORIGINAL number of EVs for battery effects (existing infrastructure)
                                    total_evs_for_grid_battery = original_total_evs
                                    grid_battery_evs = int(total_evs_for_grid_battery * grid_battery_adoption_percent / 100)
                                    
                                    # Generate variable start times
                                    grid_battery_use_normal_distribution = st.session_state.optimization_strategy.get('grid_battery_use_normal_distribution', True)
                                    grid_battery_sigma_divisor = st.session_state.optimization_strategy.get('grid_battery_sigma_divisor', 8)
                                    
                                    if grid_battery_use_normal_distribution and grid_battery_sigma_divisor:
                                        charge_sigma = grid_battery_charge_duration / grid_battery_sigma_divisor
                                        discharge_sigma = grid_battery_discharge_duration / grid_battery_sigma_divisor
                                        charge_start_times = np.random.normal(grid_battery_charge_start_hour, charge_sigma, grid_battery_evs)
                                        discharge_start_times = np.random.normal(grid_battery_discharge_start_hour, discharge_sigma, grid_battery_evs)
                                        charge_start_times = np.clip(charge_start_times, 0, 24)
                                        discharge_start_times = np.clip(discharge_start_times, 0, 24)
                                    else:
                                        charge_start_times = np.full(grid_battery_evs, grid_battery_charge_start_hour)
                                        discharge_start_times = np.full(grid_battery_evs, grid_battery_discharge_start_hour)
                                    
                                    # Calculate time periods for each EV
                                    for ev_idx in range(grid_battery_evs):
                                        ev_charge_start = charge_start_times[ev_idx]
                                        ev_discharge_start = discharge_start_times[ev_idx]
                                        
                                        # Convert to minutes
                                        ev_charge_start_minute = int(ev_charge_start * 60)
                                        ev_charge_end_minute = ev_charge_start_minute + int(grid_battery_charge_duration * 60)
                                        ev_discharge_start_minute = int(ev_discharge_start * 60)
                                        ev_discharge_end_minute = ev_discharge_start_minute + int(grid_battery_discharge_duration * 60)
                                        
                                        for minute in range(len(grid_profile_full)):
                                            time_of_day = minute % (24 * 60)
                                            
                                            # Charging phase (reduces available capacity)
                                            if (time_of_day >= ev_charge_start_minute and time_of_day < ev_charge_end_minute):
                                                ev_charge_power = grid_battery_max_rate
                                                adjusted_grid_profile[minute] -= ev_charge_power  # Decrease available capacity
                                            
                                            # Discharging phase (increases available capacity)
                                            elif (time_of_day >= ev_discharge_start_minute and time_of_day < ev_discharge_end_minute):
                                                ev_discharge_power = grid_battery_max_rate
                                                adjusted_grid_profile[minute] += ev_discharge_power  # Increase available capacity
                            
                            # Apply V2G optimization if enabled
                            if 'v2g' in active_strategies:
                                v2g_adoption_percent = st.session_state.optimization_strategy.get('v2g_adoption_percent', 0)
                                v2g_max_discharge_rate = st.session_state.optimization_strategy.get('v2g_max_discharge_rate', 0)
                                v2g_discharge_start_hour = st.session_state.optimization_strategy.get('v2g_discharge_start_hour', 18)
                                v2g_discharge_duration = st.session_state.optimization_strategy.get('v2g_discharge_duration', 3)
                                
                                if v2g_adoption_percent > 0 and v2g_max_discharge_rate > 0:
                                    # Use the ORIGINAL number of EVs for battery effects (existing infrastructure)
                                    total_evs_for_v2g = original_total_evs
                                    v2g_evs = int(total_evs_for_v2g * v2g_adoption_percent / 100)
                                    
                                    # Generate variable start times
                                    v2g_use_normal_distribution = st.session_state.optimization_strategy.get('v2g_use_normal_distribution', True)
                                    v2g_sigma_divisor = st.session_state.optimization_strategy.get('v2g_sigma_divisor', 8)
                                    
                                    if v2g_use_normal_distribution and v2g_sigma_divisor:
                                        discharge_sigma = v2g_discharge_duration / v2g_sigma_divisor
                                        discharge_start_times = np.random.normal(v2g_discharge_start_hour, discharge_sigma, v2g_evs)
                                        discharge_start_times = np.clip(discharge_start_times, 0, 24)
                                    else:
                                        discharge_start_times = np.full(v2g_evs, v2g_discharge_start_hour)
                                    
                                    # Calculate time periods for each EV
                                    for ev_idx in range(v2g_evs):
                                        ev_discharge_start = discharge_start_times[ev_idx]
                                        
                                        # Convert to minutes
                                        ev_discharge_start_minute = int(ev_discharge_start * 60)
                                        ev_discharge_end_minute = ev_discharge_start_minute + int(v2g_discharge_duration * 60)
                                        
                                        for minute in range(len(grid_profile_full)):
                                            time_of_day = minute % (24 * 60)
                                            
                                            # Discharging phase (increases available capacity)
                                            if (time_of_day >= ev_discharge_start_minute and time_of_day < ev_discharge_end_minute):
                                                ev_discharge_power = v2g_max_discharge_rate
                                                adjusted_grid_profile[minute] += ev_discharge_power  # Increase available capacity
                            
                            # Apply margin to the adjusted grid profile (EXACTLY like main simulation)
                            margin_curve = adjusted_grid_profile * available_load_fraction
                            
                            # Extend for 48 hours (2880 minutes) - EXACTLY like main simulation
                            sim_duration_min = 48 * 60  # 48 hours
                            
                            if len(margin_curve) < sim_duration_min:
                                # Repeat the profile to cover 48 hours
                                num_repeats = sim_duration_min // len(margin_curve) + 1
                                margin_curve = np.tile(margin_curve, num_repeats)[:sim_duration_min]
                            else:
                                margin_curve = margin_curve[:sim_duration_min]
                            
                            return margin_curve.tolist()
                        except Exception as e:
                            st.error(f"Error extracting margin curve with battery effects: {e}")
                            return None
                    
                    # Extract margin curve with battery effects
                    margin_curve = extract_margin_curve_with_battery_effects()
                    
                    if margin_curve is not None:
                        # Use the new TOU optimizer
                        optimal_result = optimize_tou_periods_24h(margin_curve, num_periods)
                        
                        # Update the timeline with optimal periods
                        st.session_state.time_of_use_timeline = {
                            'periods': optimal_result['periods'],
                            'selected_period': 0
                        }
                        
                        # Store the visualization
                        st.session_state.optimal_tou_visualization = optimal_result['visualization']
                        
                        # Clear any existing hour assignments
                        st.session_state.pop('hour_assignments', None)
                        st.session_state.pop('initial_timeline', None)
                        
                       
                        st.rerun()
                    else:
                        st.error("❌ No grid data available. Please select a dataset or generate synthetic data first.")
            
            # Display saved optimal TOU visualization if available (right after the button)
            if 'optimal_tou_visualization' in st.session_state:
                st.write("**📈 Optimal TOU Analysis Visualization:**")
                st.pyplot(st.session_state.optimal_tou_visualization, use_container_width=True)
                
                # Display period details from session state
                if 'optimal_tou_period_details' in st.session_state:
                    st.write("**📊 Optimal TOU Period Details (First 24 Hours):**")
                    for detail in st.session_state.optimal_tou_period_details:
                        st.write(f"• **{detail['name']}** ({detail['capacity_level']}): {detail['capacity_range_start']:.1f} - {detail['capacity_range_end']:.1f} kW")
            
            # Update timeline based on session state assignments
            if 'hour_assignments' in st.session_state and st.session_state.hour_assignments:
                # Clear all hour assignments in timeline
                for period in timeline['periods']:
                    period['hours'] = []
                
                # Reassign hours based on session state
                for hour, period_name in st.session_state.hour_assignments.items():
                    for period in timeline['periods']:
                        if period['name'] == period_name:
                            period['hours'].append(hour)
                            break
            else:
                # Initialize session state with default timeline assignments if empty
                if 'hour_assignments' not in st.session_state or not st.session_state.hour_assignments:
                    # Store initial timeline if not already stored
                    if 'initial_timeline' not in st.session_state:
                        st.session_state.initial_timeline = {}
                        for period in timeline['periods']:
                            for hour in period['hours']:
                                st.session_state.initial_timeline[hour] = period['name']
                    else:
                        # Use current timeline as initial values
                        for period in timeline['periods']:
                            for hour in period['hours']:
                                st.session_state.hour_assignments[hour] = period['name']
            
            # Update session state - convert timeline format to simulation format
            periods = []
            for period in timeline['periods']:
                if period['hours']:  # Only include periods with assigned hours
                    # Find continuous ranges of hours
                    sorted_hours = sorted(period['hours'])
                    ranges = []
                    start = sorted_hours[0]
                    end = start
                    
                    for hour in sorted_hours[1:]:
                        if hour == end + 1:  # Changed from 0.5 to 1 for 1-hour steps
                            end = hour
                        else:
                            ranges.append((start, end + 1))
                            start = hour
                            end = hour
                    
                    # Add the last range
                    ranges.append((start, end + 1))
                    
                    # Create a period for each range
                    for start_hour, end_hour in ranges:
                        periods.append({
                            'start': start_hour - 1,  # Convert from 1-24 to 0-23 format for simulation
                            'end': end_hour - 1,      # Convert from 1-24 to 0-23 format for simulation
                            'name': period['name'],
                            'color': period['color'],
                            'adoption': period['adoption']
                        })
            
            st.session_state.optimization_strategy['time_of_use_periods'] = periods
            st.session_state.optimization_strategy['smart_charging_percent'] = total_adoption

        # Clear simulation results if time of use parameters changed
        st.session_state.simulation_just_run = False
        # Don't clear simulation_results - let them persist
        
        # PV + Battery configuration
        if 'pv_battery' in active_strategies:
            st.write("**☀️ PV + Battery System Configuration:**")
            st.write("PV systems charge batteries during day, then discharge during evening peak.")
            
            # Normal distribution settings for PV Battery (at the top)
            st.write("**📊 Timing Distribution Settings:**")
            col1, col2 = st.columns(2)
            with col1:
                pv_use_normal_distribution = st.checkbox(
                    "Use Normal Distribution",
                    value=st.session_state.optimization_strategy.get('pv_use_normal_distribution', True),
                    help="Enable normal distribution for start times (vs strict boundaries)",
                    key="pv_use_normal_distribution"
                )
            with col2:
                if pv_use_normal_distribution:
                    pv_distribution_level = st.selectbox(
                        "Distribution Level",
                        options=["Low (1/16)", "Medium (1/8)", "High (1/4)"],
                        index=st.session_state.optimization_strategy.get('pv_distribution_level', 1),  # Default to Medium
                        help="Standard deviation level for normal distribution",
                        key="pv_distribution_level"
                    )
                    # Convert level to numeric value
                    if pv_distribution_level == "Low (1/16)":
                        pv_sigma_divisor = 16
                    elif pv_distribution_level == "Medium (1/8)":
                        pv_sigma_divisor = 8
                    else:  # High (1/4)
                        pv_sigma_divisor = 4
                else:
                    pv_sigma_divisor = None
            
            scenario_year = st.session_state.get('portugal_scenario', {}).get('year', '2030')
            try:
                scenario_year = int(scenario_year)
            except Exception:
                scenario_year = 2030
            if scenario_year <= 2027:
                default_battery_capacity = 12.5
                max_discharge_rate = 5
                max_charge_rate = 3
            elif scenario_year <= 2030:
                default_battery_capacity = 17.5
                max_discharge_rate = 7
                max_charge_rate = 5
            else:
                default_battery_capacity = 25
                max_discharge_rate = 10
                max_charge_rate = 7
            
            col1, col2 = st.columns(2)
            with col1:
                battery_capacity = st.number_input(
                    "Battery Capacity (kWh)",
                    min_value=1.0,
                    max_value=50.0,
                    value=float(st.session_state.optimization_strategy.get('battery_capacity', default_battery_capacity)),
                    step=0.5,
                    help="Usable battery capacity (kWh)",
                    key="pv_battery_capacity"
                )
                max_charge_rate = st.number_input(
                    "Maximum Charge Rate (kW)",
                    min_value=1.0,
                    max_value=20.0,
                    value=float(st.session_state.optimization_strategy.get('max_charge_rate', max_charge_rate)),
                    step=0.5,
                    help="Maximum charge rate of each battery (kW)",
                    key="pv_max_charge_rate"
                )
                max_discharge_rate = st.number_input(
                    "Maximum Discharge Rate (kW)",
                    min_value=1.0,
                    max_value=20.0,
                    value=float(st.session_state.optimization_strategy.get('max_discharge_rate', max_discharge_rate)),
                    step=0.5,
                    help="Maximum discharge rate of each battery (kW)",
                    key="pv_max_discharge_rate"
                )
                pv_start_hour = st.slider(
                    "PV Start Hour",
                    min_value=6.0,
                    max_value=18.0,
                    value=float(st.session_state.optimization_strategy.get('pv_start_hour', 8.0)),  # 8am default
                    step=0.5,
                    help="Hour when PV system starts working (6-18)",
                    key="pv_start_hour"
                )
                pv_duration = st.slider(
                    "PV Duration (hours)",
                    min_value=1.0,
                    max_value=12.0,
                    value=float(st.session_state.optimization_strategy.get('pv_duration', 8.0)),  # 8 hours default
                    step=0.5,
                    help="Duration of PV system operation (hours)",
                    key="pv_duration"
                )
            with col2:
                discharge_start_hour = st.slider(
                    "Discharge Start Hour",
                    min_value=16.0,
                    max_value=26.0,
                    value=float(st.session_state.optimization_strategy.get('discharge_start_hour', 18)),  # 8pm default
                    step=0.5,
                    help="Hour when battery discharge starts (18-26)",
                    key="pv_discharge_start_hour"
                )
                min_discharge_duration = battery_capacity / max_discharge_rate
                # Round to nearest 0.5 to align with step
                min_discharge_duration_rounded = round(min_discharge_duration * 2) / 2
                # Use 3.0 as default if no value in session state, otherwise use the calculated minimum
                if 'discharge_duration' not in st.session_state.optimization_strategy:
                    default_duration = 6.0
                else:
                    default_duration = st.session_state.optimization_strategy.get('discharge_duration', min_discharge_duration_rounded)
                # Ensure default also aligns with step
                default_duration = round(default_duration * 2) / 2
                discharge_duration = st.slider(
                    "Discharge Duration (hours)",
                    min_value=min_discharge_duration_rounded,
                    max_value=12.0,
                    value=default_duration,
                    step=0.5,
                    help="How long to spread the battery discharge (must be at least enough to fully discharge at max rate)",
                    key="pv_discharge_duration"
                )
                pv_adoption_percent = st.slider(
                    "PV + Battery Adoption (%)",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.optimization_strategy.get('pv_adoption_percent', 30),
                    step=5,
                    help="Percentage of households with PV + battery systems",
                    key="pv_adoption_percent"
                )
                solar_energy_percent = st.slider(
                    "Stored Solar Energy Available (%)",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.optimization_strategy.get('solar_energy_percent', 70),
                    step=5,
                    help="Percentage of stored solar energy available for evening peak (60-80%)",
                    key="pv_solar_energy_percent"
                )
            
            # Calculate charging time and system support periods
            charge_time = battery_capacity / max_charge_rate
            system_support_time = pv_duration - charge_time
            
            # Info statement at bottom, full width
            if discharge_duration == min_discharge_duration:
                actual_discharge_rate = max_discharge_rate
            else:
                actual_discharge_rate = min(battery_capacity / discharge_duration, max_discharge_rate)
            
            st.info(f"💡 PV will charge battery for {charge_time:.1f}h starting at {pv_start_hour}:00, then provide {system_support_time:.1f}h of system support. Battery will discharge for {discharge_duration:.1f}h starting at {discharge_start_hour}:00 at {actual_discharge_rate:.2f} kW (max possible: {max_discharge_rate} kW)")
            
            # Update session state for PV + battery
            st.session_state.optimization_strategy['battery_capacity'] = battery_capacity
            st.session_state.optimization_strategy['max_charge_rate'] = max_charge_rate
            st.session_state.optimization_strategy['max_discharge_rate'] = max_discharge_rate
            st.session_state.optimization_strategy['pv_start_hour'] = pv_start_hour
            st.session_state.optimization_strategy['pv_duration'] = pv_duration
            st.session_state.optimization_strategy['discharge_start_hour'] = discharge_start_hour
            st.session_state.optimization_strategy['discharge_duration'] = discharge_duration
            st.session_state.optimization_strategy['actual_discharge_rate'] = actual_discharge_rate
            st.session_state.optimization_strategy['pv_adoption_percent'] = pv_adoption_percent
            st.session_state.optimization_strategy['solar_energy_percent'] = solar_energy_percent
            st.session_state.optimization_strategy['charge_time'] = charge_time
            st.session_state.optimization_strategy['system_support_time'] = system_support_time
            st.session_state.optimization_strategy['pv_use_normal_distribution'] = pv_use_normal_distribution
            st.session_state.optimization_strategy['pv_distribution_level'] = ["Low (1/16)", "Medium (1/8)", "High (1/4)"].index(pv_distribution_level) if pv_use_normal_distribution else 1
            st.session_state.optimization_strategy['pv_sigma_divisor'] = pv_sigma_divisor
            
            # Clear simulation results if PV battery parameters changed
            st.session_state.simulation_just_run = False
            # Don't clear simulation_results - let them persist

        # Grid-Charged Batteries configuration
        if 'grid_battery' in active_strategies:
            st.write("**🔋 Grid-Charged Batteries Configuration:**")
            st.write("Batteries charged from grid during day, discharged during evening peak.")
            
            # Normal distribution settings for Grid-Charged Batteries (at the top)
            st.write("**📊 Timing Distribution Settings:**")
            col1, col2 = st.columns(2)
            with col1:
                grid_use_normal_distribution = st.checkbox(
                    "Use Normal Distribution",
                    value=st.session_state.optimization_strategy.get('grid_use_normal_distribution', True),
                    help="Enable normal distribution for start times (vs strict boundaries)",
                    key="grid_use_normal_distribution"
                )
            with col2:
                if grid_use_normal_distribution:
                    grid_distribution_level = st.selectbox(
                        "Distribution Level",
                        options=["Low (1/16)", "Medium (1/8)", "High (1/4)"],
                        index=st.session_state.optimization_strategy.get('grid_distribution_level', 1),  # Default to Medium
                        help="Standard deviation level for normal distribution",
                        key="grid_distribution_level"
                    )
                    # Convert level to numeric value
                    if grid_distribution_level == "Low (1/16)":
                        grid_sigma_divisor = 16
                    elif grid_distribution_level == "Medium (1/8)":
                        grid_sigma_divisor = 8
                    else:  # High (1/4)
                        grid_sigma_divisor = 4
                else:
                    grid_sigma_divisor = None
            
            scenario_year = st.session_state.get('portugal_scenario', {}).get('year', '2030')
            try:
                scenario_year = int(scenario_year)
            except Exception:
                scenario_year = 2030
            if scenario_year <= 2027:
                default_grid_battery_capacity = 12.5
                default_grid_battery_max_rate = 3.0
                default_grid_battery_adoption = 5
            elif scenario_year <= 2030:
                default_grid_battery_capacity = 20.0
                default_grid_battery_max_rate = 5.0
                default_grid_battery_adoption = 10
            else:
                default_grid_battery_capacity = 27.5
                default_grid_battery_max_rate = 7.0
                default_grid_battery_adoption = 15
            
            grid_battery_capacity = st.number_input(
                "Battery Capacity (kWh)",
                min_value=1.0,
                max_value=50.0,
                value=float(st.session_state.optimization_strategy.get('grid_battery_capacity', default_grid_battery_capacity)),
                step=0.5,
                help="Usable battery capacity (kWh)",
                key="grid_battery_capacity"
            )
            grid_battery_max_rate = st.number_input(
                "Maximum Charge/Discharge Rate (kW)",
                min_value=1.0,
                max_value=20.0,
                value=float(st.session_state.optimization_strategy.get('grid_battery_max_rate', default_grid_battery_max_rate)),
                step=0.5,
                help="Maximum charge and discharge rate of each battery (kW)",
                key="grid_battery_max_rate"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                grid_battery_charge_start_hour = st.slider(
                    "Charging Start Hour",
                    min_value=0,
                    max_value=23,
                    value=st.session_state.optimization_strategy.get('grid_battery_charge_start_hour', 7),
                    help="Hour when battery charging starts (0-23)",
                    key="grid_charge_start_hour"
                )
                grid_battery_charge_duration = st.slider(
                    "Charge Duration (hours)",
                    min_value=1,
                    max_value=12,
                    value=int(st.session_state.optimization_strategy.get('grid_battery_charge_duration', 8)),
                    step=1,
                    help="Duration of battery charging (hours)",
                    key="grid_charge_duration"
                )
            with col2:
                grid_battery_discharge_start_hour = st.slider(
                    "Discharge Start Hour",
                    min_value=0.0,
                    max_value=26.0,
                    value=float(st.session_state.optimization_strategy.get('grid_battery_discharge_start_hour', 18)),
                    step=0.5,
                    help="Hour when battery discharge starts (18-26)",
                    key="grid_discharge_start_hour"
                )
                grid_battery_discharge_duration = st.slider(
                    "Discharge Duration (hours)",
                    min_value=1,
                    max_value=8,
                    value=int(st.session_state.optimization_strategy.get('grid_battery_discharge_duration', 6)),
                    step=1,
                    help="Duration of battery discharge (hours)",
                    key="grid_discharge_duration"
                )
            
            grid_battery_adoption_percent = st.slider(
                "Grid-Charged Battery Adoption (%)",
                min_value=0,
                max_value=100,
                value=int(st.session_state.optimization_strategy.get('grid_battery_adoption_percent', default_grid_battery_adoption)),
                step=5,
                help="Percentage of households with grid-charged battery systems",
                key="grid_battery_adoption_percent"
            )
            
            # Calculate actual charge/discharge rates (respecting max rate limits)
            actual_charge_rate = min(grid_battery_capacity / grid_battery_charge_duration, grid_battery_max_rate)
            actual_discharge_rate = min(grid_battery_capacity / grid_battery_discharge_duration, grid_battery_max_rate)
            st.info(f"💡 Battery will charge for {grid_battery_charge_duration}h starting at {grid_battery_charge_start_hour}:00 at {actual_charge_rate:.2f} kW, then discharge for {grid_battery_discharge_duration}h starting at {grid_battery_discharge_start_hour}:00 at {actual_discharge_rate:.2f} kW (max possible: {grid_battery_max_rate:.1f} kW)")
            
            # Update session state for grid-charged batteries
            st.session_state.optimization_strategy['grid_battery_capacity'] = grid_battery_capacity
            st.session_state.optimization_strategy['grid_battery_max_rate'] = grid_battery_max_rate
            st.session_state.optimization_strategy['grid_battery_charge_start_hour'] = grid_battery_charge_start_hour
            st.session_state.optimization_strategy['grid_battery_charge_duration'] = grid_battery_charge_duration
            st.session_state.optimization_strategy['grid_battery_discharge_start_hour'] = grid_battery_discharge_start_hour
            st.session_state.optimization_strategy['grid_battery_discharge_duration'] = grid_battery_discharge_duration
            st.session_state.optimization_strategy['grid_battery_adoption_percent'] = grid_battery_adoption_percent
            st.session_state.optimization_strategy['grid_use_normal_distribution'] = grid_use_normal_distribution
            st.session_state.optimization_strategy['grid_distribution_level'] = ["Low (1/16)", "Medium (1/8)", "High (1/4)"].index(grid_distribution_level) if grid_use_normal_distribution else 1
            st.session_state.optimization_strategy['grid_sigma_divisor'] = grid_sigma_divisor
            
            # Clear simulation results if grid battery parameters changed
            st.session_state.simulation_just_run = False
            # Don't clear simulation_results - let them persist
        
        # V2G (Vehicle-to-Grid) configuration
        if 'v2g' in active_strategies:
            st.write("**🚗 Vehicle-to-Grid (V2G) Configuration:**")
            st.write("EVs can discharge back to grid during peak demand, then recharge later.")
            
            # Normal distribution settings for V2G (at the top)
            st.write("**📊 Timing Distribution Settings:**")
            col1, col2 = st.columns(2)
            with col1:
                v2g_use_normal_distribution = st.checkbox(
                    "Use Normal Distribution",
                    value=st.session_state.optimization_strategy.get('v2g_use_normal_distribution', True),
                    help="Enable normal distribution for start times (vs strict boundaries)",
                    key="v2g_use_normal_distribution"
                )
            with col2:
                if v2g_use_normal_distribution:
                    v2g_distribution_level = st.selectbox(
                        "Distribution Level",
                        options=["Low (1/16)", "Medium (1/8)", "High (1/4)"],
                        index=st.session_state.optimization_strategy.get('v2g_distribution_level', 1),  # Default to Medium
                        help="Standard deviation level for normal distribution",
                        key="v2g_distribution_level"
                    )
                    # Convert level to numeric value
                    if v2g_distribution_level == "Low (1/16)":
                        v2g_sigma_divisor = 16
                    elif v2g_distribution_level == "Medium (1/8)":
                        v2g_sigma_divisor = 8
                    else:  # High (1/4)
                        v2g_sigma_divisor = 4
                else:
                    v2g_sigma_divisor = None
            
            # Set default V2G adoption based on scenario year
            scenario_year = st.session_state.get('portugal_scenario', {}).get('year', '2030')
            try:
                scenario_year = int(scenario_year)
            except Exception:
                scenario_year = 2030
            
            if scenario_year <= 2027:
                default_v2g_adoption = 5
            elif scenario_year <= 2030:
                default_v2g_adoption = 20
            else:
                default_v2g_adoption = 40
            
            col1, col2 = st.columns(2)
            with col1:
                v2g_adoption_percent = st.slider(
                    "V2G Adoption (%)",
                    min_value=0,
                    max_value=100,
                    value=int(st.session_state.optimization_strategy.get('v2g_adoption_percent', default_v2g_adoption)),
                    step=5,
                    help="Percentage of EVs that can participate in V2G",
                    key="v2g_adoption_percent"
                )
                v2g_discharge_start_hour = st.slider(
                    "V2G Discharge Start Hour",
                    min_value=18.0,
                    max_value=26.0,
                    value=float(st.session_state.optimization_strategy.get('v2g_discharge_start_hour', 18)),
                    step=0.5,
                    help="Hour when V2G discharge starts (18-26)",
                    key="v2g_discharge_start_hour"
                )
                v2g_max_discharge_rate = st.number_input(
                    "V2G Max Discharge Rate (kW)",
                    min_value=1.0,
                    max_value=22.0,
                    value=float(st.session_state.optimization_strategy.get('v2g_max_discharge_rate', 7.0)),
                    step=0.5,
                    help="Maximum discharge rate per V2G EV (kW)",
                    key="v2g_max_discharge_rate"
                )
            
            with col2:
                # Calculate minimum discharge duration based on battery capacity and max discharge rate
                ev_capacity = st.session_state.dynamic_ev.get('capacity', 75)  # Default 75 kWh
                full_discharge_time = ev_capacity / v2g_max_discharge_rate if v2g_max_discharge_rate > 0 else 1.0
                
                # Use a fixed minimum of 1 hour to avoid slider bugs
                min_discharge_duration = max(1.0, full_discharge_time)
                
                # Get current value and ensure it's valid
                current_duration = st.session_state.optimization_strategy.get('v2g_discharge_duration', 3.0)
                # If current value is less than new minimum, use the minimum
                if current_duration < min_discharge_duration:
                    current_duration = min_discharge_duration
                
                v2g_discharge_duration = st.slider(
                    "V2G Discharge Duration (hours)",
                    min_value=1.0,
                    max_value=10.0,
                    value=current_duration,
                    step=0.5,
                    help=f"Duration of V2G discharge (hours). Full discharge time: {full_discharge_time:.1f}h at {v2g_max_discharge_rate} kW",
                    key="v2g_discharge_duration"
                )
                v2g_recharge_arrival_hour = st.slider(
                    "V2G Recharge Arrival Hour",
                    min_value=0.0,
                    max_value=30.0,
                    value=float(st.session_state.optimization_strategy.get('v2g_recharge_arrival_hour', 26.0)),
                    step=0.5,
                    help="Hour when V2G EVs start recharging (0-30)",
                    key="v2g_recharge_arrival_hour"
                )

            
            # Calculate V2G effects
            # Get total EVs from all peaks
            if 'time_peaks' in st.session_state and st.session_state.time_peaks:
                total_evs = sum(peak.get('quantity', 0) for peak in st.session_state.time_peaks)
            elif 'ev_calculator' in st.session_state and 'total_evs' in st.session_state.ev_calculator:
                total_evs = st.session_state.ev_calculator['total_evs']
            else:
                # Use a reasonable default if no peaks or calculator available
                total_evs = 32  # Default fallback
            
            total_v2g_evs = int(total_evs * v2g_adoption_percent / 100)
            
            # Calculate realistic discharge based on battery capacity constraints
            ev_capacity = st.session_state.dynamic_ev.get('capacity', 75)  # Default 75 kWh
            full_discharge_time = ev_capacity / v2g_max_discharge_rate if v2g_max_discharge_rate > 0 else 1.0
            
            if v2g_discharge_duration >= full_discharge_time:
                # Can fully discharge - use slower rate if needed
                actual_discharge_rate = ev_capacity / v2g_discharge_duration
                discharge_percentage = 1.0  # 100% discharged
                recharge_evs = total_v2g_evs  # All EVs arrive to recharge
            else:
                # Can only partially discharge
                actual_discharge_rate = v2g_max_discharge_rate
                discharge_percentage = v2g_discharge_duration / full_discharge_time
                recharge_evs = int(total_v2g_evs * discharge_percentage)  # Only partially discharged EVs arrive
            
            total_v2g_discharge = total_v2g_evs * actual_discharge_rate
            
            st.info(f"💡 {total_v2g_evs} V2G EVs will discharge {total_v2g_discharge:.1f} kW for {v2g_discharge_duration:.2f}h starting at {v2g_discharge_start_hour}:00 at {actual_discharge_rate:.2f} kW (max possible: {v2g_max_discharge_rate:.1f} kW), then {recharge_evs} EVs will arrive to recharge at {v2g_recharge_arrival_hour}:00 ({discharge_percentage:.0%} discharged)")
            
            # Update session state for V2G
            st.session_state.optimization_strategy['v2g_adoption_percent'] = v2g_adoption_percent
            st.session_state.optimization_strategy['v2g_discharge_start_hour'] = v2g_discharge_start_hour
            st.session_state.optimization_strategy['v2g_discharge_duration'] = v2g_discharge_duration
            st.session_state.optimization_strategy['v2g_max_discharge_rate'] = v2g_max_discharge_rate
            st.session_state.optimization_strategy['v2g_recharge_arrival_hour'] = v2g_recharge_arrival_hour
            st.session_state.optimization_strategy['v2g_use_normal_distribution'] = v2g_use_normal_distribution
            st.session_state.optimization_strategy['v2g_distribution_level'] = ["Low (1/16)", "Medium (1/8)", "High (1/4)"].index(v2g_distribution_level) if v2g_use_normal_distribution else 1
            st.session_state.optimization_strategy['v2g_sigma_divisor'] = v2g_sigma_divisor
            
            # Clear simulation results if V2G parameters changed
            st.session_state.simulation_just_run = False
            # Don't clear simulation_results - let them persist

 # Reverse Simulation (Capacity Analysis)
    with st.expander("🔍 Reverse Simulation", expanded=False):
         # First Optimizer: Maximum Cars Analysis
        st.subheader("🎯 Maximum Cars Optimizer")
        st.write("Find the maximum number of EVs that can fit under the margin curve with current parameters.")
        
        # Multi-step capacity analysis
        num_steps = st.slider("Number of Analysis Steps", min_value=1, max_value=4, value=1, 
                              help="Run capacity analysis in multiple steps for more accurate results")
        
       
        
        # Capacity Analysis Button (always enabled, auto-generates data if missing)
        if st.button("🎯 Find Maximum Cars", type="primary"):
            # Get current configuration parameters
            ev_config = st.session_state.dynamic_ev
            charger_config = st.session_state.charger_config
            time_peaks = st.session_state.get('time_peaks', [])
            active_strategies = st.session_state.get('active_strategies', [])
            grid_mode = st.session_state.get('grid_mode', 'Reference Only')
            available_load_fraction = st.session_state.get('available_load_fraction', 0.8)
            data_source = st.session_state.get('data_source', 'Real Dataset')
            power_values = st.session_state.get('power_values', None)
            # Handle data loading/generation (EXACTLY like simulation button)
            if data_source == "Synthetic Generation":
                if 'synthetic_load_curve' in st.session_state and st.session_state.synthetic_load_curve is not None and len(st.session_state.synthetic_load_curve) > 0:
                    power_values = st.session_state.synthetic_load_curve
                    st.session_state.power_values = power_values
                elif power_values is None or len(power_values) == 0:
                    st.info("🎲 Generating synthetic load curve based on current parameters...")
                    try:
                        import sys
                        import os
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from portable_load_generator import generate_load_curve
                        synthetic_params = st.session_state.get('synthetic_params', {})
                        season = synthetic_params.get('season', 'winter')
                        day_type = synthetic_params.get('day_type', 'weekday')
                        max_power = synthetic_params.get('max_power', 400)
                        diversity_mode = synthetic_params.get('diversity_mode', 'high')
                        result = generate_load_curve(
                            season=season,
                            day_type=day_type,
                            max_power=max_power,
                            diversity_mode=diversity_mode,
                            models_dir="portable_models",
                            return_timestamps=True
                        )
                        st.session_state.synthetic_load_curve = result['load_curve']
                        st.session_state.synthetic_timestamps = result['timestamps']
                        st.session_state.synthetic_metadata = result['metadata']
                        power_values = result['load_curve']
                        st.session_state.power_values = power_values
                        st.success(f"✅ Synthetic load curve generated automatically for capacity analysis!")
                    except Exception as e:
                        st.error(f"❌ Error generating synthetic data: {e}")
                        st.write("Please ensure the portable_models directory contains the trained models.")
                        st.stop()
            elif data_source == "Real Dataset":
                # For Real Dataset, try to use existing data or fall back to synthetic
                if power_values is None or len(power_values) == 0:
                    st.info("📊 No real dataset loaded. Falling back to synthetic generation...")
                    try:
                        import sys
                        import os
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from portable_load_generator import generate_load_curve
                        synthetic_params = st.session_state.get('synthetic_params', {})
                        season = synthetic_params.get('season', 'winter')
                        day_type = synthetic_params.get('day_type', 'weekday')
                        max_power = synthetic_params.get('max_power', 400)
                        diversity_mode = synthetic_params.get('diversity_mode', 'high')
                        result = generate_load_curve(
                            season=season,
                            day_type=day_type,
                            max_power=max_power,
                            diversity_mode=diversity_mode,
                            models_dir="portable_models",
                            return_timestamps=True
                        )
                        st.session_state.synthetic_load_curve = result['load_curve']
                        st.session_state.synthetic_timestamps = result['timestamps']
                        st.session_state.synthetic_metadata = result['metadata']
                        power_values = result['load_curve']
                        st.session_state.power_values = power_values
                        st.success(f"✅ Synthetic load curve generated automatically for capacity analysis!")
                    except Exception as e:
                        st.error(f"❌ Error generating synthetic data: {e}")
                        st.write("Please ensure the portable_models directory contains the trained models.")
                        st.stop()
            # Check if we have valid data
            if power_values is None or len(power_values) == 0:
                st.error("❌ No valid dataset available. Please load a dataset or generate synthetic data first.")
                st.stop()
            
            with st.spinner("Calculating maximum EV capacity..."):
                max_cars = find_max_cars_capacity(
                    ev_config=ev_config,
                    charger_config=charger_config,
                    time_peaks=time_peaks,
                    active_strategies=active_strategies,
                    grid_mode=grid_mode,
                    available_load_fraction=available_load_fraction,
                    power_values=power_values,
                    sim_duration=sim_duration,
                    num_steps=num_steps
                )
                
                if max_cars is not None:
                    st.session_state.capacity_analysis_results = {
                        'max_cars': max_cars,
                        'ev_config': ev_config,
                        'charger_config': charger_config,
                        'time_peaks': time_peaks,
                        'active_strategies': active_strategies,
                        'grid_mode': grid_mode,
                        'available_load_fraction': available_load_fraction,
                        'power_values': power_values
                    }
                    
                    # Automatically apply the max cars results
                    st.session_state.total_evs = max_cars
                    st.session_state.charger_config['ac_count'] = max_cars
                    
                    # Check if Time of Use is enabled
                    time_of_use_enabled = ('smart_charging' in active_strategies and 
                                          st.session_state.optimization_strategy.get('smart_charging_percent', 0) > 0)
                    
                    if time_of_use_enabled:
                        # For TOU, apply all cars to first peak (same as before)
                        if 'time_peaks' in st.session_state and len(st.session_state.time_peaks) > 0:
                            st.session_state.time_peaks[0]['quantity'] = max_cars
                    else:
                        # For multiple peaks, distribute cars proportionally
                        if 'time_peaks' in st.session_state and len(st.session_state.time_peaks) > 0:
                            # Calculate current proportions
                            current_total = sum(peak.get('quantity', 0) for peak in st.session_state.time_peaks)
                            
                            if current_total > 0:
                                # Distribute proportionally based on current ratios
                                for peak in st.session_state.time_peaks:
                                    current_quantity = peak.get('quantity', 0)
                                    proportion = current_quantity / current_total
                                    new_quantity = int(max_cars * proportion)
                                    peak['quantity'] = new_quantity
                                
                                # Ensure we don't lose cars due to rounding
                                distributed_total = sum(peak.get('quantity', 0) for peak in st.session_state.time_peaks)
                                if distributed_total < max_cars:
                                    # Add remaining cars to the first peak
                                    remaining = max_cars - distributed_total
                                    st.session_state.time_peaks[0]['quantity'] += remaining
                            else:
                                # If no cars currently, put all in first peak
                                st.session_state.time_peaks[0]['quantity'] = max_cars
                    
                    # Display capacity analysis results table
                    st.write("**📊 Capacity Analysis Results:**")
                    st.write(f"**Maximum EVs: {max_cars}**")
                    
                    # Store results in session state for persistence
                    st.session_state.capacity_analysis_completed = True
                    st.session_state.capacity_analysis_max_cars = max_cars
                    
                    # Rerun to apply the values to the configuration
                    st.rerun()
                else:
                    st.error("❌ Could not determine maximum capacity")
                
        # Display stored capacity analysis results if available
        if st.session_state.get('capacity_analysis_completed', False):
            max_cars = st.session_state.get('capacity_analysis_max_cars', 0)
            
            if max_cars > 0:
                st.write("**📊 Capacity Analysis Results:**")
                st.write(f"**Maximum EVs: {max_cars}**")
        
        # RL Capacity Optimizer (separate from first optimizer)
        try:
            # Lazy import to avoid heavy imports on page load
            from pages.components.gradient_optimizer_ui import create_gradient_optimizer_ui
            create_gradient_optimizer_ui()
        except Exception as e:
            st.error(f"❌ Gradient-Based Capacity Optimizer error: {e}")
            st.info("💡 This feature requires scikit-learn. Install with: `pip install scikit-learn`")



    # EV Number Calculator
    with st.expander("🧮 EV Number Calculator", expanded=False):
        st.write("**Calculate EVs per substation based on penetration and infrastructure:**")
        
        # Initialize calculator values in session state if not exists
        if 'ev_calculator' not in st.session_state:
            st.session_state.ev_calculator = {
                'num_substations': 69000,
                'penetration_percent': 20.0,
                'total_cars': 6100000,
                'home_charging_percent': 80.0
            }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            num_substations = st.number_input(
                "Number of Substations",
                min_value=1,
                value=st.session_state.ev_calculator['num_substations'],
                help="Total number of substations in the grid"
            )
        
        with col2:
            penetration_percent = st.number_input(
                "EV Penetration (%)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.ev_calculator['penetration_percent'],
                step=0.1,
                help="Percentage of total cars that are electric"
            )
        
        with col3:
            total_cars = st.number_input(
                "Total Number of Cars",
                min_value=0,
                value=st.session_state.ev_calculator['total_cars'],
                help="Total number of cars in the system"
            )
        
        with col4:
            home_charging_percent = st.number_input(
                "Home Charging (%)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.ev_calculator.get('home_charging_percent', 80.0),
                step=1.0,
                help="Percentage of EVs that charge at home"
            )
        
        # Calculate EVs per substation and home charging
        if num_substations > 0:
            total_evs = total_cars * penetration_percent / 100
            evs_per_substation = total_evs / num_substations
            home_charged_evs = total_evs * home_charging_percent / 100
            public_charged_evs = total_evs - home_charged_evs
            
            st.write(f"**📊 Results:**")
            st.write(f"• **Total EVs:** {total_evs:,.0f}")
            st.write(f"• **EVs per Substation:** {evs_per_substation:.2f}")
            st.write(f"• **Home Charged EVs:** {home_charged_evs:,.0f} ({home_charging_percent:.0f}%)")
            st.write(f"• **Public Charged EVs:** {public_charged_evs:,.0f} ({100-home_charging_percent:.0f}%)")
            
            # Update session state
            st.session_state.ev_calculator = {
                'num_substations': num_substations,
                'penetration_percent': penetration_percent,
                'total_cars': total_cars,
                'home_charging_percent': home_charging_percent,
                'evs_per_substation': evs_per_substation,
                'total_evs': total_evs,
                'home_charged_evs': home_charged_evs,
                'public_charged_evs': public_charged_evs
            }
        else:
            st.warning("Please enter a valid number of substations (greater than 0)")
            evs_per_substation = 0



    # Dataset Selection
    with st.expander("📊 Dataset Selection", expanded=False):
        # Data source selection
        st.write("**Data Source:**")
        data_source = st.radio("Data Source", ["Real Dataset", "Synthetic Generation"],
                              help="Choose between real historical data or synthetic load curve generation")
        
        # Store data source in session state for access by other components
        st.session_state.data_source = data_source
        
        if data_source == "Real Dataset":
            # Original dataset selection logic
            dataset_files = ["df1.csv", "df2.csv", "df3.csv"]
            # Initialize dataset selection in session state if not exists
            if 'selected_dataset' not in st.session_state:
                st.session_state.selected_dataset = "df3.csv"
            
            selected_dataset = st.selectbox(
                "Select Dataset", 
                dataset_files,
                index=dataset_files.index(st.session_state.selected_dataset),
                key="dataset_selection"
            )
            # Update session state immediately when dataset changes
            if selected_dataset != st.session_state.selected_dataset:
                st.session_state.selected_dataset = selected_dataset
                st.rerun()
            
            if selected_dataset:
                try:
                    df = pd.read_csv(f"datasets/{selected_dataset}")
                    df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                    
                    # Get date range for the calendar
                    date_min = df['date'].min()
                    date_max = df['date'].max()
                    
                    if pd.notna(date_min) and pd.notna(date_max):
                        # Use date_input for nice calendar selection
                        # Initialize selected date in session state if not exists
                        if 'selected_date' not in st.session_state:
                            # Set default date to 2023/01/18
                            default_date = pd.to_datetime('2023-01-18').date()
                            # Check if default date is within the dataset range
                            if date_min.date() <= default_date <= date_max.date():
                                st.session_state.selected_date = default_date
                            else:
                                st.session_state.selected_date = date_min.date()
                        
                        selected_date = st.date_input(
                            "Select Date",
                            value=st.session_state.selected_date,
                            min_value=date_min.date(),
                            max_value=date_max.date(),
                            key="dataset_date_input"
                        )
                        # Update session state immediately when date changes
                        if selected_date != st.session_state.selected_date:
                            st.session_state.selected_date = selected_date
                            st.rerun()
                        
                        if selected_date:
                            # Filter data for selected date
                            day_data = df[df['date'].dt.date == selected_date]
                            
                            if not day_data.empty:
                                # Extract power values (3rd column, index 2) - no scaling here
                                power_values = day_data.iloc[:, 2].values
                                st.session_state.power_values = power_values  # Store in session state
                                
                                # Display dataset curve summary and preview
                                st.write("**📊 Dataset Curve Summary:**")
                                st.success(f"✅ Dataset load curve ready!")
                                st.write(f"**Loaded Profile:** {len(power_values)} data points")
                                st.write(f"**Mean Load:** {np.mean(power_values):.2f} kW")
                                st.write(f"**Max Load:** {np.max(power_values):.2f} kW")
                                st.write(f"**Min Load:** {np.min(power_values):.2f} kW")
                                st.write(f"**Dataset:** {selected_dataset}")
                                st.write(f"**Date:** {selected_date}")
                                
                                # Show curve preview
                                fig, ax = plt.subplots(figsize=(10, 4))
                                
                                # Calculate time axis based on actual data length
                                # Assuming data covers 24 hours, calculate interval
                                total_hours = 24
                                data_points = len(power_values)
                                interval_minutes = (total_hours * 60) / data_points
                                
                                # Create time axis in hours
                                time_hours = np.arange(len(power_values)) * interval_minutes / 60
                                
                                # Plot the data
                                ax.plot(time_hours, power_values, linewidth=1, alpha=0.8, color='blue')
                                ax.set_title("Dataset Load Curve Preview")
                                ax.set_xlabel("Time (hours)")
                                ax.set_ylabel("Load (kW)")
                                ax.grid(True, alpha=0.3)
                                
                                # Set x-axis to show the actual time range
                                ax.set_xlim(0, total_hours)
                                
                             
                                
                                st.pyplot(fig)
                                plt.close()
                            else:
                                st.error("No data found for selected date")
                                power_values = None
                        else:
                            power_values = None
                    else:
                        st.error("No valid dates found in dataset")
                        power_values = None
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
                    power_values = None
            else:
                power_values = None
                
        else:  # Synthetic Generation
            st.write("**🎲 Synthetic Load Curve Generation**")
            st.write("Generate synthetic load curves using trained AI models.")
            
            # Initialize synthetic parameters in session state if not exists
            if 'synthetic_params' not in st.session_state:
                st.session_state.synthetic_params = {
                    'day_type': 'weekday',
                    'season': 'winter',
                    'max_power': 400,
                    'diversity_mode': 'normal'
                }
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Initialize synthetic_params if not exists
                if 'synthetic_params' not in st.session_state:
                    st.session_state.synthetic_params = {
                        'day_type': 'weekday',
                        'season': 'winter',
                        'max_power': 400,
                        'diversity_mode': 'normal'
                    }
                
                # Day type selection
                day_type = st.selectbox(
                    "Day Type",
                    ["weekday", "weekend"],
                    index=0 if st.session_state.synthetic_params['day_type'] == 'weekday' else 1,
                    help="Choose between weekday or weekend load patterns"
                )
                st.session_state.synthetic_params['day_type'] = day_type
                
                # Season selection
                season = st.selectbox(
                    "Season",
                    ["winter", "spring", "summer", "autumn"],
                    index=["winter", "spring", "summer", "autumn"].index(st.session_state.synthetic_params['season']),
                    help="Select the season for load pattern generation"
                )
                st.session_state.synthetic_params['season'] = season
            
            with col2:
                # Maximum power selection
                max_power = st.slider(
                    "Maximum Substation Load (kW)",
                    min_value=200,
                    max_value=800,
                    value=st.session_state.synthetic_params['max_power'],
                    step=50,
                    help="Maximum load capacity of the substation (200-800 kW)"
                )
                st.session_state.synthetic_params['max_power'] = max_power
                
                # Diversity mode selection
                diversity_mode = st.selectbox(
                    "Diversity Mode",
                    ["normal", "high", "extreme"],
                    index=["normal", "high", "extreme"].index(st.session_state.synthetic_params['diversity_mode']),
                    help="Controls the variability of generated load curves"
                )
                st.session_state.synthetic_params['diversity_mode'] = diversity_mode
            
            # Store current parameters in session state
            current_params = {
                'day_type': day_type,
                'season': season,
                'max_power': max_power,
                'diversity_mode': diversity_mode
            }
            st.session_state.synthetic_params = current_params
            
            # Add button to generate synthetic load curve
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🎲 Generate Synthetic Curve", type="primary"):
                    try:
                        # Import the portable load generator
                        import sys
                        import os
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        
                        from portable_load_generator import generate_load_curve
                        
                        # Generate the load curve
                        result = generate_load_curve(
                            season=season,
                            day_type=day_type,
                            max_power=max_power,
                            diversity_mode=diversity_mode,
                            models_dir="portable_models",
                            return_timestamps=True
                        )
                        
                        # Store the generated data in session state
                        st.session_state.synthetic_load_curve = result['load_curve']
                        st.session_state.synthetic_timestamps = result['timestamps']
                        st.session_state.synthetic_metadata = result['metadata']
                        st.session_state.synthetic_params_hash = str(current_params)
                        
                        st.success(f"✅ Synthetic load curve generated successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error generating synthetic data: {e}")
                        st.write("Please ensure the portable_models directory contains the trained models.")
                        st.session_state.synthetic_load_curve = None
                        st.session_state.synthetic_metadata = None
            
            with col2:
                if st.button("🗑️ Clear Synthetic Curve"):
                    if 'synthetic_load_curve' in st.session_state:
                        del st.session_state.synthetic_load_curve
                    if 'synthetic_timestamps' in st.session_state:
                        del st.session_state.synthetic_timestamps
                    if 'synthetic_metadata' in st.session_state:
                        del st.session_state.synthetic_metadata
                    if 'synthetic_params_hash' in st.session_state:
                        del st.session_state.synthetic_params_hash
                    st.success("🗑️ Synthetic curve cleared!")
                    st.rerun()
            
            # Display synthetic curve summary
            st.write("**📊 Synthetic Curve Summary:**")
            if 'synthetic_load_curve' in st.session_state and st.session_state.synthetic_load_curve is not None:
                curve = st.session_state.synthetic_load_curve
                metadata = st.session_state.synthetic_metadata if 'synthetic_metadata' in st.session_state else {}
                
                st.success(f"✅ Synthetic load curve ready!")
                st.write(f"**Generated Profile:** {len(curve)} data points")
                st.write(f"**Mean Load:** {np.mean(curve):.2f} kW")
                st.write(f"**Max Load:** {np.max(curve):.2f} kW")
                st.write(f"**Min Load:** {np.min(curve):.2f} kW")
                if metadata:
                    st.write(f"**Season:** {metadata.get('season', 'Unknown')}")
                    st.write(f"**Day Type:** {metadata.get('day_type', 'Unknown')}")
                
                # Show curve preview
                fig, ax = plt.subplots(figsize=(10, 4))
                # Convert x-axis to hours (48 hours = 192 data points at 15-min intervals)
                time_hours = np.arange(len(curve)) * 15 / 60  # Convert 15-min intervals to hours
                ax.plot(time_hours, curve, linewidth=1, alpha=0.8)
                ax.set_title("Synthetic Load Curve Preview")
                ax.set_xlabel("Time (hours)")
                ax.set_ylabel("Load (kW)")
                ax.grid(True, alpha=0.3)
                # Set x-axis to show 0-48 hours
                ax.set_xlim(0, 48)
                st.pyplot(fig)
                plt.close()
            else:
                st.info("ℹ️ No synthetic curve generated yet. Click 'Generate Synthetic Curve' to create one.")
            
            # Set power_values from session state
            if 'synthetic_load_curve' in st.session_state and st.session_state.synthetic_load_curve is not None:
                power_values = st.session_state.synthetic_load_curve
            else:
                power_values = None
        
        # Universal Available Load Fraction slider (moved here for both data sources)
        st.write("**🔧 Grid Capacity Settings:**")
        # Add slider for available load fraction and store in session state
        if 'available_load_fraction' not in st.session_state:
            st.session_state['available_load_fraction'] = 0.8
        st.session_state['available_load_fraction'] = st.slider(
            "Available Load Fraction",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state['available_load_fraction'],
            step=0.05,
            help="Fraction of the grid limit that can be used (e.g. 0.8 = 80% safety margin)"
        )
        available_load_fraction = st.session_state['available_load_fraction']
        
        # Apply the available load fraction to power values if they exist
        if power_values is not None and data_source == "Real Dataset":
            power_values = power_values * available_load_fraction
            st.write(f"Grid power profile scaled by {available_load_fraction:.0%} safety margin")
        
        # Grid Constraint Mode
        st.write("**Grid Constraint Mode:**")
        grid_mode = st.radio("Mode", ["Reference Only", "Grid Constrained"], 
                            help="Reference Only: Show actual load with grid limit as reference line. Grid Constrained: Load cannot exceed grid limit.")

# Main content area

col1, col2 = st.columns([2, 1])

with col1:
    st.header("📈 Simulation Results")
    
    # Display current random seed
    st.write(f"🎲 **Current Random Seed:** {st.session_state.random_seed}")
    
    # Create two columns for the buttons
    button_col1, button_col2 = st.columns([1, 1])
    
    with button_col1:
        if st.button("🚀 Run Simulation"):
            # Clear any existing results first
            if 'simulation_results' in st.session_state:
                del st.session_state.simulation_results
    
    with button_col2:
        if st.button("🎲 Change Seed"):
            # Generate a new random seed
            import random
            old_seed = st.session_state.random_seed
            st.session_state.random_seed = random.randint(1, 10000)
            st.rerun()
        
        # Update session state with current values
        st.session_state.dynamic_ev = {
            'capacity': st.session_state.dynamic_ev['capacity'],
            'AC': st.session_state.dynamic_ev['AC']
        }
        st.session_state.charger_config = {
            'ac_rate': st.session_state.charger_config['ac_rate'],
            'ac_count': st.session_state.charger_config['ac_count']
        }
        
        # Set simulation run flag and just_run flag
        st.session_state.simulation_run = True
        st.session_state.simulation_just_run = True
        
        # Handle synthetic data generation if needed
        if data_source == "Synthetic Generation" and (power_values is None or len(power_values) == 0):
            st.info("🎲 Generating synthetic load curve based on current parameters...")
            try:
                # Import the portable load generator
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                from portable_load_generator import generate_load_curve
                
                # Get current synthetic parameters
                synthetic_params = st.session_state.get('synthetic_params', {})
                season = synthetic_params.get('season', 'winter')
                day_type = synthetic_params.get('day_type', 'weekday')
                max_power = synthetic_params.get('max_power', 400)
                diversity_mode = synthetic_params.get('diversity_mode', 'high')
                
                # Generate the load curve
                result = generate_load_curve(
                    season=season,
                    day_type=day_type,
                    max_power=max_power,
                    diversity_mode=diversity_mode,
                    models_dir="portable_models",
                    return_timestamps=True
                )
                
                # Store the generated data in session state
                st.session_state.synthetic_load_curve = result['load_curve']
                st.session_state.synthetic_timestamps = result['timestamps']
                st.session_state.synthetic_metadata = result['metadata']
                
                # Update power_values for simulation
                power_values = result['load_curve']
                st.success(f"✅ Synthetic load curve generated automatically for simulation!")
                
            except Exception as e:
                st.error(f"❌ Error generating synthetic data: {e}")
                st.write("Please ensure the portable_models directory contains the trained models.")
                st.stop()
        
        # Run simulation and store results
        if power_values is not None:
            # Create dynamic EV model
            dynamic_ev_model = {
                'name': 'Custom EV',
                'capacity': st.session_state.dynamic_ev['capacity'],
                'AC': st.session_state.dynamic_ev['AC']
            }
            
            # Temporarily update EV_MODELS with our dynamic EV
            original_ev_models = EV_MODELS.copy()
            EV_MODELS['dynamic_ev'] = dynamic_ev_model
            
            try:
                # Calculate total EVs from all peaks
                if 'time_peaks' in st.session_state and st.session_state.time_peaks:
                    total_evs = sum(peak.get('quantity', 0) for peak in st.session_state.time_peaks)
                    if total_evs == 0:
                        st.error("Please configure at least one peak with EVs")
                        st.stop()
                else:
                    st.error("Please configure at least one peak with EVs")
                    st.stop()
                
                # Prepare simulation parameters
                ev_counts = {'dynamic_ev': total_evs}
                charger_counts = {
                    'ac': st.session_state.charger_config['ac_count']
                }
                
                # Create custom charger models with user-defined rates
                custom_charger_models = {
                    'ac': {'type': 'AC', 'power_kW': st.session_state.charger_config['ac_rate']}
                }
                
                # Temporarily update CHARGER_MODELS with custom rates
                original_charger_models = CHARGER_MODELS.copy()
                CHARGER_MODELS.update(custom_charger_models)
                
                # Set grid power limit based on mode
                # Always create grid profile for plotting (FULL capacity, before 80% margin)
                grid_profile_15min = power_values
                grid_profile_full = np.repeat(grid_profile_15min, 15).astype(float)
                
                sim_duration_min = 72 * 60  # Run for 72 hours
                daily_minutes = 24 * 60
                
                # Check if we're using synthetic data
                using_synthetic_data = ('synthetic_load_curve' in st.session_state and 
                                      st.session_state.synthetic_load_curve is not None)
                
                if len(grid_profile_full) < sim_duration_min:
                    if using_synthetic_data:
                        # For synthetic data, repeat the generated profile to cover the simulation duration
                        num_days_needed = sim_duration_min // daily_minutes + (1 if sim_duration_min % daily_minutes > 0 else 0)
                        extended_profile = []
                        
                        for day in range(num_days_needed):
                            # Use the same synthetic profile for each day
                            extended_profile.extend(grid_profile_full)
                        
                        grid_profile_full = np.array(extended_profile[:sim_duration_min])
                    else:
                        # For real dataset, use actual next days from the dataset
                        # Make sure df is available for real dataset
                        if 'selected_dataset' in st.session_state and st.session_state.selected_dataset:
                            try:
                                df = pd.read_csv(f"datasets/{st.session_state.selected_dataset}")
                                df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                                
                                num_days_needed = sim_duration_min // daily_minutes + (1 if sim_duration_min % daily_minutes > 0 else 0)
                                extended_profile = []
                                current_date = st.session_state.get('selected_date', pd.to_datetime('2023-01-18').date())
                                
                                for day in range(num_days_needed):
                                    day_data = df[df['date'].dt.date == current_date]
                                    
                                    if not day_data.empty:
                                        # Use FULL capacity (no 80% margin yet)
                                        day_power_values = day_data.iloc[:, 2].values
                                        day_profile = np.repeat(day_power_values, 15).astype(float)
                                        extended_profile.extend(day_profile)
                                    else:
                                        # If no data for this day, repeat the previous day's full profile
                                        extended_profile.extend(grid_profile_full)
                                    
                                    current_date = current_date + pd.Timedelta(days=1)
                                
                                grid_profile_full = np.array(extended_profile[:sim_duration_min])
                            except Exception as e:
                                st.error(f"Error extending real dataset: {e}")
                                # Fallback to repeating the current profile
                                num_days_needed = sim_duration_min // daily_minutes + (1 if sim_duration_min % daily_minutes > 0 else 0)
                                extended_profile = []
                                for day in range(num_days_needed):
                                    extended_profile.extend(grid_profile_full)
                                grid_profile_full = np.array(extended_profile[:sim_duration_min])
                        else:
                            # Fallback to repeating the current profile
                            num_days_needed = sim_duration_min // daily_minutes + (1 if sim_duration_min % daily_minutes > 0 else 0)
                            extended_profile = []
                            for day in range(num_days_needed):
                                extended_profile.extend(grid_profile_full)
                            grid_profile_full = np.array(extended_profile[:sim_duration_min])
                else:
                    grid_profile_full = grid_profile_full[:sim_duration_min]
                
                # Note: Grid limit will be set after battery effects calculation
                
                # Pre-calculate all arrival times for optimized simulation
                arrival_times = []
                
                # Check if Time of Use is enabled
                time_of_use_enabled = ('smart_charging' in st.session_state.get('active_strategies', []) and 
                                      st.session_state.optimization_strategy.get('smart_charging_percent', 0) > 0)
                
                if time_of_use_enabled:
                    # Pre-calculate Time of Use peaks
                    time_of_use_periods = st.session_state.optimization_strategy.get('time_of_use_periods', [])
                    
                    # Fallback: if no periods in optimization_strategy, use timeline
                    if not time_of_use_periods and 'time_of_use_timeline' in st.session_state:
                        timeline = st.session_state.time_of_use_timeline
                        time_of_use_periods = convert_to_simulation_format(timeline['periods'])
                        st.session_state.optimization_strategy['time_of_use_periods'] = time_of_use_periods
                        st.info(f"🔧 Fallback: Converted {len(time_of_use_periods)} TOU periods from timeline")
                    
                    # Periods should already be in simulation format from the TOU configuration
                    if time_of_use_periods:
                        # Verify conversion worked properly
                        for period in time_of_use_periods:
                            if 'start' not in period or 'end' not in period:
                                st.error(f"❌ TOU period conversion failed for period: {period}")
                                st.stop()
                            
                            # Validate period duration
                            duration = period['end'] - period['start']
                            if duration <= 0 or duration > 24:
                                st.error(f"❌ Invalid TOU period duration: {duration} hours for period {period['name']} ({period['start']}-{period['end']})")
                                st.stop()
                        
                        # Debug: Show the periods being used
                        period_info = [(p['name'], f"{p['start']}-{p['end']}h", f"{p['adoption']}%") for p in time_of_use_periods]
                        
                        
                      
                        
                        
                        
                        # Group periods by name for proper car distribution
                        period_groups = {}
                        for period in time_of_use_periods:
                            period_name = period['name']
                            if period_name not in period_groups:
                                period_groups[period_name] = []
                            period_groups[period_name].append(period)
                        
                        # Calculate how many days the simulation runs (48 hours = 2 days for car addition)
                        daily_minutes = 24 * 60
                        num_days = 2  # Run for 2 days to cover 48 hours (cars only added for 48 hours)
                        
                        # Pre-calculate all arrival times for Time of Use (repeating daily for 48 hours only)
                        for day in range(num_days):
                            day_offset_minutes = day * daily_minutes
                            
                            for period_name, periods in period_groups.items():
                                if periods:
                                    # Get adoption for this period type (should be the same for all periods of same type)
                                    period_adoption = periods[0]['adoption']
                                    
                                    if period_adoption > 0:
                                        # Calculate total EVs for this period type
                                        total_evs_for_type = int(total_evs * period_adoption / 100)
                                        
                                        if total_evs_for_type > 0:
                                            # Calculate total hours for this period type
                                            total_hours_for_type = sum(p['end'] - p['start'] for p in periods)
                                            
                                            # Split EVs proportionally among periods of this type
                                            for period in periods:
                                                # Calculate proportion: this period's hours / total hours of this type
                                                this_period_hours = period['end'] - period['start']
                                                proportion = this_period_hours / total_hours_for_type
                                                
                                                # Calculate EVs for this specific period
                                                period_evs = int(total_evs_for_type * proportion)
                                                
                                                if period_evs > 0:
                                                    # Calculate center time (middle of the period)
                                                    center_hour = (period['start'] + period['end']) / 2
                                                    center_minutes = center_hour * 60
                                                    
                                                    # Calculate span (duration of the period)
                                                    span_hours = period['end'] - period['start']
                                                    span_minutes = span_hours * 60
                                                    std_minutes = span_minutes / 2  # Standard deviation = span/4
                                                    
                                                    # Calculate charging duration to shift arrival times
                                                    ev_capacity = st.session_state.dynamic_ev.get('capacity', 75)
                                                    ev_charging_rate = st.session_state.dynamic_ev.get('AC', 11)
                                                    charging_duration_hours = ev_capacity / ev_charging_rate if ev_charging_rate > 0 else 8
                                                    charging_duration_minutes = charging_duration_hours * 60
                                                    
                                                    # Shift center by 0.4x the charging duration (so EVs finish charging at center of period)
                                                    shift_amount = charging_duration_minutes * 0.5
                                                    shifted_center_minutes = center_minutes - shift_amount
                                                    
                                                    # Generate arrival times for this period on this day
                                                    period_arrival_times = np.random.normal(shifted_center_minutes, std_minutes, period_evs)
                                                    
                                                    # Add day offset first
                                                    period_arrival_times += day_offset_minutes
                                                    
                                                    # Handle negative times by wrapping to 48-hour simulation boundary
                                                    for i in range(len(period_arrival_times)):
                                                        if period_arrival_times[i] < 0:
                                                            # Wrap negative times to 48 - |negative_time| (same as single peak)
                                                            period_arrival_times[i] = 48 * 60 + period_arrival_times[i]
                                                    
                                                    # Add to arrival times list
                                                    arrival_times.extend(period_arrival_times)
                else:
                    # Use multiple peaks if Time of Use is not enabled
                    for day in range(2):  # 2 days like TOU
                        day_offset_minutes = day * 24 * 60
                        
                        # Generate cars for each peak
                        for peak in st.session_state.time_peaks:
                            peak_quantity = peak.get('quantity', 0)
                            if peak_quantity > 0:
                                peak_mean = peak['time'] * 60
                                peak_span = peak['span'] * 60
                                sigma = peak_span
                                
                                # Calculate charging duration to shift arrival times
                                ev_capacity = st.session_state.dynamic_ev.get('capacity', 75)
                                ev_charging_rate = st.session_state.dynamic_ev.get('AC', 11)
                                charging_duration_hours = ev_capacity / ev_charging_rate if ev_charging_rate > 0 else 8
                                charging_duration_minutes = charging_duration_hours * 60
                                
                                # Shift peak by 0.4x the charging duration (so EVs finish charging at peak time)
                                shift_amount = charging_duration_minutes * 0.4
                                shifted_peak_mean = peak_mean - shift_amount
                                
                                peak_arrivals = np.random.normal(shifted_peak_mean, sigma, peak_quantity)
                                
                                # Handle negative times by wrapping to 48 - |negative_time|
                                for i in range(len(peak_arrivals)):
                                    if peak_arrivals[i] < 0:
                                        # Wrap negative times to 48 - |negative_time|
                                        peak_arrivals[i] = 48 * 60 + peak_arrivals[i]
                                
                                # Add day offset
                                peak_arrivals += day_offset_minutes
                                arrival_times.extend(peak_arrivals)
                
                # Finalize arrival times (sort once)
                arrival_times = np.array(arrival_times)
                
                # Handle clipping differently for TOU vs single peak
                if time_of_use_enabled:
                    # For TOU, allow times beyond 48 hours since we're simulating multiple days
                    # Only clip negative times (which should have been handled already, but just in case)
                    arrival_times = np.clip(arrival_times, 0, None)
                else:
                    # For single peak, clip to 48 hours as before
                    arrival_times = np.clip(arrival_times, 0, 48 * 60)  # Run for 48 hours, cars can spawn up to 48 hours
                
                arrival_times.sort()
                
                # Add V2G recharge EVs to arrival times
                if 'v2g' in st.session_state.get('active_strategies', []):
                    v2g_adoption_percent = st.session_state.optimization_strategy.get('v2g_adoption_percent', 0)
                    v2g_recharge_arrival_hour = st.session_state.optimization_strategy.get('v2g_recharge_arrival_hour', 26)
                    
                    if v2g_adoption_percent > 0:
                        total_v2g_evs = int(total_evs * v2g_adoption_percent / 100)
                        
                        # Calculate realistic discharge and recharge EVs
                        ev_capacity = st.session_state.dynamic_ev.get('capacity', 75)
                        v2g_discharge_duration = st.session_state.optimization_strategy.get('v2g_discharge_duration', 3)
                        v2g_max_discharge_rate = st.session_state.optimization_strategy.get('v2g_max_discharge_rate', 7)
                        full_discharge_time = ev_capacity / v2g_max_discharge_rate if v2g_max_discharge_rate > 0 else 1.0
                        
                        if v2g_discharge_duration >= full_discharge_time:
                            # Fully discharged - all EVs arrive to recharge
                            recharge_evs = total_v2g_evs
                        else:
                            # Partially discharged - only some EVs arrive to recharge
                            discharge_percentage = v2g_discharge_duration / full_discharge_time
                            recharge_evs = int(total_v2g_evs * discharge_percentage)
                        
                        # V2G recharge EVs arrive the next day (v2g_recharge_arrival_hour already represents next day hours)
                        v2g_recharge_arrival_minute = v2g_recharge_arrival_hour * 60
                        
                        # Generate V2G recharge arrival times (spread over 1 hour)
                        v2g_arrival_times = np.random.normal(v2g_recharge_arrival_minute, 30, recharge_evs)
                        v2g_arrival_times = np.clip(v2g_arrival_times, v2g_recharge_arrival_minute - 30, v2g_recharge_arrival_minute + 30)
                        
                        # Add V2G recharge EVs to arrival times
                        arrival_times = np.concatenate([arrival_times, v2g_arrival_times])
                        arrival_times.sort()
                
                # Step 2: Create adjusted grid profile with battery effects BEFORE simulation
                adjusted_grid_profile = grid_profile_full.copy()
                
                # Set random seed for consistent battery effects between TOU optimizer and main simulation
                np.random.seed(st.session_state.random_seed)
                
                # Apply PV + Battery optimization if enabled (charging during day, discharging during evening)
                pv_battery_support_adjusted = np.zeros_like(grid_profile_full)
                pv_battery_charge_curve = np.zeros_like(grid_profile_full)
                pv_direct_support_curve = np.zeros_like(grid_profile_full)
                pv_battery_discharge_curve = np.zeros_like(grid_profile_full)
                if 'pv_battery' in st.session_state.get('active_strategies', []):
                    pv_adoption_percent = st.session_state.optimization_strategy.get('pv_adoption_percent', 0)
                    battery_capacity = st.session_state.optimization_strategy.get('battery_capacity', 0)
                    max_charge_rate = st.session_state.optimization_strategy.get('max_charge_rate', 0)
                    max_discharge_rate = st.session_state.optimization_strategy.get('max_discharge_rate', 0)
                    solar_energy_percent = st.session_state.optimization_strategy.get('solar_energy_percent', 70)
                    pv_start_hour = st.session_state.optimization_strategy.get('pv_start_hour', 8)
                    pv_duration = st.session_state.optimization_strategy.get('pv_duration', 8)
                    charge_time = st.session_state.optimization_strategy.get('charge_time', battery_capacity / max_charge_rate)
                    system_support_time = st.session_state.optimization_strategy.get('system_support_time', pv_duration - charge_time)
                    discharge_start_hour = st.session_state.optimization_strategy.get('discharge_start_hour', 18)
                    discharge_duration = st.session_state.optimization_strategy.get('discharge_duration', battery_capacity / max_discharge_rate)
                    actual_discharge_rate = st.session_state.optimization_strategy.get('actual_discharge_rate', max_discharge_rate)
                    
                    if pv_adoption_percent > 0 and battery_capacity > 0 and max_charge_rate > 0 and actual_discharge_rate > 0:
                        total_evs_for_pv = total_evs
                        pv_evs = int(total_evs_for_pv * pv_adoption_percent / 100)
                        total_charge_power = pv_evs * max_charge_rate
                        total_system_support_power = pv_evs * max_charge_rate  # Same as charge rate for system support
                        total_discharge_power = pv_evs * actual_discharge_rate * (solar_energy_percent / 100)
                        
                        # Generate variable start times with normal distribution or strict boundaries
                        pv_use_normal_distribution = st.session_state.optimization_strategy.get('pv_use_normal_distribution', True)
                        pv_sigma_divisor = st.session_state.optimization_strategy.get('pv_sigma_divisor', 8)
                        
                        if pv_use_normal_distribution and pv_sigma_divisor:
                            pv_sigma = pv_duration / pv_sigma_divisor
                            pv_start_times = np.random.normal(pv_start_hour, pv_sigma, pv_evs)
                            pv_start_times = np.clip(pv_start_times, 0, 24)  # Clip to valid hours
                        else:
                            # Use strict boundaries - all EVs start at the same time
                            pv_start_times = np.full(pv_evs, pv_start_hour)
                        
                        if pv_use_normal_distribution and pv_sigma_divisor:
                            discharge_sigma = discharge_duration / pv_sigma_divisor
                            discharge_start_times = np.random.normal(discharge_start_hour, discharge_sigma, pv_evs)
                            discharge_start_times = np.clip(discharge_start_times, 0, 24)  # Clip to valid hours
                        else:
                            # Use strict boundaries - all EVs start at the same time
                            discharge_start_times = np.full(pv_evs, discharge_start_hour)
                        
                        # Calculate time periods for each EV
                        for ev_idx in range(pv_evs):
                            # Get individual start times
                            ev_pv_start = pv_start_times[ev_idx]
                            ev_discharge_start = discharge_start_times[ev_idx]
                            
                            # Convert to minutes
                            ev_pv_start_minute = int(ev_pv_start * 60)
                            ev_system_support_end_minute = ev_pv_start_minute + int(pv_duration * 60)
                            ev_discharge_start_minute = int(ev_discharge_start * 60)
                            ev_discharge_duration_minutes = int(discharge_duration * 60)
                            ev_discharge_end_minute = ev_discharge_start_minute + ev_discharge_duration_minutes
                            
                            # Calculate required charging rate for this EV
                            ev_battery_capacity = battery_capacity
                            ev_charging_time_hours = pv_duration
                            ev_required_charging_rate = ev_battery_capacity / ev_charging_time_hours
                            
                            for minute in range(len(grid_profile_full)):
                                time_of_day = minute % (24 * 60)
                                
                                # Day Phase: Simultaneous battery charging and grid support for this EV
                                if ev_system_support_end_minute > 24 * 60:
                                    # System support period extends beyond 24 hours, use absolute time
                                    if (minute >= ev_pv_start_minute and minute < ev_system_support_end_minute):
                                        # Calculate total PV power available for this EV
                                        ev_total_pv_power = total_system_support_power / pv_evs  # Divide by number of EVs
                                        
                                        # Calculate remaining power for grid support
                                        ev_grid_support_power = max(0, ev_total_pv_power - ev_required_charging_rate)
                                        
                                        # Apply battery charging (no grid effect - PV charges batteries directly)
                                        pv_battery_charge_curve[minute] += ev_required_charging_rate
                                        
                                        # Apply grid support (increases available capacity)
                                        if ev_grid_support_power > 0:
                                            pv_direct_support_curve[minute] += ev_grid_support_power
                                            adjusted_grid_profile[minute] += ev_grid_support_power  # Increase available capacity
                                else:
                                    # System support period is within same day, use modulo logic
                                    time_of_day = minute % (24 * 60)
                                    if (time_of_day >= ev_pv_start_minute and time_of_day < ev_system_support_end_minute):
                                        # Calculate total PV power available for this EV
                                        ev_total_pv_power = total_system_support_power / pv_evs  # Divide by number of EVs
                                        
                                        # Calculate remaining power for grid support
                                        ev_grid_support_power = max(0, ev_total_pv_power - ev_required_charging_rate)
                                        
                                        # Apply battery charging (no grid effect - PV charges batteries directly)
                                        pv_battery_charge_curve[minute] += ev_required_charging_rate
                                        
                                        # Apply grid support (increases available capacity)
                                        if ev_grid_support_power > 0:
                                            pv_direct_support_curve[minute] += ev_grid_support_power
                                            adjusted_grid_profile[minute] += ev_grid_support_power  # Increase available capacity
                
                                # Evening Phase: Battery discharge for this EV
                                if ev_discharge_end_minute > 24 * 60:
                                    # Discharging period extends beyond 24 hours, use absolute time
                                    if (minute >= ev_discharge_start_minute and minute < ev_discharge_end_minute):
                                        ev_discharge_power = total_discharge_power / pv_evs  # Divide by number of EVs
                                        pv_battery_discharge_curve[minute] += ev_discharge_power
                                        adjusted_grid_profile[minute] += ev_discharge_power  # Increase available capacity
                                else:
                                    # Discharging period is within same day, use modulo logic
                                    time_of_day = minute % (24 * 60)
                                    if (time_of_day >= ev_discharge_start_minute and time_of_day < ev_discharge_end_minute):
                                        ev_discharge_power = total_discharge_power / pv_evs  # Divide by number of EVs
                                        pv_battery_discharge_curve[minute] += ev_discharge_power
                                        adjusted_grid_profile[minute] += ev_discharge_power  # Increase available capacity
                # Apply Grid-Charged Batteries optimization if enabled
                grid_battery_charge_curve = np.zeros_like(grid_profile_full)
                grid_battery_discharge_curve = np.zeros_like(grid_profile_full)
                if 'grid_battery' in st.session_state.get('active_strategies', []):
                    grid_battery_adoption_percent = st.session_state.optimization_strategy.get('grid_battery_adoption_percent', 0)
                    grid_battery_capacity = st.session_state.optimization_strategy.get('grid_battery_capacity', 0)
                    grid_battery_max_rate = st.session_state.optimization_strategy.get('grid_battery_max_rate', 0)
                    grid_battery_charge_start_hour = st.session_state.optimization_strategy.get('grid_battery_charge_start_hour', 7)
                    grid_battery_charge_duration = st.session_state.optimization_strategy.get('grid_battery_charge_duration', 8)
                    grid_battery_discharge_start_hour = st.session_state.optimization_strategy.get('grid_battery_discharge_start_hour', 18)
                    grid_battery_discharge_duration = st.session_state.optimization_strategy.get('grid_battery_discharge_duration', 4)
                    
                    if grid_battery_adoption_percent > 0 and grid_battery_capacity > 0 and grid_battery_max_rate > 0:
                        total_evs_for_grid_battery = total_evs
                        grid_battery_evs = int(total_evs_for_grid_battery * grid_battery_adoption_percent / 100)
                        
                        # Calculate required charging rate based on battery capacity and duration (like PV batteries)
                        ev_required_charging_rate = grid_battery_capacity / grid_battery_charge_duration
                        ev_required_discharge_rate = grid_battery_capacity / grid_battery_discharge_duration
                        total_charge_power = grid_battery_evs * ev_required_charging_rate
                        total_discharge_power = grid_battery_evs * ev_required_discharge_rate
                        
                        # Generate variable start times with normal distribution or strict boundaries
                        grid_use_normal_distribution = st.session_state.optimization_strategy.get('grid_use_normal_distribution', True)
                        grid_sigma_divisor = st.session_state.optimization_strategy.get('grid_sigma_divisor', 8)
                        
                        if grid_use_normal_distribution and grid_sigma_divisor:
                            charge_sigma = grid_battery_charge_duration / grid_sigma_divisor
                            charge_start_times = np.random.normal(grid_battery_charge_start_hour, charge_sigma, grid_battery_evs)
                            charge_start_times = np.clip(charge_start_times, 0, 24)  # Clip to valid hours
                        else:
                            # Use strict boundaries - all EVs start at the same time
                            charge_start_times = np.full(grid_battery_evs, grid_battery_charge_start_hour)
                        
                        if grid_use_normal_distribution and grid_sigma_divisor:
                            discharge_sigma = grid_battery_discharge_duration / grid_sigma_divisor
                            discharge_start_times = np.random.normal(grid_battery_discharge_start_hour, discharge_sigma, grid_battery_evs)
                            discharge_start_times = np.clip(discharge_start_times, 0, 24)  # Clip to valid hours
                        else:
                            # Use strict boundaries - all EVs start at the same time
                            discharge_start_times = np.full(grid_battery_evs, grid_battery_discharge_start_hour)
                        
                        # Calculate time periods for each EV
                        for ev_idx in range(grid_battery_evs):
                            # Get individual start times
                            ev_charge_start = charge_start_times[ev_idx]
                            ev_discharge_start = discharge_start_times[ev_idx]
                            
                            # Convert to minutes
                            ev_charge_start_minute = int(ev_charge_start * 60)
                            ev_charge_duration_minutes = int(grid_battery_charge_duration * 60)
                            ev_charge_end_minute = ev_charge_start_minute + ev_charge_duration_minutes
                            ev_discharge_start_minute = int(ev_discharge_start * 60)
                            ev_discharge_duration_minutes = int(grid_battery_discharge_duration * 60)
                            ev_discharge_end_minute = ev_discharge_start_minute + ev_discharge_duration_minutes
                            
                            # Calculate power for this EV
                            ev_charge_power = total_charge_power / grid_battery_evs
                            ev_discharge_power = total_discharge_power / grid_battery_evs
                            
                            for minute in range(len(grid_profile_full)):
                                # Handle charging curve (reduces available capacity)
                                if ev_charge_end_minute > 24 * 60:
                                    # Charging period extends beyond 24 hours, use absolute time
                                    if (minute >= ev_charge_start_minute and minute < ev_charge_end_minute):
                                        grid_battery_charge_curve[minute] += ev_charge_power
                                        adjusted_grid_profile[minute] -= ev_charge_power  # Reduce available capacity
                                else:
                                    # Charging period is within same day, use modulo logic
                                    time_of_day = minute % (24 * 60)
                                    if (time_of_day >= ev_charge_start_minute and time_of_day < ev_charge_end_minute):
                                        grid_battery_charge_curve[minute] += ev_charge_power
                                        adjusted_grid_profile[minute] -= ev_charge_power  # Reduce available capacity
                                
                                # Handle discharging curve (increases available capacity)
                                if ev_discharge_end_minute > 24 * 60:
                                    # Discharging period extends beyond 24 hours, use absolute time
                                    if (minute >= ev_discharge_start_minute and minute < ev_discharge_end_minute):
                                        grid_battery_discharge_curve[minute] += ev_discharge_power
                                        adjusted_grid_profile[minute] += ev_discharge_power  # Increase available capacity
                                else:
                                    # Discharging period is within same day, use modulo logic
                                    time_of_day = minute % (24 * 60)
                                    if (time_of_day >= ev_discharge_start_minute and time_of_day < ev_discharge_end_minute):
                                        grid_battery_discharge_curve[minute] += ev_discharge_power
                                        adjusted_grid_profile[minute] += ev_discharge_power  # Increase available capacity
                
                # Apply V2G (Vehicle-to-Grid) optimization if enabled
                v2g_discharge_curve = np.zeros_like(grid_profile_full)
                if 'v2g' in st.session_state.get('active_strategies', []):
                    v2g_adoption_percent = st.session_state.optimization_strategy.get('v2g_adoption_percent', 0)
                    v2g_max_discharge_rate = st.session_state.optimization_strategy.get('v2g_max_discharge_rate', 0)
                    v2g_discharge_start_hour = st.session_state.optimization_strategy.get('v2g_discharge_start_hour', 18)
                    v2g_discharge_duration = st.session_state.optimization_strategy.get('v2g_discharge_duration', 2)
                    
                    if v2g_adoption_percent > 0 and v2g_max_discharge_rate > 0:
                        total_v2g_evs = int(total_evs * v2g_adoption_percent / 100)
                        total_v2g_discharge_power = total_v2g_evs * v2g_max_discharge_rate
                        
                        # Generate variable start times with normal distribution or strict boundaries
                        v2g_use_normal_distribution = st.session_state.optimization_strategy.get('v2g_use_normal_distribution', True)
                        v2g_sigma_divisor = st.session_state.optimization_strategy.get('v2g_sigma_divisor', 8)
                        
                        if v2g_use_normal_distribution and v2g_sigma_divisor:
                            v2g_sigma = v2g_discharge_duration / v2g_sigma_divisor
                            v2g_start_times = np.random.normal(v2g_discharge_start_hour, v2g_sigma, total_v2g_evs)
                            v2g_start_times = np.clip(v2g_start_times, 0, 24)  # Clip to valid hours
                        else:
                            # Use strict boundaries - all EVs start at the same time
                            v2g_start_times = np.full(total_v2g_evs, v2g_discharge_start_hour)
                        
                        # Calculate time periods for each EV
                        for ev_idx in range(total_v2g_evs):
                            # Get individual start time
                            ev_v2g_start = v2g_start_times[ev_idx]
                            
                            # Convert to minutes
                            ev_v2g_start_minute = int(ev_v2g_start * 60)
                            ev_v2g_duration_minutes = int(v2g_discharge_duration * 60)
                            ev_v2g_end_minute = ev_v2g_start_minute + ev_v2g_duration_minutes
                            
                            # Calculate power for this EV
                            ev_v2g_power = total_v2g_discharge_power / total_v2g_evs
                            
                            for minute in range(len(grid_profile_full)):
                                # Use absolute time comparison for periods that cross midnight
                                if ev_v2g_end_minute > 24 * 60:
                                    # Period extends beyond 24 hours, use absolute time
                                    if (minute >= ev_v2g_start_minute and minute < ev_v2g_end_minute):
                                        v2g_discharge_curve[minute] += ev_v2g_power
                                        adjusted_grid_profile[minute] += ev_v2g_power  # Increase available capacity
                                else:
                                    # Period is within same day, use modulo logic
                                    time_of_day = minute % (24 * 60)
                                    if (time_of_day >= ev_v2g_start_minute and time_of_day < ev_v2g_end_minute):
                                        v2g_discharge_curve[minute] += ev_v2g_power
                                        adjusted_grid_profile[minute] += ev_v2g_power  # Increase available capacity

                # Step 2: Apply 80% margin to the adjusted grid profile
                final_grid_profile = adjusted_grid_profile * st.session_state['available_load_fraction']

                # Step 3: Run simulation with the correct grid limit (battery effects + margin)
                # Set grid power limit for simulation
                if grid_mode == "Grid Constrained":
                    # Pass the final grid profile (with battery effects + margin) to the simulation
                    grid_power_limit = final_grid_profile
                else:
                    grid_power_limit = None  # No constraint for Reference Only mode
                
                # Run simulation for 60 hours
                sim = SimulationSetup(
                    ev_counts=ev_counts,
                    charger_counts=charger_counts,
                    sim_duration=60 * 60,  # Run for 60 hours
                    arrival_time_mean=12 * 60,
                    arrival_time_span=4 * 60,
                    grid_power_limit=grid_power_limit,  # Pass grid constraint to simulation
                    verbose=False  # Disable verbose output for faster simulation
                )
                sim.evs = []
                
                # Calculate how many V2G recharge EVs we have
                v2g_recharge_count = 0
                if 'v2g' in st.session_state.get('active_strategies', []):
                    v2g_adoption_percent = st.session_state.optimization_strategy.get('v2g_adoption_percent', 0)
                    if v2g_adoption_percent > 0:
                        v2g_recharge_count = int(total_evs * v2g_adoption_percent / 100)
                
                # Calculate how many PV battery EVs we have
                pv_battery_count = 0
                if 'pv_battery' in st.session_state.get('active_strategies', []):
                    pv_adoption_percent = st.session_state.optimization_strategy.get('pv_adoption_percent', 0)
                    if pv_adoption_percent > 0:
                        pv_battery_count = int(total_evs * pv_adoption_percent / 100)
                
                for i, arrival_time in enumerate(arrival_times):
                    ev_name = f"Custom_EV_{i+1}"
                    
                    # Determine EV types
                    is_v2g_recharge = i >= (len(arrival_times) - v2g_recharge_count)
                    is_pv_battery = i >= (len(arrival_times) - v2g_recharge_count - pv_battery_count) and i < (len(arrival_times) - v2g_recharge_count)
                    
                    # Set SOC based on EV type
                    if is_v2g_recharge:
                        # V2G EVs have lower SOC since they discharged earlier
                        # Calculate realistic discharge based on battery capacity constraints
                        v2g_discharge_duration = st.session_state.optimization_strategy.get('v2g_discharge_duration', 3)
                        v2g_max_discharge_rate = st.session_state.optimization_strategy.get('v2g_max_discharge_rate', 7)
                        ev_capacity = st.session_state.dynamic_ev.get('capacity', 75)
                        full_discharge_time = ev_capacity / v2g_max_discharge_rate if v2g_max_discharge_rate > 0 else 1.0
                        
                        if v2g_discharge_duration >= full_discharge_time:
                            # Fully discharged - use slower rate if needed
                            actual_discharge_rate = ev_capacity / v2g_discharge_duration
                            total_discharged_kwh = ev_capacity  # Fully discharged
                        else:
                            # Partially discharged
                            actual_discharge_rate = v2g_max_discharge_rate
                            total_discharged_kwh = v2g_discharge_duration * v2g_max_discharge_rate
                        
                        # Calculate SOC after discharge (assuming they started at 80% SOC)
                        initial_soc = 0.8
                        discharged_soc = total_discharged_kwh / ev_capacity
                        soc = max(0.1, initial_soc - discharged_soc)  # Minimum 10% SOC
                    elif is_pv_battery:
                        # PV battery EVs start with higher SOC since they charge during the day
                        soc = st.session_state.get('ev_soc', 0.2) + 0.3  # 30% higher SOC due to daytime charging
                        soc = min(0.9, soc)  # Cap at 90% SOC
                    else:
                        # Regular EVs start at configured SOC
                        soc = st.session_state.get('ev_soc', 0.2)
                    
                    ev = EV(name=ev_name, battery_capacity=ev_capacity,
                           max_charging_power=ev_ac, soc=soc)
                    ev.ev_type = 'dynamic_ev'
                    ev.is_pv_battery = is_pv_battery  # Mark PV battery EVs
                    sim.evs.append(ev)
                for i, ev in enumerate(sim.evs):
                    sim.env.process(sim._ev_arrival(ev, arrival_times[i]))
                sim.env.run(until=72 * 60)  # Run for 72 hours
                constrained_load_curve = sim.load_curve.copy()
                
                
                # Sum the load curve for 48-hour plotting (0-48h + 48-72h elongated and summed)
                if len(constrained_load_curve) > 48 * 60:
                    
                    # Split load curve at 48 hours
                    pre_48h_curve = constrained_load_curve[:48 * 60]
                    post_48h_curve = constrained_load_curve[48 * 60:]
                    
                    
                    
                    # Elongate post-48h curve to match pre-48h size by filling with zeros
                    if len(post_48h_curve) < len(pre_48h_curve):
                        zeros_needed = len(pre_48h_curve) - len(post_48h_curve)
                        post_48h_curve = np.concatenate([post_48h_curve, np.zeros(zeros_needed)])
                        
                    
                    # Sum the two curves for final result
                    constrained_load_curve = pre_48h_curve + post_48h_curve
                    
                
                # Store the original EV-only load curve for display purposes
                original_ev_only_load = constrained_load_curve.copy()
                
                # Restore original EV_MODELS
                EV_MODELS.clear()
                EV_MODELS.update(original_ev_models)

                # The simulation already uses the grid constraint, so we don't need to apply it again
                # constrained_load_curve already contains the simulation results with grid constraints applied

                # Step 4: Store results
                # Calculate queue times manually
                queue_times = [ev.queue_time for ev in sim.evs if hasattr(ev, 'queue_time') and ev.queue_time > 0]
                avg_queue_time = np.mean(queue_times) if queue_times else 0
                num_queued_evs = len(queue_times)
                
                st.session_state.simulation_results = {
                    'load_curve': constrained_load_curve,
                    'original_load_curve': constrained_load_curve,  # This is the simulation result
                    'original_ev_only_load': original_ev_only_load,
                    'pv_battery_support': pv_battery_support_adjusted if 'pv_battery' in st.session_state.get('active_strategies', []) else None,
                    'pv_battery_charge_curve': pv_battery_charge_curve if 'pv_battery' in st.session_state.get('active_strategies', []) else None,
                    'pv_direct_support_curve': pv_direct_support_curve if 'pv_battery' in st.session_state.get('active_strategies', []) else None,
                    'pv_battery_discharge_curve': pv_battery_discharge_curve if 'pv_battery' in st.session_state.get('active_strategies', []) else None,
                    'grid_battery_charge_curve': grid_battery_charge_curve if 'grid_battery' in st.session_state.get('active_strategies', []) else None,
                    'grid_battery_discharge_curve': grid_battery_discharge_curve if 'grid_battery' in st.session_state.get('active_strategies', []) else None,
                    'v2g_discharge_curve': v2g_discharge_curve if 'v2g' in st.session_state.get('active_strategies', []) else None,
                    'grid_limit': final_grid_profile,  # This is the adjusted grid profile WITH 80% margin
                    'original_grid_limit': grid_profile_full,  # This is the full capacity BEFORE 80% margin
                    'adjusted_grid_limit_before_margin': adjusted_grid_profile,  # This is adjusted capacity BEFORE 80% margin
                    'grid_mode': grid_mode,
                    'total_evs': total_evs,
                    'sim_duration': sim_duration,
                    'avg_queue_time': avg_queue_time,
                    'num_queued_evs': num_queued_evs,
                    'rejected_evs': getattr(sim, 'rejected_evs', []),
                    'smart_charging_applied': 'smart_charging' in st.session_state.get('active_strategies', []),
                    'smart_charging_percent': st.session_state.optimization_strategy.get('smart_charging_percent', 0),
                    'shift_start_time': st.session_state.optimization_strategy.get('shift_start_time', 23),
                    'pv_battery_applied': 'pv_battery' in st.session_state.get('active_strategies', []),
                    'pv_adoption_percent': st.session_state.optimization_strategy.get('pv_adoption_percent', 0),
                    'battery_capacity': st.session_state.optimization_strategy.get('battery_capacity', 0),
                    'max_charge_rate': st.session_state.optimization_strategy.get('max_charge_rate', 0),
                    'max_discharge_rate': st.session_state.optimization_strategy.get('max_discharge_rate', 0),
                    'pv_start_hour': st.session_state.optimization_strategy.get('pv_start_hour', 8),
                    'pv_duration': st.session_state.optimization_strategy.get('pv_duration', 8),
                    'charge_time': st.session_state.optimization_strategy.get('charge_time', 0),
                    'system_support_time': st.session_state.optimization_strategy.get('system_support_time', 0),
                    'discharge_start_hour': st.session_state.optimization_strategy.get('discharge_start_hour', 18),
                    'solar_energy_percent': st.session_state.optimization_strategy.get('solar_energy_percent', 70),
                    'grid_battery_applied': 'grid_battery' in st.session_state.get('active_strategies', []),
                    'grid_battery_adoption_percent': st.session_state.optimization_strategy.get('grid_battery_adoption_percent', 0),
                    'grid_battery_capacity': st.session_state.optimization_strategy.get('grid_battery_capacity', 0),
                    'grid_battery_max_rate': st.session_state.optimization_strategy.get('grid_battery_max_rate', 0),
                    'grid_battery_charge_start_hour': st.session_state.optimization_strategy.get('grid_battery_charge_start_hour', 7),
                    'grid_battery_charge_duration': st.session_state.optimization_strategy.get('grid_battery_charge_duration', 8),
                    'grid_battery_discharge_start_hour': st.session_state.optimization_strategy.get('grid_battery_discharge_start_hour', 18),
                    'grid_battery_discharge_duration': st.session_state.optimization_strategy.get('grid_battery_discharge_duration', 4),
                    'v2g_applied': 'v2g' in st.session_state.get('active_strategies', []),
                    'v2g_adoption_percent': st.session_state.optimization_strategy.get('v2g_adoption_percent', 0),
                    'v2g_max_discharge_rate': st.session_state.optimization_strategy.get('v2g_max_discharge_rate', 0),
                    'v2g_discharge_start_hour': st.session_state.optimization_strategy.get('v2g_discharge_start_hour', 18),
                    'v2g_discharge_duration': st.session_state.optimization_strategy.get('v2g_discharge_duration', 3),
                    'v2g_recharge_arrival_hour': st.session_state.optimization_strategy.get('v2g_recharge_arrival_hour', 26),
                    # Synthetic data information
                    'using_synthetic_data': using_synthetic_data,
                    'synthetic_metadata': st.session_state.get('synthetic_metadata', None)
                }
                
            except Exception as e:
                st.error(f"Simulation error: {e}")
            finally:
                # Restore original models
                EV_MODELS.clear()
                EV_MODELS.update(original_ev_models)
                CHARGER_MODELS.clear()
                CHARGER_MODELS.update(original_charger_models)
    
    # Display simulation results if available
    if ('simulation_results' in st.session_state and st.session_state.simulation_results is not None):
        results = st.session_state.simulation_results
        
        st.success("Simulation completed successfully!")
        
        # Graph Control Panel
        st.subheader("🎛️ Graph Controls")
        col_controls1, col_controls2, col_controls3 = st.columns(3)
        
        with col_controls1:
            show_battery_effects = st.checkbox("Show Battery Effects", value=True, key="show_battery_effects",
                                             help="Display PV battery, grid battery, and V2G effects on the graph")
            show_average_line = st.checkbox("Show Average Line", value=True, key="show_average_line",
                                          help="Display horizontal average line on the bottom graph")
        
        with col_controls2:
            show_legend = st.checkbox("Show Legend", value=True, key="show_legend",
                                    help="Display graph legend")
            smooth_graph = st.checkbox("Smooth Graph", value=False, key="smooth_graph",
                                     help="Use smooth lines instead of stepped lines for the graph")
        
        with col_controls3:
            show_safety_margin = st.checkbox("Show Safety Margin", value=True, key="show_safety_margin",
                                           help="Display 20% safety margin line")
        
        
    
        
        # Plot results with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[2, 1])
        
        # Define time axis - show 24-hour periods instead of continuous hours
        sim_duration = st.session_state.get('sim_duration', 36)  # Get from UI slider
        max_plot_points = sim_duration * 60  # Convert to minutes
        
        # Limit the data to the UI slider duration for plotting
        plot_load_curve = results['load_curve'][:max_plot_points]
        
        # Convert to day-hour format (01-00, 01-04, 01-08, ..., 01-24, 02-04, etc.)
        time_hours = np.arange(len(plot_load_curve)) / 60
        time_day_hour = time_hours  # Keep continuous hours for plotting
        
        # Initialize grid-related variables
        plot_grid_limit = None
        plot_adjusted_grid_limit = None
        plot_original_grid_limit = None
        grid_time = None
        grid_time_day_hour = None
        
        # Top plot: EV Load and Grid Limits
        if results['grid_limit'] is not None:
            # Limit grid data to UI slider duration for plotting
            plot_grid_limit = results['grid_limit'][:max_plot_points]
            plot_adjusted_grid_limit = results['adjusted_grid_limit_before_margin'][:max_plot_points] if 'adjusted_grid_limit_before_margin' in results else None
            plot_original_grid_limit = results['original_grid_limit'][:max_plot_points] if 'original_grid_limit' in results else None
            
            # Handle case where load curve is 48 hours but grid curves are 60 hours
            # Load curve is already processed to 48 hours in sim_setup.py
            # Grid curves remain at 60 hours, so we need to trim them to match load curve
            if len(plot_load_curve) < len(plot_grid_limit):
                # Load curve is shorter (48 hours), trim grid curves to match
                plot_grid_limit = plot_grid_limit[:len(plot_load_curve)]
                if plot_adjusted_grid_limit is not None:
                    plot_adjusted_grid_limit = plot_adjusted_grid_limit[:len(plot_load_curve)]
                if plot_original_grid_limit is not None:
                    plot_original_grid_limit = plot_original_grid_limit[:len(plot_load_curve)]
            
            grid_time = np.arange(len(plot_grid_limit)) / 60
            grid_time_day_hour = grid_time  # Keep continuous hours for plotting
            time_hours = np.arange(len(plot_load_curve)) / 60
            time_day_hour = time_hours  # Keep continuous hours for plotting
        
        # Resample data to 10-minute intervals if smooth_graph is enabled
        if smooth_graph:
            # Resample from 1-minute to 10-minute intervals
            resample_factor = 10
            plot_load_curve = plot_load_curve[::resample_factor]
            if plot_grid_limit is not None:
                plot_grid_limit = plot_grid_limit[::resample_factor]
            if plot_adjusted_grid_limit is not None:
                plot_adjusted_grid_limit = plot_adjusted_grid_limit[::resample_factor]
            if plot_original_grid_limit is not None:
                plot_original_grid_limit = plot_original_grid_limit[::resample_factor]
            
            # Adjust time arrays accordingly
            time_hours = np.arange(len(plot_load_curve)) * resample_factor / 60
            time_day_hour = time_hours
            if grid_time is not None:
                grid_time = np.arange(len(plot_grid_limit)) * resample_factor / 60
                grid_time_day_hour = grid_time
        
        # 1. Grid Limit (red dashed line)
        if 'adjusted_grid_limit_before_margin' in results and plot_adjusted_grid_limit is not None:
            if smooth_graph:
                ax1.plot(grid_time_day_hour, plot_adjusted_grid_limit, color='red', linestyle='--', alpha=0.7, label='Grid Limit')
            else:
                ax1.step(grid_time_day_hour, plot_adjusted_grid_limit, where='post', color='red', linestyle='--', alpha=0.7, label='Grid Limit')
        
        # 2. Battery Effects (charging and discharging)
        if show_battery_effects and plot_original_grid_limit is not None:
            pv_charge = results.get('pv_battery_charge_curve') if results.get('pv_battery_applied', False) else None
            grid_discharge = results.get('grid_battery_discharge_curve') if results.get('grid_battery_applied', False) else None
            v2g_discharge = results.get('v2g_discharge_curve') if results.get('v2g_applied', False) else None
            grid_charge = results.get('grid_battery_charge_curve') if results.get('grid_battery_applied', False) else None
            
            # Get PV direct system support and battery discharge curves
            pv_direct_support = results.get('pv_direct_support_curve') if results.get('pv_battery_applied', False) else None
            pv_battery_discharge = results.get('pv_battery_discharge_curve') if results.get('pv_battery_applied', False) else None
            
            # Resample battery effects data if smooth_graph is enabled
            if smooth_graph:
                resample_factor = 10
                if pv_charge is not None:
                    pv_charge = pv_charge[:max_plot_points][::resample_factor]
                if grid_discharge is not None:
                    grid_discharge = grid_discharge[:max_plot_points][::resample_factor]
                if v2g_discharge is not None:
                    v2g_discharge = v2g_discharge[:max_plot_points][::resample_factor]
                if grid_charge is not None:
                    grid_charge = grid_charge[:max_plot_points][::resample_factor]
                if pv_direct_support is not None:
                    pv_direct_support = pv_direct_support[:max_plot_points][::resample_factor]
                if pv_battery_discharge is not None:
                    pv_battery_discharge = pv_battery_discharge[:max_plot_points][::resample_factor]
            
            # PV direct system support (lightgreen shading - PV discharge during the day)
            if pv_direct_support is not None and np.any(pv_direct_support > 0):
                if smooth_graph:
                    plot_pv_direct_support = pv_direct_support
                else:
                    plot_pv_direct_support = pv_direct_support[:max_plot_points]
                ax1.fill_between(grid_time_day_hour, plot_original_grid_limit, plot_original_grid_limit + plot_pv_direct_support, 
                               color='lightgreen', alpha=0.4, label='PV Direct System Support (Increases Capacity)')
            
            # 3. Battery Charging Effects (separate PV and Grid charging)
            if pv_charge is not None and np.any(pv_charge > 0):
                # PV battery charging (orange shading with diagonal stripes - stacked on top of PV direct support)
                if smooth_graph:
                    plot_pv_charge = pv_charge
                else:
                    plot_pv_charge = pv_charge[:max_plot_points]
                # Calculate the base level for stacking (original grid limit + PV direct support)
                base_level = plot_original_grid_limit
                if pv_direct_support is not None and np.any(pv_direct_support > 0):
                    base_level = plot_original_grid_limit + plot_pv_direct_support
                
                ax1.fill_between(grid_time_day_hour, base_level, base_level + plot_pv_charge, 
                               color='orange', alpha=0.4, hatch='////', label='PV Battery Charging (No grid effect)')
            
            if grid_charge is not None and np.any(grid_charge > 0):
                # Grid battery charging (lightcoral shading - reduces grid capacity)
                if smooth_graph:
                    plot_grid_charge = grid_charge
                else:
                    plot_grid_charge = grid_charge[:max_plot_points]
                ax1.fill_between(grid_time_day_hour, plot_original_grid_limit - plot_grid_charge, plot_original_grid_limit, 
                               color='lightcoral', alpha=0.3, label='Grid Battery Charging (Reduces Capacity)')
            
            # Combine all battery discharging effects (PV battery + grid battery + V2G discharge)
            has_battery_discharge_effects = (pv_battery_discharge is not None and np.any(pv_battery_discharge > 0)) or (grid_discharge is not None and np.any(grid_discharge > 0)) or (v2g_discharge is not None and np.any(v2g_discharge > 0))
            if has_battery_discharge_effects:
                if smooth_graph:
                    combined_battery_discharge = np.zeros_like(plot_original_grid_limit)
                else:
                    combined_battery_discharge = np.zeros_like(plot_original_grid_limit)
                if pv_battery_discharge is not None:
                    if smooth_graph:
                        plot_pv_battery_discharge = pv_battery_discharge
                    else:
                        plot_pv_battery_discharge = pv_battery_discharge[:max_plot_points]
                    combined_battery_discharge += plot_pv_battery_discharge
                if grid_discharge is not None:
                    if smooth_graph:
                        plot_grid_discharge = grid_discharge
                    else:
                        plot_grid_discharge = grid_discharge[:max_plot_points]
                    combined_battery_discharge += plot_grid_discharge
                if v2g_discharge is not None:
                    if smooth_graph:
                        plot_v2g_discharge = v2g_discharge
                    else:
                        plot_v2g_discharge = v2g_discharge[:max_plot_points]
                    combined_battery_discharge += plot_v2g_discharge
                
                # Plot combined battery discharging effects (light blue color for battery discharge during peak)
                ax1.fill_between(grid_time_day_hour, plot_original_grid_limit, plot_original_grid_limit + combined_battery_discharge, 
                               color='#87CEEB', alpha=0.4, label='Battery Discharge Effects (Combined)')
        
        # 4. Grid Limit with margin (orange dashed line)
        if plot_grid_limit is not None:
            safety_percentage = st.session_state.get('available_load_fraction', 0.8) * 100
            if smooth_graph:
                ax1.plot(grid_time_day_hour, plot_grid_limit, color='orange', linestyle='--', alpha=0.9, label=f'Grid Limit ({safety_percentage:.0f}% safety margin)')
            else:
                ax1.step(grid_time_day_hour, plot_grid_limit, where='post', color='orange', linestyle='--', alpha=0.9, label=f'Grid Limit ({safety_percentage:.0f}% safety margin)')
    
        # 5. EV Load (blue line)
        if smooth_graph:
            ax1.plot(time_day_hour, plot_load_curve, 'b-', linewidth=2, label='EV Load')
        else:
            ax1.step(time_day_hour, plot_load_curve, where='post', color='blue', linewidth=2, label='EV Load')
        
        # Ensure y-axis shows both lines
        if results['grid_limit'] is not None and plot_grid_limit is not None and plot_original_grid_limit is not None:
            min_y = 0
            
            # Calculate maximum values including all battery effects
            max_values = [np.max(plot_grid_limit), np.max(plot_original_grid_limit), np.max(plot_load_curve)]
            
            # Add battery effects to the maximum calculation
            if show_battery_effects:
                # PV direct system support
                pv_direct_support = results.get('pv_direct_support_curve') if results.get('pv_battery_applied', False) else None
                if pv_direct_support is not None:
                    pv_direct_support_plot = pv_direct_support[:max_plot_points]
                    if smooth_graph:
                        pv_direct_support_plot = pv_direct_support_plot[::10]
                    max_values.append(np.max(plot_original_grid_limit + pv_direct_support_plot))
                
                # PV battery charging
                pv_charge = results.get('pv_battery_charge_curve') if results.get('pv_battery_applied', False) else None
                if pv_charge is not None:
                    pv_charge_plot = pv_charge[:max_plot_points]
                    if smooth_graph:
                        pv_charge_plot = pv_charge_plot[::10]
                    # Calculate base level for PV charging
                    base_level = plot_original_grid_limit
                    if pv_direct_support is not None:
                        pv_direct_support_plot = pv_direct_support[:max_plot_points]
                        if smooth_graph:
                            pv_direct_support_plot = pv_direct_support_plot[::10]
                        base_level = plot_original_grid_limit + pv_direct_support_plot
                    max_values.append(np.max(base_level + pv_charge_plot))
                
                # Grid battery charging (reduces capacity, so check minimum)
                grid_charge = results.get('grid_battery_charge_curve') if results.get('grid_battery_applied', False) else None
                if grid_charge is not None:
                    grid_charge_plot = grid_charge[:max_plot_points]
                    if smooth_graph:
                        grid_charge_plot = grid_charge_plot[::10]
                    min_y = min(min_y, np.min(plot_original_grid_limit - grid_charge_plot))
                
                # Battery discharge effects (increases capacity)
                pv_battery_discharge = results.get('pv_battery_discharge_curve') if results.get('pv_battery_applied', False) else None
                grid_discharge = results.get('grid_battery_discharge_curve') if results.get('grid_battery_applied', False) else None
                v2g_discharge = results.get('v2g_discharge_curve') if results.get('v2g_applied', False) else None
                
                combined_discharge = np.zeros_like(plot_original_grid_limit)
                if pv_battery_discharge is not None:
                    pv_discharge_plot = pv_battery_discharge[:max_plot_points]
                    if smooth_graph:
                        pv_discharge_plot = pv_discharge_plot[::10]
                    combined_discharge += pv_discharge_plot
                if grid_discharge is not None:
                    grid_discharge_plot = grid_discharge[:max_plot_points]
                    if smooth_graph:
                        grid_discharge_plot = grid_discharge_plot[::10]
                    combined_discharge += grid_discharge_plot
                if v2g_discharge is not None:
                    v2g_discharge_plot = v2g_discharge[:max_plot_points]
                    if smooth_graph:
                        v2g_discharge_plot = v2g_discharge_plot[::10]
                    combined_discharge += v2g_discharge_plot
                
                if np.any(combined_discharge > 0):
                    max_values.append(np.max(plot_original_grid_limit + combined_discharge))
            
            max_y = max(max_values) * 1.1
            
            # Safety checks to ensure reasonable y-axis limits
            if max_y > 10000:  # If battery effects create extremely high values
                max_y = max(max_values) * 1.05  # Reduce padding
            if max_y < 100:  # If values are very low
                max_y = max(max_values) * 1.2  # Increase padding for visibility
            
            # Ensure minimum range for visibility
            if max_y - min_y < 50:
                max_y = min_y + 50
            
            ax1.set_ylim(min_y, max_y)
        else:
            # Set y-axis limits based only on load curve if no grid data
            min_y = 0
            max_y = np.max(plot_load_curve) * 1.1
            ax1.set_ylim(min_y, max_y)
        
        if show_legend:
            ax1.legend(loc='lower left', fontsize=12, frameon=True)
        
        ax1.set_xlabel('Time (Day-Hour)')
        ax1.set_ylabel('Power (kW)')
        mode_text = "Grid Constrained" if results['grid_mode'] == "Grid Constrained" else "Reference Only"
        ax1.set_title(f'EV Charging Load vs Grid Capacity ({mode_text})')
        ax1.grid(True, alpha=0.3)
        
        # Create custom x-axis labels in day-hour format
        max_hours = int(np.ceil(time_day_hour[-1])) if len(time_day_hour) > 0 else 48
        x_ticks = []
        x_labels = []
        
        for hour in range(0, max_hours + 1, 4):  # Every 4 hours
            day = (hour // 24) + 1
            hour_in_day = hour % 24
            
            # Format as "Day-Hour" (e.g., "01-00", "01-04", "01-08", ..., "02-00", "02-04")
            x_labels.append(f"{day:02d}-{hour_in_day:02d}")
            x_ticks.append(hour)
        
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_labels, rotation=0, fontsize=12, ha='center')
        
        # Set x-axis limits to show the full range
        ax1.set_xlim(0, max_hours)
        
        # Bottom plot: Available Grid Capacity
        if results['grid_limit'] is not None:
            # Use the same limited data for bottom plot
            # Calculate available capacity up to the adjusted grid limit (with battery effects and 80% margin)
            available_capacity_total = plot_grid_limit - plot_load_curve
            available_capacity_total = np.maximum(available_capacity_total, 0)  # Ensure non-negative
            
            # Calculate available capacity up to the adjusted grid limit before 80% margin (with battery effects)
            available_capacity_before_margin = plot_adjusted_grid_limit - plot_load_curve
            available_capacity_before_margin = np.maximum(available_capacity_before_margin, 0)  # Ensure non-negative
            
            # Plot available capacity up to adjusted limit before 80% margin (with battery effects)
            ax2.fill_between(grid_time_day_hour, 0, available_capacity_before_margin, color='lightgreen', alpha=0.7, label='Available Grid Capacity')
            if smooth_graph:
                ax2.plot(grid_time_day_hour, available_capacity_before_margin, 'g-', linewidth=2, label='_nolegend_')
            else:
                ax2.step(grid_time_day_hour, available_capacity_before_margin, where='post', color='green', linewidth=2, label='_nolegend_')
            
            # Add red dashed line to show the 20% margin that shouldn't be used
            if show_safety_margin:
                margin_capacity = plot_adjusted_grid_limit - plot_grid_limit
                if smooth_graph:
                    ax2.plot(grid_time_day_hour, margin_capacity, 'r--', linewidth=2, alpha=0.8, label='20% Safety Margin')
                else:
                    ax2.step(grid_time_day_hour, margin_capacity, where='post', color='red', linestyle='--', linewidth=2, alpha=0.8, label='20% Safety Margin')
            
            # Add average line if requested
            if show_average_line and np.any(available_capacity_before_margin > 0):
                avg_available = np.mean(available_capacity_before_margin)
                ax2.axhline(y=avg_available, color='blue', linestyle='-', alpha=0.7, linewidth=2, label=f'Average ({avg_available:.1f} kW)')
            
            # Add horizontal line at zero
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Set y-axis limits for bottom plot
            max_available = np.max(available_capacity_before_margin) * 1.1 if np.max(available_capacity_before_margin) > 0 else 100
            
            # Also consider margin capacity if it's being shown
            if show_safety_margin:
                margin_capacity = plot_adjusted_grid_limit - plot_grid_limit
                max_available = max(max_available, np.max(margin_capacity) * 1.1)
            
            # Ensure minimum range for visibility
            if max_available < 50:
                max_available = 50
            
            # Safety checks to ensure reasonable y-axis limits
            if max_available > 10000:  # If battery effects create extremely high values
                max_available = np.max(available_capacity_before_margin) * 1.05  # Reduce padding
            if max_available < 100:  # If values are very low
                max_available = max(max_available, 100)  # Ensure minimum visibility
            
            ax2.set_ylim(0, max_available)
            
            if show_legend:
                ax2.legend(loc='upper right', fontsize=12, frameon=True)
        
        ax2.set_xlabel('Time (Day-Hour)')
        ax2.set_ylabel('Available Capacity (kW)')
        ax2.set_title('Available Grid Capacity')
        ax2.grid(True, alpha=0.3)
        
        # Use the same x-axis format for bottom plot
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels, rotation=0, fontsize=12, ha='center')
        
        # Set x-axis limits to show the full range
        ax2.set_xlim(0, max_hours)
        
        # Adjust layout
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Show info message if graph was auto-resized due to battery effects
        if show_battery_effects and results['grid_limit'] is not None:
            # Check if battery effects caused significant scaling
            base_max = max(np.max(plot_grid_limit), np.max(plot_original_grid_limit), np.max(plot_load_curve))
            actual_max = max_y / 1.1  # Remove the padding to get actual max
            
            
            # Check if bottom plot was also adjusted
            base_available_max = np.max(plot_grid_limit - plot_load_curve) if len(plot_grid_limit) == len(plot_load_curve) else 0
            actual_available_max = max_available / 1.1  # Remove the padding to get actual max
            
        
        # Display statistics under the graph
        st.header("📊 Performance Metrics")
        
        # Filter load curve to hours 5-48 (minutes 300-2880) for performance calculations
        # Keep graphs as they are, only filter calculations
        start_minute = 300  # 5 hours * 60 minutes
        end_minute = 2880   # 48 hours * 60 minutes
        
        # Create filtered load curve for performance calculations
        if len(results['load_curve']) >= end_minute:
            filtered_load_curve = results['load_curve'][start_minute:end_minute]
        else:
            # If simulation is shorter than 48 hours, use what's available
            filtered_load_curve = results['load_curve']
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            st.metric("Peak Load", f"{np.max(filtered_load_curve):.2f} kW", 
                     help="Maximum instantaneous power demand during hours 5-48 of the simulation")
            
            # Calculate average load only for periods with active charging
            active_periods = filtered_load_curve > 0
            if np.any(active_periods):
                active_average = np.mean(filtered_load_curve[active_periods])
                st.metric("Average Load (Active Periods)", f"{active_average:.2f} kW",
                         help="Average power demand only during periods when EVs are actively charging (hours 5-48)")
            else:
                st.metric("Average Load (Active Periods)", "0.00 kW",
                         help="Average power demand only during periods when EVs are actively charging (hours 5-48)")
            
            st.metric("Total EVs", f"{results['total_evs']} EVs",
                     help="Total number of EVs in the simulation")
            
            # Calculate max simultaneous EVs
            if np.any(active_periods):
                ev_charging_rate = st.session_state.dynamic_ev.get('AC', 11.0)  # Default 11 kW
                max_simultaneous_evs = int(np.max(filtered_load_curve) / ev_charging_rate)
                st.metric("Max Simultaneous EVs", f"{max_simultaneous_evs} EVs",
                         help="Maximum number of EVs charging at the same time (based on peak load, hours 5-48)")
            else:
                st.metric("Max Simultaneous EVs", "0 EVs",
                         help="Maximum number of EVs charging at the same time (based on peak load, hours 5-48)")
        
        with col_stats2:
            st.metric("Total Energy", f"{np.sum(filtered_load_curve) / 60:.2f} kWh",
                     help="Total energy consumed by all EVs during hours 5-48 of the simulation (kWh)")
            
            # Calculate peak hours analysis
            if np.any(active_periods):
                peak_load = np.max(filtered_load_curve)
                peak_threshold = peak_load * 0.9  # 90% of peak load
                peak_hours = filtered_load_curve >= peak_threshold
                peak_hour_count = np.sum(peak_hours)
                # Adjust time to account for filtering (add 5 hours to start time)
                peak_start = (np.where(peak_hours)[0][0] / 60 + 5) if np.any(peak_hours) else 0
                peak_end = (np.where(peak_hours)[0][-1] / 60 + 5) if np.any(peak_hours) else 0
                st.metric("Peak Hours (90% of max)", f"{peak_start:.1f}-{peak_end:.1f}h",
                         help="Time range when load is ≥90% of peak load (hours 5-48)")
            else:
                st.metric("Peak Hours (90% of max)", "No peak",
                         help="Time range when load is ≥90% of peak load (hours 5-48)")
            
            # Calculate average energy per EV
            if results['total_evs'] > 0:
                avg_energy_per_ev = (np.sum(filtered_load_curve) / 60) / results['total_evs']
                st.metric("Avg Energy per EV", f"{avg_energy_per_ev:.1f} kWh",
                         help="Average energy consumed per EV during hours 5-48 of the simulation")
            else:
                st.metric("Avg Energy per EV", "0.0 kWh",
                         help="Average energy consumed per EV during hours 5-48 of the simulation")
            
            # Calculate charging session duration
            if results['total_evs'] > 0:
                # Calculate based on energy consumed and charging rate
                total_energy_kwh = np.sum(filtered_load_curve) / 60  # Convert from kW-minutes to kWh
                avg_energy_per_ev = total_energy_kwh / results['total_evs']
                ev_charging_rate = st.session_state.dynamic_ev.get('AC', 11.0)  # Default 11 kW
                
                # Average session duration = average energy per EV / charging rate
                avg_session_duration = avg_energy_per_ev / ev_charging_rate
                st.metric("Avg Charging Session", f"{avg_session_duration:.1f}h",
                         help="Average charging session duration per EV (based on energy consumed, hours 5-48)")
            else:
                st.metric("Avg Charging Session", "0.0h",
                         help="Average charging session duration per EV (based on energy consumed, hours 5-48)")
        
        with col_stats3:
            if results['grid_limit'] is not None:
                # Calculate available capacity (same as second graph) - use filtered data for calculations
                if len(results['adjusted_grid_limit_before_margin']) >= end_minute:
                    filtered_grid_limit = results['adjusted_grid_limit_before_margin'][start_minute:end_minute]
                else:
                    filtered_grid_limit = results['adjusted_grid_limit_before_margin']
                
                available_capacity_before_margin = filtered_grid_limit - filtered_load_curve
                available_capacity_before_margin = np.maximum(available_capacity_before_margin, 0)  # Ensure non-negative
                
                max_available_capacity = np.max(available_capacity_before_margin)
                min_available_capacity = np.min(available_capacity_before_margin)
                avg_available_capacity = np.mean(available_capacity_before_margin)
                
                st.metric("Max Available Capacity", f"{max_available_capacity:.1f} kW",
                         help="Maximum available grid capacity during hours 5-48 (from second graph)")
                st.metric("Min Available Capacity", f"{min_available_capacity:.1f} kW",
                         help="Minimum available grid capacity during hours 5-48 (from second graph)")
                st.metric("Avg Available Capacity", f"{avg_available_capacity:.1f} kW",
                         help="Average available grid capacity during hours 5-48 (from second graph)")
                
                # Calculate grid utilization factor (avg vs max available capacity)
                if max_available_capacity > 0:
                    grid_utilization = (avg_available_capacity / max_available_capacity) * 100
                    st.metric("Grid Utilization Factor", f"{grid_utilization:.1f}%",
                             help="Ratio of average to max available capacity during hours 5-48 (higher = more consistent grid usage)")
                else:
                    st.metric("Grid Utilization Factor", "0.0%",
                             help="Ratio of average to max available capacity during hours 5-48 (higher = more consistent grid usage)")
            else:
                st.metric("Grid Mode", results['grid_mode'],
                         help="Current grid constraint mode (None = no limits)")
                st.metric("Grid Limit", "None",
                         help="Grid capacity limit (None = no limits applied)")
                st.metric("Avg Available Capacity", "∞ kW",
                         help="Average available grid capacity (unlimited)")
                st.metric("Grid Utilization Factor", "∞%",
                         help="Ratio of average to max available capacity (unlimited grid)")
        
        # Display smart charging information
        if results.get('smart_charging_applied', False):
            st.success(f"🌙 Smart Charging Applied: {results['smart_charging_percent']}% of EVs shifted to {results['shift_start_time']}:00")
        
        # Display PV + Battery information
        if results.get('pv_battery_applied', False):
            actual_duration = results['battery_capacity'] / results['max_discharge_rate']
            end_hour = (results['discharge_start_hour'] + actual_duration) % 24
            st.success(f"☀️ PV + Battery Applied: {results['pv_adoption_percent']}% adoption, {results['battery_capacity']:.1f} kWh capacity, {results['max_discharge_rate']:.1f} kW max rate, {results['discharge_start_hour']}:00-{end_hour:.1f}:00 ({actual_duration:.1f}h)")
        
        # Display Grid-Charged Battery information
        if results.get('grid_battery_applied', False):
            charge_end_hour = (results['grid_battery_charge_start_hour'] + results['grid_battery_charge_duration']) % 24
            discharge_end_hour = (results['grid_battery_discharge_start_hour'] + results['grid_battery_discharge_duration']) % 24
            st.success(f"🔋 Grid-Charged Batteries Applied: {results['grid_battery_adoption_percent']}% adoption, {results['grid_battery_capacity']:.1f} kWh capacity, charge {results['grid_battery_charge_start_hour']}:00-{charge_end_hour}:00 ({results['grid_battery_charge_duration']}h), discharge {results['grid_battery_discharge_start_hour']}:00-{discharge_end_hour}:00 ({results['grid_battery_discharge_duration']}h)")
        
        # Display V2G information
        if results.get('v2g_applied', False):
            discharge_end_hour = (results['v2g_discharge_start_hour'] + results['v2g_discharge_duration']) % 24
            st.success(f"🚗 V2G Applied: {results['v2g_adoption_percent']}% adoption, {results['v2g_max_discharge_rate']:.1f} kW max rate, discharge {results['v2g_discharge_start_hour']}:00-{discharge_end_hour}:00 ({results['v2g_discharge_duration']}h), recharge arrival at {results['v2g_recharge_arrival_hour']}:00")
            
           

        
        # Display rejection notifications
        if results['rejected_evs']:
            st.warning(f"⚠️ {len(results['rejected_evs'])} EVs were rejected")
            for ev, reason in results['rejected_evs']:
                st.write(f"• {ev.name}: {reason}")

# Information panel on the right (only shown when no simulation has been run)
with col2:
    if ('simulation_results' in st.session_state and st.session_state.simulation_results is not None):
        st.header("ℹ️ Simulation Information")
        
        results = st.session_state.simulation_results
        
        # EV Configuration
        st.markdown("**🚗 EV Configuration**")
        st.write(f"• **Battery Capacity:** {st.session_state.dynamic_ev['capacity']} kWh")
        st.write(f"• **AC Charging Rate:** {st.session_state.dynamic_ev['AC']} kW")
        
        # Charger Configuration
        if 'charger_config' in st.session_state:
            st.markdown("**🔌 Charger Configuration**")
            config = st.session_state.charger_config
            st.write(f"• **AC Chargers:** {config['ac_count']} chargers ({config['ac_rate']} kW each)")
            total_charger_power = config['ac_count'] * config['ac_rate']
            st.write(f"• **Total Charger Capacity:** {total_charger_power} kW")
            st.write(f"• **Charger Utilization:** {total_charger_power / results['total_evs']:.1f} kW per EV")
        
        # Time Control & Peak Summary
        st.markdown("**⏰ Time Control**")
        st.write(f"• **Total EVs:** {results['total_evs']}")
        st.write(f"• **Simulation Duration:** {results['sim_duration']} hours")
        for peak in st.session_state.time_peaks:
            if peak['enabled']:
                st.write(f"• **{peak['name']}:** {peak['quantity']} EVs at {peak['time']}:00 ± {peak['span']}h")
        
        # EV Adoption Percentage
        if 'ev_calculator' in st.session_state:
            ev_calc = st.session_state.ev_calculator
            total_cars_system = ev_calc.get('total_cars', 6100000)
            num_substations = ev_calc.get('num_substations', 69000)
            
            # Calculate adoption percentage: total cars in the system at this year/(number of cars in simulation (per substation) * number of substations) 
            adoption_percentage = total_cars_system / (results['total_evs'] * num_substations) * 100
            st.markdown("**📊 EV Adoption**")
            st.write(f"• **System EVs:** {total_cars_system:,.0f}")
            st.write(f"• **Simulation EVs (per substation):** {results['total_evs']:,.0f}")
            st.write(f"• **Adoption Rate:** {adoption_percentage:.2f}%")
        
        # Active Strategies Summary
        active_strategies = st.session_state.get('active_strategies', [])
        if active_strategies:
            st.markdown("**🎯 Active Strategies**")
            for strategy in active_strategies:
                if strategy == 'smart_charging':
                    st.write(f"• **Smart Charging:** {results.get('smart_charging_percent', 0)}% adoption")
                elif strategy == 'pv_battery':
                    st.write(f"• **PV + Battery:** {results.get('pv_adoption_percent', 0)}% adoption")
                elif strategy == 'grid_battery':
                    st.write(f"• **Grid Battery:** {results.get('grid_battery_adoption_percent', 0)}% adoption")
                elif strategy == 'v2g':
                    st.write(f"• **V2G:** {results.get('v2g_adoption_percent', 0)}% adoption")
                elif strategy == 'time_of_use':
                    st.write(f"• **Time of Use:** Active")
        
        # Grid Mode
        st.markdown("**⚡ Grid Configuration**")
        st.write(f"• **Grid Mode:** {results.get('grid_mode', 'Reference Only')}")
        if results.get('grid_mode') == "Grid Constrained":
            st.write(f"• **Safety Margin:** {st.session_state.get('available_load_fraction', 0.8) * 100:.0f}%")
        
        # Rejection Information
        if results.get('rejected_evs'):
            st.markdown("**❌ Rejection Information**")
            st.write(f"• **Rejected EVs:** {len(results['rejected_evs'])}")
            st.write(f"• **Rejection Rate:** {len(results['rejected_evs']) / results['total_evs'] * 100:.1f}%")
        
    else:
        st.header("ℹ️ Information")
        st.write("**How to use this simulation:**")
        st.write("**Option A:** Apply a scenario (recommended for quick start)")
        st.write("**Option B:** Manual configuration:")
        st.write("1. **Configure EVs** - Select EV model and initial charge level")
        st.write("2. **Configure Chargers** - Select charger type and count")
        st.write("3. **Set Time Control** - Configure simulation duration and charging peaks")
        st.write("**Then (for both options):**")
        st.write("4. **Enable Strategies** - Activate PV, V2G, or other optimization strategies")
        st.write("5. **Select Data** - Choose real dataset or generate synthetic load curves")
        st.write("6. **Run Simulation** - Execute the simulation and view results")
        
        st.write("**📊 Results include:**")
        st.write("• Grid load curves")
        st.write("• EV charging patterns")
        st.write("• Optimization strategy effects")
        st.write("• Peak demand analysis")

            # Check if we have valid data

# Import the new TOU optimizer
from pages.components.tou_optimizer import (
    optimize_tou_periods_24h, 
    convert_to_simulation_format, 
    validate_periods
)

def extract_margin_curve_from_current_data():
    """
    Extract margin curve from current grid data without running a simulation.
    
    Returns:
        List of margin curve values or None if no data available
    """
    try:
        # Get current configuration
        available_load_fraction = st.session_state.get('available_load_fraction', 0.8)
        capacity_margin = min(0.95, available_load_fraction + 0.1)  # Add 0.1 but cap at 0.95
        
        # Check if we have synthetic data
        if 'synthetic_load_curve' in st.session_state and st.session_state.synthetic_load_curve is not None:
            # Use synthetic data
            power_values = st.session_state.synthetic_load_curve
        elif 'selected_dataset' in st.session_state and st.session_state.selected_dataset:
            # Use real dataset
            try:
                df = pd.read_csv(f"datasets/{st.session_state.selected_dataset}")
                selected_date = st.session_state.get('selected_date', pd.to_datetime('2023-01-18').date())
                
                # Filter data for selected date
                df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                day_data = df[df['date'].dt.date == selected_date]
                
                if not day_data.empty:
                    power_values = day_data.iloc[:, 2].values
                else:
                    return None
            except Exception as e:
                st.error(f"Error reading dataset: {e}")
                return None
        else:
            return None
        
        # Upsample 15-minute data to 1-minute intervals
        margin_curve = np.repeat(power_values, 15).astype(float) * capacity_margin
        
        # Extend for 48 hours (2880 minutes)
        sim_duration_min = 48 * 60  # 48 hours
        
        if len(margin_curve) < sim_duration_min:
            # Repeat the profile to cover 48 hours
            num_repeats = sim_duration_min // len(margin_curve) + 1
            margin_curve = np.tile(margin_curve, num_repeats)[:sim_duration_min]
        else:
            margin_curve = margin_curve[:sim_duration_min]
        
        return margin_curve.tolist()
        
    except Exception as e:
        st.error(f"Error extracting margin curve: {e}")
        return None