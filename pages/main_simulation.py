import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os
import sys
import streamlit.components.v1 as components

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_setup import SimulationSetup
from EV import EV, EV_MODELS
from charger import CHARGER_MODELS
from pages.components.capacity_analyzer import find_max_cars_capacity
# Remove heavy RL import from top level - will import only when needed
# from rl_components.capacity_optimizer_ui import create_capacity_optimizer_ui

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
        
        # Display single peak
        if st.session_state.time_peaks:
            peak = st.session_state.time_peaks[0]  # Only use the first peak
            st.write(f"**📊 {peak['name']}**")
            with st.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    new_name = st.text_input("Peak Name", value=str(peak['name']), key="peak_name_0")
                    new_time = st.slider("Peak Time (hours)", min_value=0, max_value=23, value=int(peak['time']), key="peak_time_0")
                
                with col2:
                    # Ensure span is a float
                    span_value = float(peak['span']) if peak['span'] is not None else 1.5
                    new_span = st.slider("Time Span (1σ in hours)", min_value=0.5, max_value=12.0, value=span_value, step=0.5, key="peak_span_0", help="Spread of arrival times (1 standard deviation)")
                    new_quantity = st.number_input("EV Quantity", min_value=1, max_value=100, value=int(peak['quantity']), key="peak_quantity_0")
                
                # Update peak data
                st.session_state.time_peaks[0] = {
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
        

        
        # Initialize scenario values in session state if not exists
        if 'portugal_scenario' not in st.session_state:
            st.session_state.portugal_scenario = {
                'year': '2030',
                'scenario': 'realistic',
                'applied': False
            }
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_year = st.selectbox(
                "Year",
                options=list(portugal_ev_scenarios.keys())[1:] + list(st.session_state.get('custom_scenarios', {}).keys()),
                index=list(portugal_ev_scenarios.keys())[1:].index(st.session_state.portugal_scenario['year']) if st.session_state.portugal_scenario['year'] in list(portugal_ev_scenarios.keys())[1:] else 0,
                help="Select the target year for the scenario"
            )
        
        with col2:
            # Get available scenarios for selected year
            if selected_year in portugal_ev_scenarios:
                # Built-in scenarios for this year
                available_scenarios = ['conservative', 'realistic', 'aggressive']
            else:
                # Custom scenarios for this year
                available_scenarios = list(st.session_state.get('custom_scenarios', {}).get(selected_year, {}).get('scenarios', {}).keys())
            
            selected_scenario = st.selectbox(
                "Scenario",
                options=available_scenarios,
                index=0,
                help="Select the scenario type"
            )
        
        # Get scenario data
        if selected_scenario in ['conservative', 'realistic', 'aggressive']:
            # Built-in Portugal scenario
            scenario_data = portugal_ev_scenarios[selected_year]['scenarios'][selected_scenario]
            year_data = portugal_ev_scenarios[selected_year]
            total_cars_million = year_data['total_cars_million']
            total_evs_million = year_data['total_evs_million']
            home_charging_percent = year_data['home_charging_percent']
            home_charged_evs_million = year_data['home_charged_evs_million']
            smart_charging_percent = year_data.get('smart_charging_percent', 0)
            pv_adoption_percent = year_data.get('pv_adoption_percent', 0)
            battery_capacity = year_data.get('battery_capacity', 0)
            discharge_rate = year_data.get('discharge_rate', 0)
            solar_energy_percent = year_data.get('solar_energy_percent', 70)
            notes = year_data['notes']
        else:
            # Custom scenario
            scenario_data = st.session_state.custom_scenarios[selected_year]['scenarios'][selected_scenario]
            total_cars_million = st.session_state.custom_scenarios[selected_year]['total_cars_million']
            # Default values for custom scenarios
            total_evs_million = total_cars_million * scenario_data['ev_penetration']
            home_charging_percent = 0.80  # Default 80% for custom scenarios
            home_charged_evs_million = total_evs_million * home_charging_percent
            smart_charging_percent = 0  # Default 0% for custom scenarios
            pv_adoption_percent = 30  # Default 30% for custom scenarios
            battery_capacity = 17.5  # Default battery capacity for custom scenarios
            discharge_rate = 7.0  # Default discharge rate for custom scenarios
            solar_energy_percent = 70  # Default solar energy percent for custom scenarios
            notes = "Custom scenario"
        
        substation_count = portugal_ev_scenarios['substation_count']
        
        # Calculate EVs per substation
        total_cars = total_cars_million * 1_000_000
        evs_per_substation = (total_cars * scenario_data['ev_penetration']) / substation_count
        
        # Calculate grid battery parameters (needed for the button)
        try:
            scenario_year_int = int(selected_year)
        except Exception:
            scenario_year_int = 2030
        if scenario_year_int <= 2027:
            grid_battery_capacity = 12.5
            grid_battery_max_rate = 3.0
            grid_battery_adoption = 5
        elif scenario_year_int <= 2030:
            grid_battery_capacity = 20.0
            grid_battery_max_rate = 5.0
            grid_battery_adoption = 10
        else:
            grid_battery_capacity = 27.5
            grid_battery_max_rate = 7.0
            grid_battery_adoption = 15
        grid_battery_charge_start_hour = 7
        grid_battery_charge_duration = 8
        grid_battery_discharge_start_hour = 20
        grid_battery_discharge_duration = 4
        
        # Calculate V2G parameters (needed for display)
        if scenario_year_int <= 2027:
            v2g_adoption_percent = 5
        elif scenario_year_int <= 2030:
            v2g_adoption_percent = 20
        else:
            v2g_adoption_percent = 40
        
        # V2G discharge rate same as charging rate
        v2g_discharge_rate = scenario_data['charging_power_kW']
        
        # Apply scenario button (moved to top)
        if st.button("🚀 Apply Scenario", type="primary"):
            # Update EV parameters
            st.session_state.dynamic_ev = {
                'capacity': scenario_data['avg_battery_kWh'],
                'AC': scenario_data['charging_power_kW']
            }
            
            # Set default SOC for scenario (20% for realistic scenarios)
            st.session_state.ev_soc = 0.2
            
            # Update charger parameters
            st.session_state.charger_config = {
                'ac_rate': scenario_data['charging_power_kW'],
                'ac_count': math.ceil(evs_per_substation)
            }
            
            # Update EV calculator
            st.session_state.ev_calculator = {
                'num_substations': substation_count,
                'penetration_percent': scenario_data['ev_penetration'] * 100,
                'total_cars': int(total_cars),
                'evs_per_substation': evs_per_substation
            }
            
            # Update time peaks with calculated EVs
            total_evs = math.ceil(evs_per_substation)
            if total_evs > 0:
                # Update or create the single peak
                if 'time_peaks' in st.session_state and st.session_state.time_peaks:
                    # Update existing peak
                    st.session_state.time_peaks[0]['quantity'] = total_evs
                else:
                    # Create default peak if none exists
                    st.session_state.time_peaks = [
                        {
                            'name': 'Evening Peak',
                            'time': 19,
                            'span': 3,
                            'quantity': total_evs,
                            'enabled': True
                        }
                    ]
            
            # Update optimization strategy based on smart charging percentage
            if smart_charging_percent > 0:
                st.session_state.optimization_strategy['smart_charging_percent'] = smart_charging_percent
                st.session_state.optimization_strategy['shift_start_time'] = 23
            
            # Update PV + battery parameters from scenario data
            st.session_state.optimization_strategy['pv_adoption_percent'] = pv_adoption_percent
            st.session_state.optimization_strategy['battery_capacity'] = battery_capacity
            st.session_state.optimization_strategy['max_discharge_rate'] = discharge_rate
            st.session_state.optimization_strategy['discharge_start_hour'] = 20  # Default 8pm
            st.session_state.optimization_strategy['solar_energy_percent'] = solar_energy_percent
            
            # Update V2G parameters from scenario data
            # V2G adoption based on scenario year
            if scenario_year_int <= 2027:
                v2g_adoption_percent = 5
            elif scenario_year_int <= 2030:
                v2g_adoption_percent = 20
            else:
                v2g_adoption_percent = 40
            
            # V2G discharge rate same as charging rate
            v2g_discharge_rate = scenario_data['charging_power_kW']
            
            st.session_state.optimization_strategy['v2g_adoption_percent'] = v2g_adoption_percent
            st.session_state.optimization_strategy['v2g_max_discharge_rate'] = v2g_discharge_rate
            st.session_state.optimization_strategy['v2g_discharge_start_hour'] = 20  # Default 8pm
            st.session_state.optimization_strategy['v2g_discharge_duration'] = 3  # Default 3 hours
            st.session_state.optimization_strategy['v2g_recharge_arrival_hour'] = 26  # Default 2am next day
            
            # --- Set grid-charged battery (normal battery) values ---
            st.session_state.optimization_strategy['grid_battery_adoption_percent'] = grid_battery_adoption
            st.session_state.optimization_strategy['grid_battery_capacity'] = grid_battery_capacity
            st.session_state.optimization_strategy['grid_battery_max_rate'] = grid_battery_max_rate
            st.session_state.optimization_strategy['grid_battery_charge_start_hour'] = grid_battery_charge_start_hour
            st.session_state.optimization_strategy['grid_battery_charge_duration'] = grid_battery_charge_duration
            st.session_state.optimization_strategy['grid_battery_discharge_start_hour'] = grid_battery_discharge_start_hour
            st.session_state.optimization_strategy['grid_battery_discharge_duration'] = grid_battery_discharge_duration
            
            st.session_state.portugal_scenario = {
                'year': selected_year,
                'scenario': selected_scenario,
                'applied': True
            }
            
            # Set success message in session state
            st.session_state.scenario_success_message = f"✅ Scenario applied! {total_evs} EVs distributed across peaks. Smart charging set to {smart_charging_percent}%. Grid battery set to {grid_battery_adoption}% adoption. V2G set to {v2g_adoption_percent}% adoption."
            st.session_state.scenario_success_timer = 0
            
            st.rerun()
        
        # Display scenario details after the button
        st.write(f"**📊 Scenario Details:**")
        st.write(f"**Year:** {selected_year}")
        st.write(f"**Scenario:** {selected_scenario.title()}")
        
        st.write(f"**📈 General Data:**")
        st.write(f"• **Total Cars:** {total_cars_million:.1f}M")
        st.write(f"• **Total EVs:** {total_evs_million:.1f}M")
        st.write(f"• **EV Penetration:** {scenario_data['ev_penetration']*100:.0f}%")
        st.write(f"• **Home Charging:** {home_charging_percent*100:.0f}% ({home_charged_evs_million:.1f}M EVs)")
        st.write(f"• **EVs per Substation:** {evs_per_substation:.2f}")
        
        # --- EV Configuration (moved up) ---
        st.write(f"**🚗 EV Configuration:**")
        st.write(f"• **Battery Capacity:** {scenario_data['avg_battery_kWh']:.0f} kWh")
        st.write(f"• **Charging Power:** {scenario_data['charging_power_kW']:.1f} kW")
        st.write(f"• **Initial SOC:** 20%")
        
        # --- Charger Configuration (added) ---
        st.write(f"**🔌 Charger Configuration:**")
        st.write(f"• **AC Chargers:** {math.ceil(evs_per_substation)} chargers ({scenario_data['charging_power_kW']:.1f} kW each)")
        st.write(f"• **Total Charger Capacity:** {math.ceil(evs_per_substation) * scenario_data['charging_power_kW']:.1f} kW")
        
        st.write(f"**🌙 Smart Charging:**")
        st.write(f"• **Smart Charging:** {smart_charging_percent:.0f}%")
        
        st.write(f"**☀️ PV + Battery System:**")
        st.write(f"• **PV + Battery Adoption:** {pv_adoption_percent:.0f}%")
        st.write(f"• **Battery Capacity:** {battery_capacity:.1f} kWh")
        st.write(f"• **Discharge Rate:** {discharge_rate:.1f} kW")
        st.write(f"• **Solar Energy Available:** {solar_energy_percent:.0f}%")
        
        # --- Grid-Charged Battery (Normal Battery) Section ---
        st.write(f"**🔋 Grid-Charged Battery (Normal Battery):**")
        st.write(f"• **Adoption:** {grid_battery_adoption}%")
        st.write(f"• **Capacity:** {grid_battery_capacity} kWh")
        st.write(f"• **Max Rate:** {grid_battery_max_rate} kW")
        st.write(f"• **Charge:** {grid_battery_charge_start_hour}:00 for {grid_battery_charge_duration}h")
        st.write(f"• **Discharge:** {grid_battery_discharge_start_hour}:00 for {grid_battery_discharge_duration}h")
        
        # --- V2G Section ---
        st.write(f"**🚗 Vehicle-to-Grid (V2G):**")
        st.write(f"• **Adoption:** {v2g_adoption_percent}%")
        st.write(f"• **Discharge Rate:** {v2g_discharge_rate:.1f} kW (same as charging rate)")
        st.write(f"• **Discharge:** 20:00 for 3h")
        st.write(f"• **Recharge Arrival:** 02:00 next day (26:00)")
        
        st.write(f"**📝 Notes:** {notes}")
        
        # Display temporary success message if exists (right after the button)
        if 'scenario_success_message' in st.session_state and 'scenario_success_timer' in st.session_state:
            # Increment timer
            st.session_state.scenario_success_timer += 1
            
            # Show message for 2 seconds (assuming ~1 rerun per second)
            if st.session_state.scenario_success_timer < 3:  # 2-3 reruns
                st.success(st.session_state.scenario_success_message)
            else:
                # Clear the message after timer expires
                del st.session_state.scenario_success_message
                del st.session_state.scenario_success_timer

        # Custom scenario creation
        st.divider()
        if st.button("➕ Create New Scenario", type="secondary"):
            st.session_state.show_custom_scenario = True
        
        if st.session_state.get('show_custom_scenario', False):
            st.write("**🔧 Create Custom Scenario:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                custom_year = st.number_input(
                    "Year",
                    min_value=2020,
                    max_value=2050,
                    value=2030,
                    help="Target year for the scenario"
                )
                
                custom_name = st.text_input(
                    "Scenario Name",
                    value="custom",
                    help="Name for your custom scenario (e.g., 'my_scenario')"
                )
                
                custom_total_cars_million = st.number_input(
                    "Total Cars (Million)",
                    min_value=0.1,
                    max_value=20.0,
                    value=6.0,
                    step=0.1,
                    help="Total number of cars in millions"
                )
            
            with col2:
                custom_penetration = st.slider(
                    "EV Penetration (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=25.0,
                    step=0.1,
                    help="Percentage of total cars that are electric"
                )
                
                custom_battery_kwh = st.number_input(
                    "Average Battery Capacity (kWh)",
                    min_value=10.0,
                    max_value=200.0,
                    value=75.0,
                    step=1.0,
                    help="Average battery capacity in kWh"
                )
                
                custom_charging_power = st.number_input(
                    "Average Charging Power (kW)",
                    min_value=1.0,
                    max_value=100.0,
                    value=11.0,
                    step=0.1,
                    help="Average charging power in kW"
                )
            
            # Add a third column for home charging parameters
            col3 = st.columns(1)[0]
            
            with col3:
                custom_home_charging_percent = st.slider(
                    "Home Charging (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                    help="Percentage of EVs that charge at home"
                )
                
                custom_notes = st.text_area(
                    "Scenario Notes",
                    value="Custom scenario with user-defined parameters",
                    help="Brief description or notes about this scenario"
                )
            
            # Calculate EVs per substation for custom scenario
            custom_total_cars = custom_total_cars_million * 1_000_000
            custom_total_evs_million = custom_total_cars_million * custom_penetration / 100
            custom_home_charged_evs_million = custom_total_evs_million * custom_home_charging_percent / 100
            custom_evs_per_substation = (custom_total_cars * custom_penetration / 100) / substation_count
            
            st.write(f"**📊 Custom Scenario Preview:**")
            st.write(f"• **Year:** {custom_year}")
            st.write(f"• **Name:** {custom_name}")
            st.write(f"• **Total Cars:** {custom_total_cars_million:.1f}M")
            st.write(f"• **Total EVs:** {custom_total_evs_million:.1f}M")
            st.write(f"• **EV Penetration:** {custom_penetration:.1f}%")
            st.write(f"• **Home Charging:** {custom_home_charging_percent:.0f}% ({custom_home_charged_evs_million:.1f}M EVs)")
            st.write(f"• **Battery Capacity:** {custom_battery_kwh:.0f} kWh")
            st.write(f"• **Charging Power:** {custom_charging_power:.1f} kW")
            st.write(f"• **EVs per Substation:** {custom_evs_per_substation:.2f}")
            st.write(f"• **Notes:** {custom_notes}")
            
            col_save1, col_save2 = st.columns(2)
            
            with col_save1:
                if st.button("💾 Save Custom Scenario", type="primary"):
                    # Initialize custom scenarios in session state if not exists
                    if 'custom_scenarios' not in st.session_state:
                        st.session_state.custom_scenarios = {}
                    
                    # Add custom scenario
                    if str(custom_year) not in st.session_state.custom_scenarios:
                        st.session_state.custom_scenarios[str(custom_year)] = {
                            'total_cars_million': custom_total_cars_million,
                            'total_evs_million': custom_total_evs_million,
                            'home_charging_percent': custom_home_charging_percent / 100,
                            'home_charged_evs_million': custom_home_charged_evs_million,
                            'notes': custom_notes,
                            'scenarios': {}
                        }
                    
                    st.session_state.custom_scenarios[str(custom_year)]['scenarios'][custom_name] = {
                        'ev_penetration': custom_penetration / 100,
                        'avg_battery_kWh': custom_battery_kwh,
                        'charging_power_kW': custom_charging_power
                    }
                    
                    st.success(f"✅ Custom scenario '{custom_name}' for year {custom_year} saved!")
                    st.session_state.show_custom_scenario = False
                    st.rerun()
            
            with col_save2:
                if st.button("❌ Cancel", type="secondary"):
                    st.session_state.show_custom_scenario = False
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
                'discharge_start_hour': 20,
                'solar_energy_percent': 70,
                'grid_battery_adoption_percent': 10,
                'grid_battery_capacity': 20.0,
                'grid_battery_max_rate': 5.0,
                'grid_battery_charge_start_hour': 7,
                'grid_battery_charge_duration': 8,
                'grid_battery_discharge_start_hour': 20,
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
            
            # Initialize timeline if not in session state
            if 'time_of_use_timeline' not in st.session_state:
                st.session_state.time_of_use_timeline = {
                    'periods': [
                                    {'name': 'Super Off-Peak', 'color': '#87CEEB', 'adoption': 0, 'hours': list(range(2, 6))},
            {'name': 'Off-Peak', 'color': '#90EE90', 'adoption': 0, 'hours': list(range(1, 2)) + list(range(6, 8)) + list(range(22, 25))},
            {'name': 'Mid-Peak', 'color': '#FFD700', 'adoption': 0, 'hours': list(range(8, 9)) + list(range(11, 18)) + list(range(21, 22))},
            {'name': 'Peak', 'color': '#FF6B6B', 'adoption': 100, 'hours': list(range(9, 11)) + list(range(18, 21))}
                    ],
                    'selected_period': 0
                }
            
            # Store original timeline if not already stored
            if 'original_timeline' not in st.session_state:
                st.session_state.original_timeline = {
                    'periods': [
                        {'name': 'Super Off-Peak', 'color': '#87CEEB', 'adoption': 0, 'hours': list(range(2, 6))},
                        {'name': 'Off-Peak', 'color': '#90EE90', 'adoption': 0, 'hours': list(range(1, 2)) + list(range(6, 8)) + list(range(22, 25))},
                        {'name': 'Mid-Peak', 'color': '#FFD700', 'adoption': 0, 'hours': list(range(8, 9)) + list(range(11, 18)) + list(range(21, 22))},
                        {'name': 'Peak', 'color': '#FF6B6B', 'adoption': 100, 'hours': list(range(9, 11)) + list(range(18, 21))}
                    ]
                }
            
            timeline = st.session_state.time_of_use_timeline
            
            # Period adoption percentages at the top
            st.write("**📊 Period Adoption Percentages:**")
            st.write("Specify what percentage of users choose each period:")
            
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
                    
                    # Consistent spacing and alignment
                    st.markdown(f"""
                    <div style="margin: 8px 0 4px 0;">
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
            if total_adoption > 100:
                st.warning(f"⚠️ Total adoption is {total_adoption}% (exceeds 100%)")
            elif total_adoption < 100:
                st.info(f"ℹ️ Total adoption is {total_adoption}% ({100 - total_adoption}% of users not using Time of Use)")
            else:
                st.success(f"✅ Total adoption is {total_adoption}%")
            
            
            # Add period legend in 2 columns (compressed)
            legend_col1, legend_col2 = st.columns(2)
            with legend_col1:
                st.markdown("**Period Legend:**")
                st.markdown("- **S** = Super Off-Peak")
                st.markdown("- **O** = Off-Peak")
            with legend_col2:
                st.markdown("&nbsp;")  # Empty space for alignment
                st.markdown("- **M** = Mid-Peak")
                st.markdown("- **P** = Peak")
            
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
                if period['name'] == "Super Off-Peak":
                    short_name = "S"
                elif period['name'] == "Off-Peak":
                    short_name = "O"
                elif period['name'] == "Mid-Peak":
                    short_name = "M"
                elif period['name'] == "Peak":
                    short_name = "P"
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
            
            if st.button("🔄 Reset Time-of-Use Timeline", type="secondary"):
                # Clear session state assignments to restore original timeline defaults
                st.session_state.pop('hour_assignments', None)
                st.session_state.pop('initial_timeline', None)
                
                # Restore original timeline
                if 'original_timeline' in st.session_state:
                    st.session_state.time_of_use_timeline = st.session_state.original_timeline.copy()
                
                st.success("✅ Timeline reset to default values!")
                st.rerun()
            
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

                    
                    # Initialize with stored initial values or current timeline
                    st.session_state.hour_assignments = {}
                    if 'initial_timeline' in st.session_state and st.session_state.initial_timeline:
                        # Use stored initial values
                        for hour, period_name in st.session_state.initial_timeline.items():
                            st.session_state.hour_assignments[hour] = period_name
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
                    min_value=20.0,
                    max_value=30.0,
                    value=float(st.session_state.optimization_strategy.get('discharge_start_hour', 20)),  # 8pm default
                    step=0.5,
                    help="Hour when battery discharge starts (20-30)",
                    key="pv_discharge_start_hour"
                )
                min_discharge_duration = battery_capacity / max_discharge_rate
                # Round to nearest 0.5 to align with step
                min_discharge_duration_rounded = round(min_discharge_duration * 2) / 2
                # Use 3.0 as default if no value in session state, otherwise use the calculated minimum
                if 'discharge_duration' not in st.session_state.optimization_strategy:
                    default_duration = 3.0
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
            
            # Clear simulation results if PV battery parameters changed
            st.session_state.simulation_just_run = False
            # Don't clear simulation_results - let them persist

        # Grid-Charged Batteries configuration
        if 'grid_battery' in active_strategies:
            st.write("**🔋 Grid-Charged Batteries Configuration:**")
            st.write("Batteries charged from grid during day, discharged during evening peak.")
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
                    "Charging Duration (hours)",
                    min_value=1,
                    max_value=12,
                    value=st.session_state.optimization_strategy.get('grid_battery_charge_duration', 8),
                    step=1,
                    help="Duration of battery charging (hours)",
                    key="grid_charge_duration"
                )
            with col2:
                grid_battery_discharge_start_hour = st.slider(
                    "Discharge Start Hour",
                    min_value=20.0,
                    max_value=30.0,
                    value=float(st.session_state.optimization_strategy.get('grid_battery_discharge_start_hour', 20)),
                    step=0.5,
                    help="Hour when battery discharge starts (20-30)",
                    key="grid_discharge_start_hour"
                )
                grid_battery_discharge_duration = st.slider(
                    "Discharge Duration (hours)",
                    min_value=1,
                    max_value=8,
                    value=st.session_state.optimization_strategy.get('grid_battery_discharge_duration', 4),
                    step=1,
                    help="Duration of battery discharge (hours)",
                    key="grid_discharge_duration"
                )
            
            grid_battery_adoption_percent = st.slider(
                "Grid-Charged Battery Adoption (%)",
                min_value=0,
                max_value=100,
                value=st.session_state.optimization_strategy.get('grid_battery_adoption_percent', default_grid_battery_adoption),
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
            
            # Clear simulation results if grid battery parameters changed
            st.session_state.simulation_just_run = False
            # Don't clear simulation_results - let them persist
        
        # V2G (Vehicle-to-Grid) configuration
        if 'v2g' in active_strategies:
            st.write("**🚗 Vehicle-to-Grid (V2G) Configuration:**")
            st.write("EVs can discharge back to grid during peak demand, then recharge later.")
            
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
                    value=st.session_state.optimization_strategy.get('v2g_adoption_percent', default_v2g_adoption),
                    step=5,
                    help="Percentage of EVs that can participate in V2G",
                    key="v2g_adoption_percent"
                )
                v2g_discharge_start_hour = st.slider(
                    "V2G Discharge Start Hour",
                    min_value=17.0,
                    max_value=30.0,
                    value=float(st.session_state.optimization_strategy.get('v2g_discharge_start_hour', 20)),
                    step=0.5,
                    help="Hour when V2G discharge starts (20-30)",
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
                    min_value=24.0,
                    max_value=30.0,
                    value=float(st.session_state.optimization_strategy.get('v2g_recharge_arrival_hour', 26.0)),
                    step=0.5,
                    help="Hour when V2G EVs start recharging (24-30 = next day 0-6)",
                    key="v2g_recharge_arrival_hour"
                )

            
            # Calculate V2G effects
            # Get total EVs from the single peak
            if 'time_peaks' in st.session_state and st.session_state.time_peaks:
                total_evs = st.session_state.time_peaks[0]['quantity']
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
            
            # Clear simulation results if V2G parameters changed
            st.session_state.simulation_just_run = False
            # Don't clear simulation_results - let them persist

 # Reverse Simulation (Capacity Analysis)
    with st.expander("🔍 Reverse Simulation", expanded=False):
        st.write("**Find the maximum number of EVs that can fit under the margin curve with current parameters.**")
        
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
            with st.spinner("🔍 Analyzing system capacity..."):
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
                    st.success(f"🎯 **Maximum Capacity Found: {max_cars} EVs**")
                else:
                    st.error("❌ Could not determine maximum capacity")

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

        # RL Capacity Optimizer (separate from first optimizer)
        st.write("---")
        st.subheader("🤖 RL Capacity Optimizer")
        try:
            # Lazy import to avoid heavy imports on page load
            from rl_components.capacity_optimizer_ui import create_capacity_optimizer_ui
            create_capacity_optimizer_ui()
        except Exception as e:
            st.error(f"❌ RL Capacity Optimizer error: {e}")
            st.info("💡 This feature requires Stable-Baselines3. Install with: `pip install stable-baselines3`")



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
                                
                                # Add debug info
                                st.write(f"**Debug Info:** {data_points} points, {interval_minutes:.1f} min intervals")
                                
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
    
    if st.button("🚀 Run Simulation"):
        # Clear any existing results first
        if 'simulation_results' in st.session_state:
            del st.session_state.simulation_results
        
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
                # Calculate total EVs from the single peak
                if 'time_peaks' in st.session_state and st.session_state.time_peaks:
                    total_evs = st.session_state.time_peaks[0]['quantity']
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
                
                sim_duration_min = 48 * 60  # Always 48 hours for simulation
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
                    
                    if time_of_use_periods:
                        # Group periods by name for proper car distribution
                        period_groups = {}
                        for period in time_of_use_periods:
                            period_name = period['name']
                            if period_name not in period_groups:
                                period_groups[period_name] = []
                            period_groups[period_name].append(period)
                        
                        # Calculate how many days the simulation runs (always 48 hours = 2 days)
                        daily_minutes = 24 * 60
                        num_days = 2  # Always 2 days for 48-hour simulation
                        
                        # Pre-calculate all arrival times for Time of Use (repeating daily)
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
                                                    std_minutes = span_minutes / 4  # Standard deviation = span/4
                                                    
                                                    # Generate arrival times for this period on this day
                                                    period_arrival_times = np.random.normal(center_minutes, std_minutes, period_evs)
                                                    
                                                    # Clip to period boundaries and add day offset
                                                    period_arrival_times = np.clip(period_arrival_times, 
                                                                                 period['start'] * 60, 
                                                                                 period['end'] * 60)
                                                    period_arrival_times += day_offset_minutes
                                                    
                                                    # Add to arrival times list
                                                    arrival_times.extend(period_arrival_times)
                else:
                    # Use default single peak if Time of Use is not enabled
                    peak = st.session_state.time_peaks[0]
                    peak_mean = peak['time'] * 60
                    peak_span = peak['span'] * 60
                    sigma = peak_span
                    
                    peak_arrivals = np.random.normal(peak_mean, sigma, peak['quantity'])
                    arrival_times.extend(peak_arrivals)
                
                # Finalize arrival times (clip and sort once)
                arrival_times = np.array(arrival_times)
                arrival_times = np.clip(arrival_times, 0, 48 * 60 - 60)  # Always 48 hours for simulation
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
                    discharge_start_hour = st.session_state.optimization_strategy.get('discharge_start_hour', 20)
                    discharge_duration = st.session_state.optimization_strategy.get('discharge_duration', battery_capacity / max_discharge_rate)
                    actual_discharge_rate = st.session_state.optimization_strategy.get('actual_discharge_rate', max_discharge_rate)
                    
                    if pv_adoption_percent > 0 and battery_capacity > 0 and max_charge_rate > 0 and actual_discharge_rate > 0:
                        total_evs_for_pv = total_evs
                        pv_evs = int(total_evs_for_pv * pv_adoption_percent / 100)
                        total_charge_power = pv_evs * max_charge_rate
                        total_system_support_power = pv_evs * max_charge_rate  # Same as charge rate for system support
                        total_discharge_power = pv_evs * actual_discharge_rate * (solar_energy_percent / 100)
                        
                        # Calculate time periods
                        pv_start_minute = pv_start_hour * 60
                        charge_end_minute = pv_start_minute + int(charge_time * 60)
                        system_support_end_minute = pv_start_minute + int(pv_duration * 60)
                        discharge_start_minute = discharge_start_hour * 60
                        discharge_duration_minutes = int(discharge_duration * 60)
                        discharge_end_minute = discharge_start_minute + discharge_duration_minutes
                        
                        for minute in range(len(grid_profile_full)):
                            time_of_day = minute % (24 * 60)
                            
                            # Phase 1: Battery charging period (PV charges batteries, no grid effect)
                            if (time_of_day >= pv_start_minute and time_of_day < charge_end_minute):
                                pv_battery_charge_curve[minute] = total_charge_power
                                # No effect on grid capacity - PV charges batteries directly
                            
                            # Phase 2: System support period (increases available capacity)
                            elif (time_of_day >= charge_end_minute and time_of_day < system_support_end_minute):
                                pv_direct_support_curve[minute] = total_system_support_power
                                adjusted_grid_profile[minute] += total_system_support_power  # Increase available capacity
                            
                            # Phase 3: Evening discharge period (increases available capacity)
                            elif (time_of_day >= discharge_start_minute and time_of_day < discharge_end_minute):
                                pv_battery_discharge_curve[minute] = total_discharge_power
                                adjusted_grid_profile[minute] += total_discharge_power  # Increase available capacity
                
                # Apply Grid-Charged Batteries optimization if enabled
                grid_battery_charge_curve = np.zeros_like(grid_profile_full)
                grid_battery_discharge_curve = np.zeros_like(grid_profile_full)
                if 'grid_battery' in st.session_state.get('active_strategies', []):
                    grid_battery_adoption_percent = st.session_state.optimization_strategy.get('grid_battery_adoption_percent', 0)
                    grid_battery_capacity = st.session_state.optimization_strategy.get('grid_battery_capacity', 0)
                    grid_battery_max_rate = st.session_state.optimization_strategy.get('grid_battery_max_rate', 0)
                    grid_battery_charge_start_hour = st.session_state.optimization_strategy.get('grid_battery_charge_start_hour', 7)
                    grid_battery_charge_duration = st.session_state.optimization_strategy.get('grid_battery_charge_duration', 8)
                    grid_battery_discharge_start_hour = st.session_state.optimization_strategy.get('grid_battery_discharge_start_hour', 20)
                    grid_battery_discharge_duration = st.session_state.optimization_strategy.get('grid_battery_discharge_duration', 4)
                    
                    if grid_battery_adoption_percent > 0 and grid_battery_capacity > 0 and grid_battery_max_rate > 0:
                        total_evs_for_grid_battery = total_evs
                        grid_battery_evs = int(total_evs_for_grid_battery * grid_battery_adoption_percent / 100)
                        
                        # Use the same simple logic as PV battery
                        charge_rate = grid_battery_capacity / grid_battery_charge_duration
                        discharge_rate = grid_battery_capacity / grid_battery_discharge_duration
                        total_charge_power = grid_battery_evs * charge_rate
                        total_discharge_power = grid_battery_evs * discharge_rate
                        
                        # Create charging curve (reduces available capacity) - only during charging hours
                        charge_start_minute = grid_battery_charge_start_hour * 60
                        charge_duration_minutes = int(grid_battery_charge_duration * 60)
                        charge_end_minute = charge_start_minute + charge_duration_minutes
                        
                        # Create discharging curve (increases available capacity) - only during discharging hours
                        discharge_start_minute = grid_battery_discharge_start_hour * 60
                        discharge_duration_minutes = int(grid_battery_discharge_duration * 60)
                        discharge_end_minute = discharge_start_minute + discharge_duration_minutes
                        
                        for minute in range(len(grid_profile_full)):
                            # Handle charging curve (reduces available capacity)
                            if charge_end_minute > 24 * 60:
                                # Charging period extends beyond 24 hours, use absolute time
                                if (minute >= charge_start_minute and minute < charge_end_minute):
                                    grid_battery_charge_curve[minute] = total_charge_power
                                    adjusted_grid_profile[minute] -= total_charge_power  # Reduce available capacity
                            else:
                                # Charging period is within same day, use modulo logic
                                time_of_day = minute % (24 * 60)
                                if (time_of_day >= charge_start_minute and time_of_day < charge_end_minute):
                                    grid_battery_charge_curve[minute] = total_charge_power
                                    adjusted_grid_profile[minute] -= total_charge_power  # Reduce available capacity
                            
                            # Handle discharging curve (increases available capacity)
                            if discharge_end_minute > 24 * 60:
                                # Discharging period extends beyond 24 hours, use absolute time
                                if (minute >= discharge_start_minute and minute < discharge_end_minute):
                                    grid_battery_discharge_curve[minute] = total_discharge_power
                                    adjusted_grid_profile[minute] += total_discharge_power  # Increase available capacity
                            else:
                                # Discharging period is within same day, use modulo logic
                                time_of_day = minute % (24 * 60)
                                if (time_of_day >= discharge_start_minute and time_of_day < discharge_end_minute):
                                    grid_battery_discharge_curve[minute] = total_discharge_power
                                    adjusted_grid_profile[minute] += total_discharge_power  # Increase available capacity
                


                # Step 1: Run simulation with grid constraint
                # Create dynamic EV model
                dynamic_ev_model = {
                    'name': 'Custom EV',
                    'capacity': st.session_state.dynamic_ev['capacity'],
                    'AC': st.session_state.dynamic_ev['AC']
                }
                
                # Temporarily update EV_MODELS with our dynamic EV
                original_ev_models = EV_MODELS.copy()
                EV_MODELS['dynamic_ev'] = dynamic_ev_model
                
                # Step 1: Calculate battery effects and create final grid limit BEFORE simulation
                adjusted_grid_profile = grid_profile_full.copy()
                
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
                    discharge_start_hour = st.session_state.optimization_strategy.get('discharge_start_hour', 20)
                    discharge_duration = st.session_state.optimization_strategy.get('discharge_duration', battery_capacity / max_discharge_rate)
                    actual_discharge_rate = st.session_state.optimization_strategy.get('actual_discharge_rate', max_discharge_rate)
                    
                    if pv_adoption_percent > 0 and battery_capacity > 0 and max_charge_rate > 0 and actual_discharge_rate > 0:
                        total_evs_for_pv = total_evs
                        pv_evs = int(total_evs_for_pv * pv_adoption_percent / 100)
                        total_charge_power = pv_evs * max_charge_rate
                        total_system_support_power = pv_evs * max_charge_rate  # Same as charge rate for system support
                        total_discharge_power = pv_evs * actual_discharge_rate * (solar_energy_percent / 100)
                        
                        # Calculate time periods
                        pv_start_minute = pv_start_hour * 60
                        charge_end_minute = pv_start_minute + int(charge_time * 60)
                        system_support_end_minute = pv_start_minute + int(pv_duration * 60)
                        discharge_start_minute = discharge_start_hour * 60
                        discharge_duration_minutes = int(discharge_duration * 60)
                        discharge_end_minute = discharge_start_minute + discharge_duration_minutes
                        
                        for minute in range(len(grid_profile_full)):
                            time_of_day = minute % (24 * 60)
                            
                            # Phase 1: Battery charging period (PV charges batteries, no grid effect)
                            if (time_of_day >= pv_start_minute and time_of_day < charge_end_minute):
                                pv_battery_charge_curve[minute] = total_charge_power
                                # No effect on grid capacity - PV charges batteries directly
                            
                            # Phase 2: System support period (increases available capacity)
                            elif (time_of_day >= charge_end_minute and time_of_day < system_support_end_minute):
                                pv_direct_support_curve[minute] = total_system_support_power
                                adjusted_grid_profile[minute] += total_system_support_power  # Increase available capacity
                            
                            # Phase 3: Evening discharge period (increases available capacity)
                            elif (time_of_day >= discharge_start_minute and time_of_day < discharge_end_minute):
                                pv_battery_discharge_curve[minute] = total_discharge_power
                                adjusted_grid_profile[minute] += total_discharge_power  # Increase available capacity
                
                # Apply Grid-Charged Batteries optimization if enabled
                grid_battery_charge_curve = np.zeros_like(grid_profile_full)
                grid_battery_discharge_curve = np.zeros_like(grid_profile_full)
                if 'grid_battery' in st.session_state.get('active_strategies', []):
                    grid_battery_adoption_percent = st.session_state.optimization_strategy.get('grid_battery_adoption_percent', 0)
                    grid_battery_capacity = st.session_state.optimization_strategy.get('grid_battery_capacity', 0)
                    grid_battery_max_rate = st.session_state.optimization_strategy.get('grid_battery_max_rate', 0)
                    grid_battery_charge_start_hour = st.session_state.optimization_strategy.get('grid_battery_charge_start_hour', 7)
                    grid_battery_charge_duration = st.session_state.optimization_strategy.get('grid_battery_charge_duration', 8)
                    grid_battery_discharge_start_hour = st.session_state.optimization_strategy.get('grid_battery_discharge_start_hour', 20)
                    grid_battery_discharge_duration = st.session_state.optimization_strategy.get('grid_battery_discharge_duration', 4)
                    
                    if grid_battery_adoption_percent > 0 and grid_battery_capacity > 0 and grid_battery_max_rate > 0:
                        total_evs_for_grid_battery = total_evs
                        grid_battery_evs = int(total_evs_for_grid_battery * grid_battery_adoption_percent / 100)
                        
                        # Use the same simple logic as PV battery
                        charge_rate = grid_battery_capacity / grid_battery_charge_duration
                        discharge_rate = grid_battery_capacity / grid_battery_discharge_duration
                        total_charge_power = grid_battery_evs * charge_rate
                        total_discharge_power = grid_battery_evs * discharge_rate
                        
                        # Create charging curve (reduces available capacity) - only during charging hours
                        charge_start_minute = grid_battery_charge_start_hour * 60
                        charge_duration_minutes = int(grid_battery_charge_duration * 60)
                        charge_end_minute = charge_start_minute + charge_duration_minutes
                        
                        # Create discharging curve (increases available capacity) - only during discharging hours
                        discharge_start_minute = grid_battery_discharge_start_hour * 60
                        discharge_duration_minutes = int(grid_battery_discharge_duration * 60)
                        discharge_end_minute = discharge_start_minute + discharge_duration_minutes
                        
                        for minute in range(len(grid_profile_full)):
                            # Handle charging curve (reduces available capacity)
                            if charge_end_minute > 24 * 60:
                                # Charging period extends beyond 24 hours, use absolute time
                                if (minute >= charge_start_minute and minute < charge_end_minute):
                                    grid_battery_charge_curve[minute] = total_charge_power
                                    adjusted_grid_profile[minute] -= total_charge_power  # Reduce available capacity
                            else:
                                # Charging period is within same day, use modulo logic
                                time_of_day = minute % (24 * 60)
                                if (time_of_day >= charge_start_minute and time_of_day < charge_end_minute):
                                    grid_battery_charge_curve[minute] = total_charge_power
                                    adjusted_grid_profile[minute] -= total_charge_power  # Reduce available capacity
                            
                            # Handle discharging curve (increases available capacity)
                            if discharge_end_minute > 24 * 60:
                                # Discharging period extends beyond 24 hours, use absolute time
                                if (minute >= discharge_start_minute and minute < discharge_end_minute):
                                    grid_battery_discharge_curve[minute] = total_discharge_power
                                    adjusted_grid_profile[minute] += total_discharge_power  # Increase available capacity
                            else:
                                # Discharging period is within same day, use modulo logic
                                time_of_day = minute % (24 * 60)
                                if (time_of_day >= discharge_start_minute and time_of_day < discharge_end_minute):
                                    grid_battery_discharge_curve[minute] = total_discharge_power
                                    adjusted_grid_profile[minute] += total_discharge_power  # Increase available capacity
                
                # Apply V2G (Vehicle-to-Grid) optimization if enabled
                v2g_discharge_curve = np.zeros_like(grid_profile_full)
                if 'v2g' in st.session_state.get('active_strategies', []):
                    v2g_adoption_percent = st.session_state.optimization_strategy.get('v2g_adoption_percent', 0)
                    v2g_max_discharge_rate = st.session_state.optimization_strategy.get('v2g_max_discharge_rate', 0)
                    v2g_discharge_start_hour = st.session_state.optimization_strategy.get('v2g_discharge_start_hour', 20)
                    v2g_discharge_duration = st.session_state.optimization_strategy.get('v2g_discharge_duration', 2)
                    
                    if v2g_adoption_percent > 0 and v2g_max_discharge_rate > 0:
                        total_v2g_evs = int(total_evs * v2g_adoption_percent / 100)
                        total_v2g_discharge_power = total_v2g_evs * v2g_max_discharge_rate
                        discharge_start_minute = v2g_discharge_start_hour * 60
                        discharge_duration_minutes = int(v2g_discharge_duration * 60)
                        discharge_end_minute = discharge_start_minute + discharge_duration_minutes
                        
                        for minute in range(len(grid_profile_full)):
                            # Use absolute time comparison for periods that cross midnight
                            if discharge_end_minute > 24 * 60:
                                # Period extends beyond 24 hours, use absolute time
                                if (minute >= discharge_start_minute and minute < discharge_end_minute):
                                    v2g_discharge_curve[minute] = total_v2g_discharge_power
                                    adjusted_grid_profile[minute] += total_v2g_discharge_power  # Increase available capacity
                            else:
                                # Period is within same day, use modulo logic
                                time_of_day = minute % (24 * 60)
                                if (time_of_day >= discharge_start_minute and time_of_day < discharge_end_minute):
                                    v2g_discharge_curve[minute] = total_v2g_discharge_power
                                    adjusted_grid_profile[minute] += total_v2g_discharge_power  # Increase available capacity

                # Step 2: Apply 80% margin to the adjusted grid profile
                final_grid_profile = adjusted_grid_profile * st.session_state['available_load_fraction']

                # Step 3: Run simulation with the correct grid limit (battery effects + margin)
                # Set grid power limit for simulation
                if grid_mode == "Grid Constrained":
                    # Pass the final grid profile (with battery effects + margin) to the simulation
                    grid_power_limit = final_grid_profile
                else:
                    grid_power_limit = None  # No constraint for Reference Only mode
                
                # Always run simulation for 48 hours to avoid cars stacking at the end
                # The UI sim_duration slider only affects graph plotting
                sim = SimulationSetup(
                    ev_counts=ev_counts,
                    charger_counts=charger_counts,
                    sim_duration=48 * 60,  # Always 48 hours for simulation
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
                sim.env.run(until=48 * 60)  # Always run for 48 hours
                constrained_load_curve = sim.load_curve.copy()
                
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
                    'discharge_start_hour': st.session_state.optimization_strategy.get('discharge_start_hour', 20),
                    'solar_energy_percent': st.session_state.optimization_strategy.get('solar_energy_percent', 70),
                    'grid_battery_applied': 'grid_battery' in st.session_state.get('active_strategies', []),
                    'grid_battery_adoption_percent': st.session_state.optimization_strategy.get('grid_battery_adoption_percent', 0),
                    'grid_battery_capacity': st.session_state.optimization_strategy.get('grid_battery_capacity', 0),
                    'grid_battery_max_rate': st.session_state.optimization_strategy.get('grid_battery_max_rate', 0),
                    'grid_battery_charge_start_hour': st.session_state.optimization_strategy.get('grid_battery_charge_start_hour', 7),
                    'grid_battery_charge_duration': st.session_state.optimization_strategy.get('grid_battery_charge_duration', 8),
                    'grid_battery_discharge_start_hour': st.session_state.optimization_strategy.get('grid_battery_discharge_start_hour', 20),
                    'grid_battery_discharge_duration': st.session_state.optimization_strategy.get('grid_battery_discharge_duration', 4),
                    'v2g_applied': 'v2g' in st.session_state.get('active_strategies', []),
                    'v2g_adoption_percent': st.session_state.optimization_strategy.get('v2g_adoption_percent', 0),
                    'v2g_max_discharge_rate': st.session_state.optimization_strategy.get('v2g_max_discharge_rate', 0),
                    'v2g_discharge_start_hour': st.session_state.optimization_strategy.get('v2g_discharge_start_hour', 20),
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
        
        with col_controls3:
            show_safety_margin = st.checkbox("Show Safety Margin", value=True, key="show_safety_margin",
                                           help="Display 20% safety margin line")
        
        # Plot results with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[2, 1])
        
        # Define time axis - respect UI sim_duration slider for plotting
        sim_duration = st.session_state.get('sim_duration', 36)  # Get from UI slider
        max_plot_points = sim_duration * 60  # Convert to minutes
        
        # Limit the data to the UI slider duration for plotting
        plot_load_curve = results['load_curve'][:max_plot_points]
        time_hours = np.arange(len(plot_load_curve)) / 60
        
        # Top plot: EV Load and Grid Limits
        if results['grid_limit'] is not None:
            # Limit grid data to UI slider duration for plotting
            plot_grid_limit = results['grid_limit'][:max_plot_points]
            plot_adjusted_grid_limit = results['adjusted_grid_limit_before_margin'][:max_plot_points] if 'adjusted_grid_limit_before_margin' in results else None
            plot_original_grid_limit = results['original_grid_limit'][:max_plot_points] if 'original_grid_limit' in results else None
            grid_time = np.arange(len(plot_grid_limit)) / 60
            
            # 1. Grid Limit (red dashed line)
            if 'adjusted_grid_limit_before_margin' in results:
                ax1.step(grid_time, plot_adjusted_grid_limit, where='post', color='red', linestyle='--', alpha=0.7, label='Grid Limit')
            
            # 2. Battery Effects (charging and discharging)
            if show_battery_effects:
                pv_charge = results.get('pv_battery_charge_curve') if results.get('pv_battery_applied', False) else None
                grid_discharge = results.get('grid_battery_discharge_curve') if results.get('grid_battery_applied', False) else None
                v2g_discharge = results.get('v2g_discharge_curve') if results.get('v2g_applied', False) else None
                grid_charge = results.get('grid_battery_charge_curve') if results.get('grid_battery_applied', False) else None
                
                # Get PV direct system support and battery discharge curves
                pv_direct_support = results.get('pv_direct_support_curve') if results.get('pv_battery_applied', False) else None
                pv_battery_discharge = results.get('pv_battery_discharge_curve') if results.get('pv_battery_applied', False) else None
                
                # PV direct system support (lightgreen shading - PV discharge during the day)
                if pv_direct_support is not None and np.any(pv_direct_support > 0):
                    plot_pv_direct_support = pv_direct_support[:max_plot_points]
                    ax1.fill_between(grid_time, plot_original_grid_limit, plot_original_grid_limit + plot_pv_direct_support, 
                                   color='lightgreen', alpha=0.4, label='PV Direct System Support')
                
                # 3. Battery Charging Effects (separate PV and Grid charging)
                if pv_charge is not None and np.any(pv_charge > 0):
                    # PV battery charging (orange shading - above red line to show extra energy)
                    plot_pv_charge = pv_charge[:max_plot_points]
                    ax1.fill_between(grid_time, plot_original_grid_limit, plot_original_grid_limit + plot_pv_charge, 
                                   color='orange', alpha=0.4, label='PV Battery Charging (Extra Energy)')
                
                if grid_charge is not None and np.any(grid_charge > 0):
                    # Grid battery charging (lightcoral shading - reduces grid capacity)
                    plot_grid_charge = grid_charge[:max_plot_points]
                    ax1.fill_between(grid_time, plot_original_grid_limit - plot_grid_charge, plot_original_grid_limit, 
                                   color='lightcoral', alpha=0.3, label='Grid Battery Charging (Reduces Capacity)')
                
                # Combine all battery discharging effects (PV battery + grid battery + V2G discharge)
                has_battery_discharge_effects = (pv_battery_discharge is not None and np.any(pv_battery_discharge > 0)) or (grid_discharge is not None and np.any(grid_discharge > 0)) or (v2g_discharge is not None and np.any(v2g_discharge > 0))
                if has_battery_discharge_effects:
                    combined_battery_discharge = np.zeros_like(plot_original_grid_limit)
                    if pv_battery_discharge is not None:
                        plot_pv_battery_discharge = pv_battery_discharge[:max_plot_points]
                        combined_battery_discharge += plot_pv_battery_discharge
                    if grid_discharge is not None:
                        plot_grid_discharge = grid_discharge[:max_plot_points]
                        combined_battery_discharge += plot_grid_discharge
                    if v2g_discharge is not None:
                        plot_v2g_discharge = v2g_discharge[:max_plot_points]
                        combined_battery_discharge += plot_v2g_discharge
                    
                    # Plot combined battery discharging effects (light blue color for battery discharge during peak)
                    ax1.fill_between(grid_time, plot_original_grid_limit, plot_original_grid_limit + combined_battery_discharge, 
                                   color='#87CEEB', alpha=0.4, label='Battery Discharge Effects (Combined)')
            
            # 4. Grid Limit with margin (orange dashed line)
            safety_percentage = st.session_state.get('available_load_fraction', 0.8) * 100
            ax1.step(grid_time, plot_grid_limit, where='post', color='orange', linestyle='--', alpha=0.9, label=f'Grid Limit ({safety_percentage:.0f}% safety margin)')
        
        # 5. EV Load (blue line)
        ax1.plot(time_hours, plot_load_curve, 'b-', linewidth=2, label='EV Load')
        
        # Ensure y-axis shows both lines
        if results['grid_limit'] is not None:
            min_y = 0
            max_y = max(np.max(plot_grid_limit), np.max(plot_original_grid_limit), np.max(plot_load_curve)) * 1.1
            ax1.set_ylim(min_y, max_y)
        
        if show_legend:
            ax1.legend(loc='lower left', fontsize=12, frameon=True)
        
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Power (kW)')
        mode_text = "Grid Constrained" if results['grid_mode'] == "Grid Constrained" else "Reference Only"
        ax1.set_title(f'EV Charging Load vs Grid Capacity ({mode_text})')
        ax1.grid(True, alpha=0.3)
        
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
            ax2.fill_between(grid_time, 0, available_capacity_before_margin, color='lightgreen', alpha=0.7, label='Available Grid Capacity')
            ax2.plot(grid_time, available_capacity_before_margin, 'g-', linewidth=2, label='_nolegend_')
            
            # Add red dashed line to show the 20% margin that shouldn't be used
            if show_safety_margin:
                margin_capacity = plot_adjusted_grid_limit - plot_grid_limit
                ax2.plot(grid_time, margin_capacity, 'r--', linewidth=2, alpha=0.8, label='20% Safety Margin')
            
            # Add average line if requested
            if show_average_line and np.any(available_capacity_before_margin > 0):
                avg_available = np.mean(available_capacity_before_margin)
                ax2.axhline(y=avg_available, color='blue', linestyle='-', alpha=0.7, linewidth=2, label=f'Average ({avg_available:.1f} kW)')
            
            # Add horizontal line at zero
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Set y-axis limits for bottom plot
            max_available = np.max(available_capacity_before_margin) * 1.1 if np.max(available_capacity_before_margin) > 0 else 100
            ax2.set_ylim(0, max_available)
            
            if show_legend:
                ax2.legend(loc='upper right', fontsize=12, frameon=True)
        
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Available Capacity (kW)')
        ax2.set_title('Available Grid Capacity')
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Display statistics under the graph
        st.header("📊 Performance Metrics")
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            st.metric("Peak Load", f"{np.max(results['load_curve']):.2f} kW", 
                     help="Maximum instantaneous power demand during the simulation")
            
            # Calculate average load only for periods with active charging
            active_periods = results['load_curve'] > 0
            if np.any(active_periods):
                active_average = np.mean(results['load_curve'][active_periods])
                st.metric("Average Load (Active Periods)", f"{active_average:.2f} kW",
                         help="Average power demand only during periods when EVs are actively charging")
            else:
                st.metric("Average Load (Active Periods)", "0.00 kW",
                         help="Average power demand only during periods when EVs are actively charging")
            
            st.metric("Total EVs", f"{results['total_evs']} EVs",
                     help="Total number of EVs in the simulation")
            
            # Calculate max simultaneous EVs
            if np.any(active_periods):
                ev_charging_rate = st.session_state.dynamic_ev.get('AC', 11.0)  # Default 11 kW
                max_simultaneous_evs = int(np.max(results['load_curve']) / ev_charging_rate)
                st.metric("Max Simultaneous EVs", f"{max_simultaneous_evs} EVs",
                         help="Maximum number of EVs charging at the same time (based on peak load)")
            else:
                st.metric("Max Simultaneous EVs", "0 EVs",
                         help="Maximum number of EVs charging at the same time (based on peak load)")
        
        with col_stats2:
            st.metric("Total Energy", f"{np.sum(results['load_curve']) / 60:.2f} kWh",
                     help="Total energy consumed by all EVs during the simulation (kWh)")
            
            # Calculate peak hours analysis
            if np.any(active_periods):
                peak_load = np.max(results['load_curve'])
                peak_threshold = peak_load * 0.9  # 90% of peak load
                peak_hours = results['load_curve'] >= peak_threshold
                peak_hour_count = np.sum(peak_hours)
                peak_start = np.where(peak_hours)[0][0] / 60 if np.any(peak_hours) else 0
                peak_end = np.where(peak_hours)[0][-1] / 60 if np.any(peak_hours) else 0
                st.metric("Peak Hours (90% of max)", f"{peak_start:.1f}-{peak_end:.1f}h",
                         help="Time range when load is ≥90% of peak load")
            else:
                st.metric("Peak Hours (90% of max)", "No peak",
                         help="Time range when load is ≥90% of peak load")
            
            # Calculate average energy per EV
            if results['total_evs'] > 0:
                avg_energy_per_ev = (np.sum(results['load_curve']) / 60) / results['total_evs']
                st.metric("Avg Energy per EV", f"{avg_energy_per_ev:.1f} kWh",
                         help="Average energy consumed per EV during the simulation")
            else:
                st.metric("Avg Energy per EV", "0.0 kWh",
                         help="Average energy consumed per EV during the simulation")
            
            # Calculate charging session duration
            if results['total_evs'] > 0:
                # Calculate based on energy consumed and charging rate
                total_energy_kwh = np.sum(results['load_curve']) / 60  # Convert from kW-minutes to kWh
                avg_energy_per_ev = total_energy_kwh / results['total_evs']
                ev_charging_rate = st.session_state.dynamic_ev.get('AC', 11.0)  # Default 11 kW
                
                # Average session duration = average energy per EV / charging rate
                avg_session_duration = avg_energy_per_ev / ev_charging_rate
                st.metric("Avg Charging Session", f"{avg_session_duration:.1f}h",
                         help="Average charging session duration per EV (based on energy consumed)")
            else:
                st.metric("Avg Charging Session", "0.0h",
                         help="Average charging session duration per EV (based on energy consumed)")
        
        with col_stats3:
            if results['grid_limit'] is not None:
                # Calculate available capacity (same as second graph)
                available_capacity_before_margin = results['adjusted_grid_limit_before_margin'] - results['load_curve']
                available_capacity_before_margin = np.maximum(available_capacity_before_margin, 0)  # Ensure non-negative
                
                max_available_capacity = np.max(available_capacity_before_margin)
                min_available_capacity = np.min(available_capacity_before_margin)
                avg_available_capacity = np.mean(available_capacity_before_margin)
                
                st.metric("Max Available Capacity", f"{max_available_capacity:.1f} kW",
                         help="Maximum available grid capacity during the simulation (from second graph)")
                st.metric("Min Available Capacity", f"{min_available_capacity:.1f} kW",
                         help="Minimum available grid capacity during the simulation (from second graph)")
                st.metric("Avg Available Capacity", f"{avg_available_capacity:.1f} kW",
                         help="Average available grid capacity during the simulation (from second graph)")
                
                # Calculate grid utilization factor (avg vs max available capacity)
                if max_available_capacity > 0:
                    grid_utilization = (avg_available_capacity / max_available_capacity) * 100
                    st.metric("Grid Utilization Factor", f"{grid_utilization:.1f}%",
                             help="Ratio of average to max available capacity (higher = more consistent grid usage)")
                else:
                    st.metric("Grid Utilization Factor", "0.0%",
                             help="Ratio of average to max available capacity (higher = more consistent grid usage)")
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