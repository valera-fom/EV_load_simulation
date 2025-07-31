import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sim_setup import SimulationSetup
    from EV import EV_MODELS
    from charger import CHARGER_MODELS
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Initialize session state for persistent values
if 'load_testing_config' not in st.session_state:
    st.session_state.load_testing_config = {
        'ev_counts': {},
        'charger_counts': {'ac': 1, 'dc': 0, 'fast_dc': 0},
        'arrival_time_mean': 12.0,
        'arrival_time_span': 4.0,
        'sim_duration': 24,
        'selected_dataset': 'df1.csv',
        'selected_date': None,
        'grid_constraint_mode': 'Grid Constrained'
    }

# Title and description
st.title("⚡ EV Load Testing")
st.markdown("Test EV charging load patterns with predefined vehicles and chargers")

# Reset button
if st.sidebar.button("⭮ Reset", type="primary", help="Reset all configuration to default values"):
    st.session_state.load_testing_config = {
        'ev_counts': {},
        'charger_counts': {'ac': 1, 'dc': 0, 'fast_dc': 0},
        'arrival_time_mean': 12.0,
        'arrival_time_span': 4.0,
        'sim_duration': 24,
        'selected_dataset': 'df1.csv',
        'selected_date': None,
        'grid_constraint_mode': 'Grid Constrained'
    }
    st.rerun()

# Sidebar for configuration
st.sidebar.header("📊 Simulation Configuration")

def clear_simulation_flag():
    st.session_state.simulation_run = False

# EV Counts
with st.sidebar.expander("🚗 EV Counts", expanded=False):
    # Group 1: Passenger BEVs
    st.write("**🚙 Passenger BEVs**")
    ev_counts = {}
    passenger_bevs = ["bev_small", "bev_medium", "bev_big"]
    for ev_type in passenger_bevs:
        specs = EV_MODELS[ev_type]
        current_value = st.session_state.load_testing_config['ev_counts'].get(ev_type, 1 if ev_type == "bev_small" else 0)
        help_text = f"Number of {specs['name']} vehicles\n\nCharacteristics:\n• Battery: {specs['capacity']} kWh\n• AC Charging: {specs['AC']} kW\n• DC Charging: {specs['DC']} kW"
        value = st.number_input(
            f"{specs['name']} ({ev_type})",
            min_value=0,
            max_value=50,
            value=current_value,
            key=f"ev_count_{ev_type}",
            help=help_text,
            on_change=clear_simulation_flag
        )
        ev_counts[ev_type] = value

    # Group 2: Passenger PHEVs
    st.write("**🔋 Passenger PHEVs**")
    passenger_phevs = ["phev_small", "phev_medium", "phev_big"]
    for ev_type in passenger_phevs:
        specs = EV_MODELS[ev_type]
        current_value = st.session_state.load_testing_config['ev_counts'].get(ev_type, 0)
        dc_charging = "Not available" if specs['DC'] is None else f"{specs['DC']} kW"
        help_text = f"Number of {specs['name']} vehicles\n\nCharacteristics:\n• Battery: {specs['capacity']} kWh\n• AC Charging: {specs['AC']} kW\n• DC Charging: {dc_charging}"
        value = st.number_input(
            f"{specs['name']} ({ev_type})",
            min_value=0,
            max_value=50,
            value=current_value,
            key=f"ev_count_{ev_type}",
            help=help_text,
            on_change=clear_simulation_flag
        )
        ev_counts[ev_type] = value

    # Group 3: Commercial BEVs
    st.write("**🚐 Commercial BEVs**")
    commercial_bevs = ["commercial_bev_small", "commercial_bev_medium", "commercial_bev_big"]
    for ev_type in commercial_bevs:
        specs = EV_MODELS[ev_type]
        current_value = st.session_state.load_testing_config['ev_counts'].get(ev_type, 0)
        help_text = f"Number of {specs['name']} vehicles\n\nCharacteristics:\n• Battery: {specs['capacity']} kWh\n• AC Charging: {specs['AC']} kW\n• DC Charging: {specs['DC']} kW"
        value = st.number_input(
            f"{specs['name']} ({ev_type})",
            min_value=0,
            max_value=50,
            value=current_value,
            key=f"ev_count_{ev_type}",
            help=help_text,
            on_change=clear_simulation_flag
        )
        ev_counts[ev_type] = value

    # Group 4: Heavy BEVs
    st.write("**🚛 Heavy BEVs**")
    heavy_bevs = ["heavy_bev_small", "heavy_bev_medium", "heavy_bev_big"]
    for ev_type in heavy_bevs:
        specs = EV_MODELS[ev_type]
        current_value = st.session_state.load_testing_config['ev_counts'].get(ev_type, 0)
        ac_charging = "Not available" if specs['AC'] is None else f"{specs['AC']} kW"
        help_text = f"Number of {specs['name']} vehicles\n\nCharacteristics:\n• Battery: {specs['capacity']} kWh\n• AC Charging: {ac_charging}\n• DC Charging: {specs['DC']} kW"
        value = st.number_input(
            f"{specs['name']} ({ev_type})",
            min_value=0,
            max_value=50,
            value=current_value,
            key=f"ev_count_{ev_type}",
            help=help_text,
            on_change=clear_simulation_flag
        )
        ev_counts[ev_type] = value

# Charger Counts
with st.sidebar.expander("🔌 Charger Counts", expanded=False):
    charger_counts = {}
    for charger_type, specs in CHARGER_MODELS.items():
        current_value = st.session_state.load_testing_config['charger_counts'].get(charger_type, 0)
        value = st.number_input(
            f"{charger_type.upper()} Chargers ({specs['power_kW']} kW)",
            min_value=0,
            max_value=20,
            value=current_value,
            key=f"charger_count_{charger_type}",
            help=f"Number of {charger_type.upper()} chargers",
            on_change=clear_simulation_flag
        )
        charger_counts[charger_type] = value

# Time Configuration
with st.sidebar.expander("⏰ Time Configuration", expanded=False):
    arrival_time_mean = st.slider(
        "Mean Arrival Time (hours)",
        min_value=0.0,
        max_value=24.0,
        value=st.session_state.load_testing_config['arrival_time_mean'],
        step=0.5,
        key="arrival_time_mean_slider",
        help="Center of arrival time distribution",
        on_change=clear_simulation_flag
    )

    arrival_time_span = st.slider(
        "Arrival Time Span (2σ in hours)",
        min_value=0.5,
        max_value=8.0,
        value=st.session_state.load_testing_config['arrival_time_span'],
        step=0.5,
        key="arrival_time_span_slider",
        help="Spread of arrival times (2 standard deviations)",
        on_change=clear_simulation_flag
    )

    sim_duration = st.slider(
        "Simulation Duration (hours)",
        min_value=6,
        max_value=48,
        value=st.session_state.load_testing_config['sim_duration'],
        step=1,
        key="sim_duration_slider",
        help="Total simulation time",
        on_change=clear_simulation_flag
    )

# Dataset selection
with st.sidebar.expander("📁 Dataset Selection", expanded=False):
    dataset_folder = "datasets"
    dataset_files = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]
    dataset_choice = st.selectbox("Choose dataset", dataset_files, 
                                 index=dataset_files.index(st.session_state.load_testing_config['selected_dataset']) if st.session_state.load_testing_config['selected_dataset'] in dataset_files else 0,
                                 on_change=clear_simulation_flag)
    st.session_state.load_testing_config['selected_dataset'] = dataset_choice

    # Add slider for available load fraction
    available_load_fraction = st.slider(
        "Available Load Fraction",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="Fraction of the grid limit that can be used (e.g. 0.8 = 80%)"
    )

    # Load selected dataset
    dataset_path = os.path.join(dataset_folder, dataset_choice)
    df = pd.read_csv(dataset_path, header=None)

    # Always treat first column as date, second as max_power (in memory only)
    df = df.rename(columns={0: 'date', 1: 'max_power'})

    # Parse the date column robustly (in memory)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Date selection
    st.write("**📅 Day Selection**")
    date_min = pd.Timestamp('2023-01-01')
    date_max = pd.Timestamp('2024-12-31')
    date_options = df['date'][(df['date'] >= date_min) & (df['date'] <= date_max)]
    if len(date_options) == 0:
        st.warning("No valid dates in this dataset!")
        selected_date = None
        grid_power_limit = 0
    else:
        # Use stored date or default to first available date
        stored_date = st.session_state.load_testing_config['selected_date']
        default_date = stored_date if stored_date and stored_date in date_options.values else date_options.iloc[0].date()
        
        selected_date = st.date_input(
            "Select day",
            value=default_date,
            min_value=date_min.date(),
            max_value=date_max.date(),
            on_change=clear_simulation_flag
        )
        st.session_state.load_testing_config['selected_date'] = selected_date
        
        # Find the row for the selected date
        if pd.api.types.is_datetime64_any_dtype(df['date']):
            row = df[df['date'].dt.date == selected_date]
        else:
            row = df[df['date'] == pd.Timestamp(selected_date)]
        if not row.empty:
            grid_power_limit = float(row.iloc[0]['max_power'])
        else:
            grid_power_limit = 0
            st.warning("Selected day not found in dataset!")

    # Grid constraint mode toggle
    st.write("**⚡ Grid Constraint Mode**")
    grid_constraint_mode = st.radio(
        "Choose mode:",
        ["Grid Constrained", "Reference Only"],
        index=0 if st.session_state.load_testing_config['grid_constraint_mode'] == "Grid Constrained" else 1,
        help="Grid Constrained: Load cannot exceed grid limit. Reference Only: Show actual load with grid limit as reference line.",
        on_change=clear_simulation_flag
    )
    st.session_state.load_testing_config['grid_constraint_mode'] = grid_constraint_mode

# Extract grid power profile for the selected day (column 2, index 2)
grid_profile = None
if selected_date is not None:
    # Filter for the selected day
    day_rows = df[df['date'].dt.date == selected_date]
    if not day_rows.empty:
        # Get the 3rd column (index 2) as the profile (should be one value per 15 min)
        grid_profile_15min = day_rows.iloc[:, 2].astype(float).to_numpy() * available_load_fraction
        # Expand to per-minute steps (repeat each value 15 times)
        grid_profile = np.repeat(grid_profile_15min, 15).astype(float)
        
        # Extend profile for longer simulations by using actual next days
        sim_duration_min = sim_duration * 60
        daily_minutes = 24 * 60  # 24 hours in minutes
        
        if len(grid_profile) < sim_duration_min:
            # Calculate how many days we need
            num_days_needed = sim_duration_min // daily_minutes + (1 if sim_duration_min % daily_minutes > 0 else 0)
            
            # Load multiple days from the dataset
            extended_profile = []
            current_date = selected_date
            
            for day in range(num_days_needed):
                # Get data for current date
                day_data = df[df['date'].dt.date == current_date]
                
                if not day_data.empty:
                    # Extract and scale power values for this day
                    day_power_values = day_data.iloc[:, 2].astype(float).values * available_load_fraction
                    # Expand to per-minute steps
                    day_profile = np.repeat(day_power_values, 15).astype(float)
                    extended_profile.extend(day_profile)
                else:
                    # If no data for this day, use the previous day's scaled profile
                    extended_profile.extend(grid_profile)
                
                # Move to next day
                current_date = current_date + pd.Timedelta(days=1)
            
            # Truncate to exact simulation duration
            grid_profile = np.array(extended_profile[:sim_duration_min])
        else:
            # If the profile is longer, truncate
            grid_profile = grid_profile[:sim_duration_min]
    else:
        grid_profile = None

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📈 Simulation Results")
    
    # Check if we have any EVs and chargers
    total_evs = sum(ev_counts.values())
    total_chargers = sum(charger_counts.values())
    
    if total_evs == 0:
        st.warning("⚠️ Please add at least one EV to run the simulation.")
    elif total_chargers == 0:
        st.warning("⚠️ Please add at least one charger to run the simulation.")
    else:
        # Update session state only when simulation is run
        if st.button("🚀 Run Simulation", type="primary"):
            # Update session state with current values
            st.session_state.load_testing_config['ev_counts'] = ev_counts
            st.session_state.load_testing_config['charger_counts'] = charger_counts
            st.session_state.load_testing_config['arrival_time_mean'] = arrival_time_mean
            st.session_state.load_testing_config['arrival_time_span'] = arrival_time_span
            st.session_state.load_testing_config['sim_duration'] = sim_duration
            st.session_state.load_testing_config['selected_dataset'] = dataset_choice
            st.session_state.load_testing_config['selected_date'] = selected_date
            st.session_state.load_testing_config['grid_constraint_mode'] = grid_constraint_mode
            
            # Set simulation run flag
            st.session_state.simulation_run = True
            st.rerun()
        
        # Only show simulation results if simulation was run
        if 'simulation_run' in st.session_state and st.session_state.simulation_run:
            st.success("✅ Simulation completed!")
            
            # Create simulation setup
            arrival_time_mean_min = arrival_time_mean * 60
            arrival_time_span_min = arrival_time_span * 60
            sim_duration_min = sim_duration * 60
            
            # Set grid power limit based on mode
            if grid_constraint_mode == "Grid Constrained":
                grid_power_input = grid_profile if grid_profile is not None else grid_power_limit
            else:
                grid_power_input = None
            
            sim = SimulationSetup(
                ev_counts=ev_counts,
                charger_counts=charger_counts,
                sim_duration=sim_duration_min,
                arrival_time_mean=arrival_time_mean_min,
                arrival_time_span=arrival_time_span_min,
                grid_power_limit=grid_power_input
            )
            
            # Run simulation
            load_curve = sim.run_simulation()
            
            # Create plot
            time_hours = np.arange(len(load_curve)) / 60
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(time_hours, load_curve, 'b-', linewidth=2, label='EV Load')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Total Load (kW)')

            # Add grid limit lines if applicable
            if grid_profile is not None:
                grid_time = np.arange(len(grid_profile))/60  # Convert to hours
                # Plot allowed (scaled) grid limit
                ax.step(grid_time, grid_profile, where='post', color='orange', linestyle='--', alpha=0.9, label=f'Allowed Grid Limit ({available_load_fraction*100:.0f}%)')
                # Plot true (unscaled) grid limit
                true_grid_profile_15min = day_rows.iloc[:, 2].astype(float).to_numpy()
                true_grid_profile = np.repeat(true_grid_profile_15min, 15).astype(float)
                if len(true_grid_profile) < len(grid_profile):
                    extended_true = []
                    current_date = selected_date
                    while len(extended_true) < len(grid_profile):
                        day_data = df[df['date'].dt.date == current_date]
                        if not day_data.empty:
                            vals = day_data.iloc[:, 2].values
                            extended_true.extend(np.repeat(vals, 15).astype(float))
                        else:
                            extended_true.extend(true_grid_profile)
                        current_date = current_date + pd.Timedelta(days=1)
                    true_grid_profile = np.array(extended_true[:len(grid_profile)])
                else:
                    true_grid_profile = true_grid_profile[:len(grid_profile)]
                ax.step(grid_time, true_grid_profile, where='post', color='red', linestyle='--', alpha=0.7, label='True Grid Limit (100%)')
                # Ensure y-axis shows both lines
                min_y = 0
                max_y = max(np.max(true_grid_profile), np.max(grid_profile), np.max(load_curve)) * 1.1
                ax.set_ylim(min_y, max_y)
                ax.legend(loc='upper right', fontsize=12, frameon=True)
            elif grid_power_limit is not None:
                ax.axhline(y=grid_power_limit, color='orange', linestyle='--', alpha=0.9, label=f'Allowed Grid Limit ({available_load_fraction*100:.0f}%)')
                ax.axhline(y=grid_power_limit/available_load_fraction, color='red', linestyle='--', alpha=0.7, label='True Grid Limit (100%)')
                min_y = 0
                max_y = max(grid_power_limit, grid_power_limit/available_load_fraction, np.max(load_curve)) * 1.1
                ax.set_ylim(min_y, max_y)
                ax.legend(loc='upper right', fontsize=12, frameon=True)
            
            mode_text = "Grid Constrained" if grid_constraint_mode == "Grid Constrained" else "Reference Only"
            ax.set_title(f'EV Charging Load Curve ({mode_text})\n(Arrival: {arrival_time_mean:.1f}h ± {arrival_time_span:.1f}h)')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Display statistics
            st.subheader("📊 Statistics")
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                st.metric("Total EVs", sum(ev_counts.values()))
                st.metric("Total Chargers", sum(charger_counts.values()))
                st.metric("Peak Load", f"{np.max(load_curve):.2f} kW")
                st.metric("Average Load", f"{np.mean(load_curve):.2f} kW")
            with col_stats2:
                st.metric("Total Energy", f"{np.sum(load_curve) / 60:.2f} kWh")
                st.metric("Simulation Time", f"{sim_duration} hours")
                st.metric("Average Queue Time", f"{getattr(sim, 'avg_queue_time', 0):.2f} min")
                st.metric("EVs That Waited", f"{getattr(sim, 'num_queued_evs', 0)}")
            
            # Display rejection notifications
            if hasattr(sim, 'rejected_evs') and sim.rejected_evs:
                st.subheader("⚠️ Rejection Notifications")
                dc_only_rejected = [ev for ev, reason in sim.rejected_evs if "DC charger not available" in reason]
                ac_only_rejected = [ev for ev, reason in sim.rejected_evs if "AC charger not available" in reason]
                other_rejected = [ev for ev, reason in sim.rejected_evs if "DC charger not available" not in reason and "AC charger not available" not in reason]
                
                if dc_only_rejected:
                    st.warning(f"🔌 DC-only vehicles rejected ({len(dc_only_rejected)}):")
                    for ev in dc_only_rejected:
                        st.write(f"  - {ev.name} (max_power: {ev.max_charging_power} kW)")
                    st.info("💡 Add DC chargers to accommodate these vehicles")
                
                if ac_only_rejected:
                    st.warning(f"🔌 AC-only vehicles rejected ({len(ac_only_rejected)}):")
                    for ev in ac_only_rejected:
                        st.write(f"  - {ev.name} (max_power: {ev.max_charging_power} kW)")
                    st.info("💡 Add AC chargers to accommodate these vehicles")
                
                if other_rejected:
                    st.error(f"❓ Other rejected vehicles ({len(other_rejected)}):")
                    for ev in other_rejected:
                        st.write(f"  - {ev.name} (max_power: {ev.max_charging_power} kW)")
            else:
                st.success("✅ All EVs scheduled successfully!")

with col2:
    st.subheader("ℹ️ Information")
    
    # Only show configuration summary if simulation was run
    if 'simulation_run' in st.session_state and st.session_state.simulation_run:
        # EV Summary
        st.write("**🚗 EVs Configured:**")
        for ev_type, count in ev_counts.items():
            if count > 0:
                specs = EV_MODELS[ev_type]
                st.write(f"• {specs['name']}: {count}")
        
        # Charger Summary
        st.write("**🔌 Chargers Configured:**")
        for charger_type, count in charger_counts.items():
            if count > 0:
                specs = CHARGER_MODELS[charger_type]
                st.write(f"• {charger_type.upper()}: {count} ({specs['power_kW']} kW)")
        
        # Time Summary
        st.write("**⏰ Time Configuration:**")
        st.write(f"• Arrival: {arrival_time_mean:.1f}h ± {arrival_time_span:.1f}h")
        st.write(f"• Duration: {sim_duration}h")
        
        # Grid Summary
        st.write("**⚡ Grid Configuration:**")
        st.write(f"• Mode: {grid_constraint_mode}")
        if grid_profile is not None:
            st.write(f"• Profile: {dataset_choice}")
            st.write(f"• Date: {selected_date}")
        elif grid_power_limit is not None:
            st.write(f"• Limit: {grid_power_limit:.2f} kW")
    else:
        st.write("**Click 'Run Simulation' to see configuration summary**")

# Footer
st.markdown("---")
st.markdown("**EV Charging Simulation Tool** - Built with Streamlit and SimPy") 