#!/usr/bin/env python3
"""
Dynamic Capacity Optimizer UI for Streamlit
Replaces Gradient Optimizer with Dynamic Car Addition during Simulation
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from EV import EV, EV_MODELS
from charger import CHARGER_MODELS
from sim_setup import SimulationSetup

def create_dynamic_capacity_optimizer_ui():
    """Create the Dynamic Capacity Optimizer UI."""
    st.subheader("🎯 Dynamic Capacity Optimizer")
    st.write("Find optimal TOU adoption percentages by dynamically adding cars based on available capacity.")
    
    # Get TOU periods from session state
    tou_periods = st.session_state.optimization_strategy.get('time_of_use_periods', [])
    if not tou_periods or (hasattr(tou_periods, '__len__') and len(tou_periods) == 0):
        st.warning("🔔 Please configure Time of Use periods first")
        return
    
    # Optimization parameters
    col1, col2 = st.columns(2)
    with col1:
        time_step = st.selectbox(
            "Simulation Step Size",
            options=[1, 5, 15],
            index=1,  # Default to 5 minutes
            help="Time step for simulation (minutes)"
        )
    
    with col2:
        max_iterations = st.slider(
            "Maximum Iterations",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of simulation iterations for accuracy"
        )
    
    # Run optimization button
    if st.button("🚀 Run Dynamic Capacity Optimization", type="primary"):
        with st.spinner("Running dynamic capacity optimization..."):
            try:
                results = run_dynamic_capacity_optimization(time_step, max_iterations)
                display_optimization_results(results, time_step, max_iterations)
            except Exception as e:
                st.error(f"❌ Optimization error: {e}")
                st.exception(e)
    
    # Display stored optimization results if available
    if st.session_state.get('dynamic_optimization_completed', False):
        stored_results = st.session_state.get('dynamic_optimization_results', {})
        stored_total_cars = stored_results.get('total_cars', 0)
        
        if stored_total_cars > 0:
            st.write("**📊 Dynamic Capacity Optimization Results:**")
            st.write(f"**Optimal EVs: {stored_total_cars}**")
            
            # Show TOU percentages in table format
            st.write("**TOU Distribution Applied:**")
            tou_data = []
            for period_name, points in stored_results.get('period_results', {}).items():
                total_points = sum(stored_results.get('period_results', {}).values())
                percentage = (points / total_points * 100) if total_points > 0 else 0
                tou_data.append({
                    'Period': period_name,
                    'Percentage': f"{percentage:.1f}%"
                })
            
            if tou_data:
                tou_df = pd.DataFrame(tou_data)
                st.dataframe(tou_df, use_container_width=True)
            
            st.success("✅ Optimization results have been applied to the configuration above.")
            
            # Display stored simulation curves
            if 'margin_curve' in stored_results:
                st.subheader("📈 Simulation Curves")
                
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
                
                # Plot margin curve and final load
                step_duration_hours = stored_results.get('time_step', 5) / 60.0
                margin_curve = np.array(stored_results['margin_curve'])
                final_load = np.array(stored_results['final_load'])
                cars_added_per_step = np.array(stored_results['cars_added_per_step'])
                active_cars_per_step = np.array(stored_results['active_cars_per_step'])
                
                hours = np.arange(len(margin_curve)) * step_duration_hours
                ax1.plot(hours, margin_curve, label='Margin Curve', color='blue')
                ax1.plot(hours, final_load, label='Final Load', color='red')
                ax1.set_title('Margin Curve vs Final Load')
                ax1.set_xlabel('Time (Hours)')
                ax1.set_ylabel('Power (kW)')
                ax1.legend()
                ax1.grid(True)
                
                # Plot cars added per step
                ax2.plot(hours, cars_added_per_step, label='Cars Added', color='green', marker='o', markersize=3)
                ax2.set_title('Cars Added per Time Step')
                ax2.set_xlabel('Time (Hours)')
                ax2.set_ylabel('Number of Cars')
                ax2.legend()
                ax2.grid(True)
                
                # Plot active cars per step
                ax3.plot(hours, active_cars_per_step, label='Active Cars', color='orange', linewidth=2)
                ax3.set_title('Active Cars per Time Step')
                ax3.set_xlabel('Time (Hours)')
                ax3.set_ylabel('Number of Active Cars')
                ax3.legend()
                ax3.grid(True)
                
                st.pyplot(fig)

def run_dynamic_capacity_optimization(time_step, max_iterations):
    """Run the dynamic capacity optimization with multiple iterations."""
    # Get configuration from session state
    ev_config = st.session_state.dynamic_ev
    charger_config = st.session_state.charger_config
    tou_periods = st.session_state.optimization_strategy.get('time_of_use_periods', [])
    active_strategies = st.session_state.get('active_strategies', [])
    grid_mode = st.session_state.get('grid_mode', 'Reference Only')
    available_load_fraction = st.session_state.get('available_load_fraction', 0.8)
    data_source = st.session_state.get('data_source', 'Real Dataset')
    power_values = st.session_state.get('power_values', None)
    
    # Debug output for TOU periods
    print(f"🔍 Dynamic Optimizer Debug - {len(tou_periods)} TOU periods:")
    for period in tou_periods:
        print(f"  {period['name']}: {period['start']}-{period['end']}h, {period['adoption']:.1f}%")
    
    # Handle data loading/generation (EXACTLY like main simulation)
    if data_source == "Synthetic Generation":
        if 'synthetic_load_curve' in st.session_state and st.session_state.synthetic_load_curve is not None and len(st.session_state.synthetic_load_curve) > 0:
            power_values = st.session_state.synthetic_load_curve
            st.session_state.power_values = power_values
        elif power_values is None or (hasattr(power_values, '__len__') and len(power_values) == 0):
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
                st.success(f"✅ Synthetic load curve generated automatically for optimization!")
            except Exception as e:
                st.error(f"❌ Error generating synthetic data: {e}")
                st.write("Please ensure the portable_models directory contains the trained models.")
                raise
    elif data_source == "Real Dataset":
        # For Real Dataset, try to use existing data or fall back to synthetic
        if power_values is None or (hasattr(power_values, '__len__') and len(power_values) == 0):
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
                st.success(f"✅ Synthetic load curve generated automatically for optimization!")
            except Exception as e:
                st.error(f"❌ Error generating synthetic data: {e}")
                st.write("Please ensure the portable_models directory contains the trained models.")
                raise
    
    # Check if we have valid data
    if power_values is None or (hasattr(power_values, '__len__') and len(power_values) == 0):
        st.warning("⚠️ No dataset available. Please upload an Excel file with datetime and taken load data, or switch to 'Synthetic Generation' to use synthetic data.")
        st.info("💡 You can also use the 'Generate Synthetic Curve' button to create synthetic data for optimization.")
        return
    
    # Initialize iteration tracking
    iteration_results = []
    current_total_evs = None  # Will be updated in each iteration
    
        # Run multiple iterations
    for iteration in range(max_iterations):
        print(f"🔄 Running iteration {iteration + 1}/{max_iterations}")
        
        # Set random seed for consistent battery effects between all optimizers
        np.random.seed(st.session_state.random_seed)
        
        # For dynamic optimizer, we need to account for battery effects like the main simulation
        # Get current configuration
        available_load_fraction = st.session_state.get('available_load_fraction', 0.8)
        
        # Get current number of cars from simulation (EXACTLY like main simulation)
        # For first iteration, use original calculation
        # For subsequent iterations, use the optimized car count from previous iteration
        if iteration == 0:
            if 'time_peaks' in st.session_state and st.session_state.time_peaks:
                current_total_evs = sum(peak.get('quantity', 0) for peak in st.session_state.time_peaks)
            elif 'ev_calculator' in st.session_state and 'total_evs' in st.session_state.ev_calculator:
                current_total_evs = st.session_state.ev_calculator['total_evs']
            else:
                # Use a reasonable default if no peaks or calculator available
                current_total_evs = 32  # Default fallback
        else:
            # Use the optimized car count from previous iteration
            current_total_evs = iteration_results[iteration - 1]['total_cars']
            print(f"📊 Using optimized car count from previous iteration: {current_total_evs}")
        
        # Use the same margin curve extraction as TOU optimizer
        # This ensures we use the correct data source and apply all battery effects consistently
        def extract_margin_curve_with_battery_effects():
            try:
                # Set random seed for consistent battery effects
                np.random.seed(st.session_state.get('random_seed', 42))
                
                # Get current configuration
                active_strategies = st.session_state.get('active_strategies', [])
                
                # Get power values (same logic as TOU optimizer)
                if 'synthetic_load_curve' in st.session_state and st.session_state.synthetic_load_curve is not None:
                    power_values = st.session_state.synthetic_load_curve
                elif 'available_load' in st.session_state and st.session_state.available_load is not None:
                    # Use available_load from Excel file (red line)
                    power_values = st.session_state.available_load
                elif 'power_values' in st.session_state and st.session_state.power_values is not None:
                    # Fallback to power_values if available_load is not available
                    power_values = st.session_state.power_values
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
                
                return margin_curve
            except Exception as e:
                st.error(f"Error extracting margin curve with battery effects: {e}")
                return None
        
        # Extract margin curve with battery effects (same as TOU optimizer)
        margin_curve = extract_margin_curve_with_battery_effects()
        
        if margin_curve is None:
            st.error("❌ No grid data available for dynamic optimization.")
            return None
        
        # Use the extracted margin curve directly (this is the yellow line)
        # Convert to time step intervals
        step_duration = time_step
        num_steps = len(margin_curve) // step_duration
        margin_curve_steps = margin_curve[::step_duration][:num_steps]
        
        # Initialize tracking arrays
        total_cars = 0
        cars_added_per_step = []
        active_cars_per_step = []
        ev_charging_rate = min(ev_config.get('AC', 11), charger_config.get('ac_rate', 11))
        ev_capacity = ev_config.get('capacity', 75)
        
        # Track charging cars: (start_step, end_step, car_id)
        charging_cars = []
        car_id_counter = 0
        
        # Optimization parameters
        max_cars_per_step = 5  # Limit cars added per step for smoothness
        buffer_factor = 0.8  # Use only 80% of available capacity for safety
        look_ahead_steps = 20  # Check future steps for safety
        
        # Calculate margin curve gradient for predictive logic
        margin_gradient = np.gradient(margin_curve_steps)
        
        # Run simulation with dynamic car addition
        current_load = np.zeros(num_steps)
        
        for step in range(num_steps):
            # First, update current load based on cars that are still charging from previous steps
            current_load[step] = sum(ev_charging_rate for start, end, _ in charging_cars if start <= step < end)
            
            # Calculate available power at this step with buffer
            available_power = float(margin_curve_steps[step] - current_load[step]) * buffer_factor
            
            if available_power > 0:
                # 1. GRADIENT-BASED: Adjust based on margin curve trend
                gradient_factor = 1.0 if margin_gradient[step] >= 0 else 0.5
                
                # 2. LOOK-AHEAD: Check if adding cars now will cause future problems
                safe_to_add = True
                if step + look_ahead_steps < num_steps:
                    future_margin = margin_curve_steps[step:step+look_ahead_steps]
                    future_load = current_load[step:step+look_ahead_steps]
                    # Check if adding one car would cause overflow in next N steps
                    test_load = future_load + ev_charging_rate
                    safe_to_add = all(future_margin[i] >= test_load[i] for i in range(look_ahead_steps))
                
                # 3. SMOOTHING: Limit cars added per step
                cars_that_fit = int((available_power / ev_charging_rate) * gradient_factor)
                cars_that_fit = min(cars_that_fit, max_cars_per_step)
                
                # Additional smoothing for early steps to prevent large spikes
                if step < 50:  # First 50 steps
                    cars_that_fit = min(cars_that_fit, 2)  # Max 2 cars per step early on
                
                if cars_that_fit > 0 and safe_to_add:
                    # Add cars to simulation
                    total_cars += cars_that_fit
                    cars_added_per_step.append(cars_that_fit)
                    
                    # Calculate charging duration for these cars (charge to full capacity)
                    # Convert step_duration from minutes to hours for proper calculation
                    step_duration_hours = step_duration / 60.0
                    charging_time_hours = ev_capacity / ev_charging_rate
                    charging_time_steps = max(1, int(charging_time_hours / step_duration_hours))
                    end_step = min(step + charging_time_steps, num_steps)
                    
                    # Add charging cars to tracking
                    for _ in range(cars_that_fit):
                        charging_cars.append((step, end_step, car_id_counter))
                        car_id_counter += 1
                    
                    # Update load for the charging duration (these cars will charge until full)
                    for future_step in range(step, end_step):
                        if future_step < num_steps:
                            current_load[future_step] += cars_that_fit * ev_charging_rate
                else:
                    cars_added_per_step.append(0)
            else:
                cars_added_per_step.append(0)
            
            # Count how many cars are actively charging at this step
            active_cars = sum(1 for start, end, _ in charging_cars if start <= step < end)
            active_cars_per_step.append(active_cars)
        
        # Aggregate results by TOU periods
        period_results = aggregate_by_tou_periods(active_cars_per_step, tou_periods, step_duration)
        
        # Store iteration results
        iteration_result = {
            'iteration': iteration + 1,
            'total_cars': total_cars,
            'cars_added_per_step': cars_added_per_step,
            'active_cars_per_step': active_cars_per_step,
            'period_results': period_results,
            'margin_curve': margin_curve_steps,
            'final_load': current_load,
            'battery_adjusted_evs': current_total_evs
        }
        
        iteration_results.append(iteration_result)
        print(f"✅ Iteration {iteration + 1} completed: {total_cars} cars optimized")
    
    # Return the final iteration results and summary
    final_result = iteration_results[-1]  # Use the last iteration as final result
    final_result['all_iterations'] = iteration_results
    final_result['convergence'] = {
        'total_iterations': max_iterations,
        'final_car_count': final_result['total_cars'],
        'car_count_progression': [r['total_cars'] for r in iteration_results]
    }
    
    return final_result

def aggregate_by_tou_periods(active_cars_per_step, tou_periods, step_duration):
    """Aggregate active cars by TOU periods (car-hours)."""
    period_points = {}
    
    # Group periods by name to aggregate all instances
    period_groups = {}
    for period in tou_periods:
        period_name = period['name']
        if period_name not in period_groups:
            period_groups[period_name] = []
        period_groups[period_name].append(period)
    
    # Calculate car-hours for each period type (summing all instances)
    for period_name, periods in period_groups.items():
        total_car_hours = 0
        
        for period in periods:
            start_hour = period['start']
            end_hour = period['end']
            period_duration_hours = end_hour - start_hour
            
            # Convert hours to step indices
            start_step = int(start_hour * 60 / step_duration)
            end_step = int(end_hour * 60 / step_duration)
            
            # Calculate average number of cars charging during this period instance
            if end_step > start_step:
                avg_cars_in_period = sum(active_cars_per_step[start_step:end_step]) / (end_step - start_step)
            else:
                avg_cars_in_period = 0
            
            # Add car-hours for this period instance
            total_car_hours += avg_cars_in_period * period_duration_hours
        
        # Store total car-hours for this period type
        period_points[period_name] = total_car_hours
    
    return period_points

def display_optimization_results(results, time_step, max_iterations):
    """Display the optimization results."""
    
    # Step 1: Calculate average cars per period from active_cars_per_step
    tou_periods = st.session_state.optimization_strategy.get('time_of_use_periods', [])
    period_averages = {}
    
    if tou_periods:
        # Get active cars per step (orange graph data)
        active_cars_per_step = np.array(results['active_cars_per_step'])
        
        # Calculate step duration in hours
        step_duration_hours = time_step / 60.0
        
        # Group periods by name to avoid duplicates
        unique_periods = {}
        for period in tou_periods:
            period_name = period['name']
            if period_name not in unique_periods:
                unique_periods[period_name] = period
        
        # Calculate averages for each unique period
        for period_name, period in unique_periods.items():
            period_start = period['start']
            period_end = period['end']
            
            # Convert period hours to step indices
            start_step = int(period_start / step_duration_hours)
            end_step = int(period_end / step_duration_hours)
            
            # Get active cars for this period (handle 24-hour wrap-around)
            period_cars = []
            for day in range(2):  # 48 hours = 2 days
                day_start = start_step + (day * 24 * 60 // time_step)
                day_end = end_step + (day * 24 * 60 // time_step)
                
                # Ensure indices are within bounds
                day_start = max(0, min(day_start, len(active_cars_per_step) - 1))
                day_end = max(0, min(day_end, len(active_cars_per_step)))
                
                if day_start < day_end:
                    period_cars.extend(active_cars_per_step[day_start:day_end])
            
            # Calculate average for this period
            if period_cars:
                period_averages[period_name] = np.mean(period_cars)
            else:
                period_averages[period_name] = 0
    
    # Step 2: Calculate percentages (original method) and show final normalized percentages
    period_data = []
    total_points = sum(results['period_results'].values())
    
    # Group periods by name to get unique periods for display
    unique_periods = {}
    for period in tou_periods:
        period_name = period['name']
        if period_name not in unique_periods:
            unique_periods[period_name] = period
    
    # Calculate average capacity for each period from margin curve
    period_avg_capacities = {}
    step_duration_hours = time_step / 60.0
    
    for period_name, period in unique_periods.items():
        # Get all margin curve values for this period across all its parts
        period_margin_values = []
        
        print(f"🔍 DEBUG: Processing {period_name} - Period data: {period}")
        
        # Handle multi-part periods by collecting all steps from all parts
        for day in range(2):  # 48 hours = 2 days
            # Convert start/end hours to list of hours for this period
            start_hour = period.get('start', 0)
            end_hour = period.get('end', 24)
            period_hours = list(range(start_hour, end_hour))
            print(f"🔍 DEBUG: {period_name} - Day {day}, Start: {start_hour}, End: {end_hour}, Hours: {period_hours}")
            
            for hour in period_hours:
                # Convert hour to step index for this day
                day_hour = hour + (day * 24)
                step_index = int(day_hour / step_duration_hours)
                
                print(f"🔍 DEBUG: {period_name} - Hour {hour} -> Day hour {day_hour} -> Step {step_index}")
                
                # Ensure index is within bounds
                if 0 <= step_index < len(results['margin_curve']):
                    margin_value = results['margin_curve'][step_index]
                    period_margin_values.append(margin_value)
                    print(f"🔍 DEBUG: {period_name} - Added margin value: {margin_value:.1f} kW")
                else:
                    print(f"🔍 DEBUG: {period_name} - Step {step_index} out of bounds (max: {len(results['margin_curve'])})")
        
        # Calculate average capacity for this period
        if period_margin_values:
            period_avg_capacities[period_name] = np.mean(period_margin_values)
            print(f"🔍 DEBUG: {period_name} - Final average capacity: {period_avg_capacities[period_name]:.1f} kW from {len(period_margin_values)} values")
        else:
            period_avg_capacities[period_name] = 0
            print(f"🔍 DEBUG: {period_name} - No margin values found, setting to 0")
    
    # Calculate simple percentages and apply them directly (no complex normalization)
    capacity_weighted_values = {}
    total_capacity_weighted = 0
    
    for period_name, period in unique_periods.items():
        original_percentage = (results['period_results'][period_name] / total_points * 100) if total_points > 0 else 0
        avg_cars = period_averages.get(period_name, 0)
        avg_capacity = period_avg_capacities.get(period_name, 0)
        
        # Calculate capacity-weighted percentage with power function to amplify the effect
        capacity_weight = 2.0  # Weight factor - squaring the capacity for stronger effect
        capacity_weighted = original_percentage * (avg_capacity ** capacity_weight) / (100.0 ** capacity_weight)  # Square the capacity
        
        print(f"🔍 DEBUG: {period_name} - Original: {original_percentage:.1f}%, Avg Capacity: {avg_capacity:.1f} kW, Squared: {(avg_capacity ** capacity_weight):.1f}, Weighted: {capacity_weighted:.1f}")
        
        capacity_weighted_values[period_name] = capacity_weighted
        total_capacity_weighted += capacity_weighted
    
    # Initialize period_data list
    period_data = []
    
    # Normalize capacity-weighted values to sum to 100%
    if total_capacity_weighted > 0:
        for period_name, capacity_weighted in capacity_weighted_values.items():
            normalized_percentage = (capacity_weighted / total_capacity_weighted) * 100
            
            # Apply the normalized percentage to all periods with this name
            for tou_period in tou_periods:
                if tou_period['name'] == period_name:
                    tou_period['adoption'] = round(normalized_percentage, 1)
            
            # Add to table data with the FINAL normalized percentage
            period_data.append({
                'Period': period_name,
                'Percentages of Adoption': f"{normalized_percentage:.1f}%"
            })
    else:
        # Fallback to original percentages if no capacity data
        for period_name, period in unique_periods.items():
            original_percentage = (results['period_results'][period_name] / total_points * 100) if total_points > 0 else 0
            for tou_period in tou_periods:
                if tou_period['name'] == period_name:
                    tou_period['adoption'] = round(original_percentage, 1)
            
            # Add to table data with the fallback percentage
            period_data.append({
                'Period': period_name,
                'Percentages of Adoption': f"{original_percentage:.1f}%"
            })
    
    # Add total cars row
    total_cars = sum(results['cars_added_per_step'])
    # Divide by 2 since optimizer runs for 48 hours but normal simulation uses 24 hours
    total_cars_24h = total_cars // 2
    period_data.append({
        'Period': 'Total Cars',
        'Percentages of Adoption': f"{total_cars_24h}"
    })
    
    # Create DataFrame and display
    df = pd.DataFrame(period_data)
    st.write("**📊 Dynamic Optimization Results Table:**")
    st.write("*Percentages of Adoption: Final percentage applied to TOU fields (sums to 100%)*")
    st.dataframe(df, use_container_width=True)
    
    # Display current optimization results if just completed
    if st.session_state.get('dynamic_optimization_just_completed', False):
        st.session_state.dynamic_optimization_just_completed = False
    
    # Now apply the optimization results AFTER normalization is complete
    # Update TOU periods with optimized percentages (now with final normalized values)
    tou_periods = st.session_state.optimization_strategy.get('time_of_use_periods', [])
    
    # Update the TOU periods in session state
    st.session_state.optimization_strategy['time_of_use_periods'] = tou_periods
    
    # Store car count for display (but don't apply to UI fields like Bayesian optimizer)
    st.session_state.dynamic_optimization_car_count = total_cars_24h
    
    # Store results in session state for persistence
    st.session_state.dynamic_optimization_completed = True
    st.session_state.dynamic_optimization_just_completed = True
    
    # Convert numpy arrays to lists for session state storage
    def safe_tolist(data):
        if hasattr(data, 'tolist'):
            return data.tolist()
        else:
            return data
    
    # Store the FINAL normalized percentages instead of original period results
    final_period_results = {}
    for period in tou_periods:
        final_period_results[period['name']] = period['adoption']
    
    st.session_state.dynamic_optimization_results = {
        'total_cars': total_cars_24h,
        'period_results': final_period_results,  # Store final normalized values
        'tou_periods': tou_periods,
        'margin_curve': safe_tolist(results['margin_curve']),
        'final_load': safe_tolist(results['final_load']),
        'cars_added_per_step': safe_tolist(results['cars_added_per_step']),
        'active_cars_per_step': safe_tolist(results['active_cars_per_step']),
        'time_step': time_step
    }
    
    # Store optimized TOU values in session state (like Bayesian optimizer)
    # Use the final normalized percentages that are already calculated and applied to TOU periods
    optimized_tou_values = {}
    
    # Debug output to verify final percentages
    print(f"🔍 Gradient Optimizer UI Debug - {len(tou_periods)} periods:")
    print(f"  Using final normalized percentages from table calculation")
    
    # Create a flexible mapping system that works with any number of periods
    # Only map periods that actually exist to avoid conflicts
    period_mapping = {}
    
    # Map based on the actual periods present using period_one, period_two, etc.
    for period in tou_periods:
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
    
    # Use the final normalized percentages that are already applied to TOU periods
    for period in tou_periods:
        period_name = period['name']
        if period_name in period_mapping:
            # Get the final percentage that was already calculated with squaring and normalization
            final_percentage = period['adoption']
            param_name = period_mapping[period_name]
            optimized_tou_values[param_name] = final_percentage
            print(f"  {period_name} -> {param_name}: {final_percentage:.2f}%")
    
    # Ensure we don't have duplicate mappings that could cause unequal distribution
    # For 2-3 periods, remove unused mappings
    if len(tou_periods) == 2:
        # For 2 periods, only use the first two mappings
        for key in ['tou_midpeak', 'tou_peak']:
            if key in optimized_tou_values:
                del optimized_tou_values[key]
    elif len(tou_periods) == 3:
        # For 3 periods, only use the first three mappings
        if 'tou_peak' in optimized_tou_values:
            del optimized_tou_values['tou_peak']
    
    print(f"  Final optimized_tou_values: {optimized_tou_values}")
    
    st.session_state.optimized_tou_values = optimized_tou_values
    
    # Apply car count results (like max cars optimizer)
    st.session_state.total_evs = total_cars_24h
    st.session_state.charger_config['ac_count'] = total_cars_24h
    
    # Check if Time of Use is enabled
    time_of_use_enabled = ('smart_charging' in st.session_state.get('active_strategies', []) and 
                          st.session_state.optimization_strategy.get('smart_charging_percent', 0) > 0)
    
    if time_of_use_enabled:
        # For TOU, apply all cars to first peak (same as before)
        if 'time_peaks' in st.session_state and len(st.session_state.time_peaks) > 0:
            st.session_state.time_peaks[0]['quantity'] = total_cars_24h
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
                    new_quantity = int(total_cars_24h * proportion)
                    peak['quantity'] = new_quantity
                
                # Ensure we don't lose cars due to rounding
                distributed_total = sum(peak.get('quantity', 0) for peak in st.session_state.time_peaks)
                if distributed_total < total_cars_24h:
                    # Add remaining cars to the first peak
                    remaining = total_cars_24h - distributed_total
                    st.session_state.time_peaks[0]['quantity'] += remaining
            else:
                # If no cars currently, put all in first peak
                st.session_state.time_peaks[0]['quantity'] = total_cars_24h
    
    # Only use optimized_tou_values approach (like Bayesian optimizer)
    
    # Rerun to apply the values to the configuration (like max cars optimizer)
    st.rerun()
    
    # Display simulation curves
    st.subheader("📈 Simulation Curves")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot margin curve and final load
    step_duration_hours = time_step / 60.0
    hours = np.arange(len(results['margin_curve'])) * step_duration_hours
    ax1.plot(hours, results['margin_curve'], label='Margin Curve', color='blue')
    ax1.plot(hours, results['final_load'], label='Final Load', color='red')
    ax1.set_title('Margin Curve vs Final Load')
    ax1.set_xlabel('Time (Hours)')
    ax1.set_ylabel('Power (kW)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot cars added per step
    ax2.plot(hours, results['cars_added_per_step'], label='Cars Added', color='green', marker='o', markersize=3)
    ax2.set_title('Cars Added per Time Step')
    ax2.set_xlabel('Time (Hours)')
    ax2.set_ylabel('Number of Cars')
    ax2.legend()
    ax2.grid(True)
    
    # Plot active cars per step
    ax3.plot(hours, results['active_cars_per_step'], label='Active Cars', color='orange', linewidth=2)
    ax3.set_title('Active Cars per Time Step')
    ax3.set_xlabel('Time (Hours)')
    ax3.set_ylabel('Number of Active Cars')
    ax3.legend()
    ax3.grid(True)
    
    st.pyplot(fig)

# Keep the old function name for compatibility
def create_gradient_optimizer_ui():
    """Legacy function name - now calls the dynamic capacity optimizer."""
    create_dynamic_capacity_optimizer_ui()

 