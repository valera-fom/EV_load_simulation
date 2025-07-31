import numpy as np
import streamlit as st
import pandas as pd
from sim_setup import SimulationSetup
from EV import EV, EV_MODELS
from charger import CHARGER_MODELS

def find_max_cars_capacity(ev_config, charger_config, time_peaks, active_strategies, 
                          grid_mode, available_load_fraction, power_values, sim_duration=24, num_steps=1):
    """
    Find the maximum number of cars that can fit under the margin curve.
    Uses the exact same approach as the normal simulation.
    """
    # Get current number of cars from configuration
    current_cars = 20  # Default fallback
    
    # Try to get current cars from time_peaks first (sum all peaks)
    if time_peaks and len(time_peaks) > 0:
        current_cars = sum(peak.get('quantity', 0) for peak in time_peaks)
    
    # If not found in time_peaks, try to get from session state
    if current_cars == 0:
        current_cars = st.session_state.get('total_evs', 20)
    
    # If still default, try to get from charger config
    if current_cars == 0:
        current_cars = charger_config.get('ac_count', 20)
    
    # Ensure we have a reasonable starting point (minimum 5 cars)
    current_cars = max(5, current_cars)
    
    # Create dynamic EV model (EXACTLY like normal simulation)
    dynamic_ev_model = {
        'name': 'Custom EV',
        'capacity': ev_config['capacity'],
        'AC': ev_config['AC']
    }
    
    # Temporarily update EV_MODELS with our dynamic EV (EXACTLY like normal simulation)
    original_ev_models = EV_MODELS.copy()
    EV_MODELS['dynamic_ev'] = dynamic_ev_model
    
    try:
        # Create custom charger models with user-defined rates (EXACTLY like normal simulation)
        custom_charger_models = {
            'ac': {'type': 'AC', 'power_kW': charger_config['ac_rate']}
        }
        
        # Temporarily update CHARGER_MODELS with custom rates (EXACTLY like normal simulation)
        original_charger_models = CHARGER_MODELS.copy()
        CHARGER_MODELS.update(custom_charger_models)
        
        # Prepare simulation parameters (EXACTLY like normal simulation)
        ev_counts = {'dynamic_ev': current_cars}
        charger_counts = {'ac': current_cars}
        
        # Calculate margin curve with more conservative margin than main simulation
        # Main simulation uses: available_load_fraction + 0.1 (e.g., 0.8 + 0.1 = 0.9)
        # We use: available_load_fraction - 0.1 (e.g., 0.8 - 0.1 = 0.7) for more conservative estimate
        capacity_margin = max(0.1, available_load_fraction - 0.1)  # Subtract 0.1 but don't go below 0.1
        
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
            st.error("❌ No grid data available for capacity analysis.")
            return current_cars
        
        # Use the extracted margin curve directly (this is the yellow line)
        adjusted_margin_curve = margin_curve
        
        # Extend grid profile for 60-hour simulation (EXACTLY like main simulation)
        daily_minutes = 24 * 60
        sim_duration_min = 60 * 60  # Run for 60 hours
        
        if len(adjusted_margin_curve) < sim_duration_min:
            # Repeat the profile to cover simulation duration
            num_repeats = sim_duration_min // len(adjusted_margin_curve) + 1
            adjusted_margin_curve = np.tile(adjusted_margin_curve, num_repeats)[:sim_duration_min]
        
        # NOW USE THE ACTUAL SIMULATION ENGINE (EXACTLY like normal simulation)
        # The margin curve already includes all battery effects, so we can use it directly
        final_grid_profile = adjusted_margin_curve  # No need to apply available_load_fraction again
        
        # Create SimulationSetup instance with the same parameters as main simulation
        if grid_mode == "Grid Constrained":
            grid_power_limit = final_grid_profile  # Pass the actual grid profile like main simulation
        else:
            grid_power_limit = None  # No constraint for Reference Only mode
        
        simulation = SimulationSetup(
                ev_counts=ev_counts,
                charger_counts=charger_counts,
            sim_duration=60 * 60,  # Run for 60 hours (same as main simulation)
            arrival_time_mean=12 * 60,  # Same as main simulation
            arrival_time_span=4 * 60,   # Same as main simulation
            grid_power_limit=grid_power_limit,  # Same as main simulation
            verbose=False  # Same as main simulation
        )
        simulation.grid_power_profile = adjusted_margin_curve
        simulation.evs = []
        
        # Calculate how many V2G recharge EVs we have (same as main simulation)
        v2g_recharge_count = 0
        if 'v2g' in active_strategies:
            v2g_adoption_percent = st.session_state.optimization_strategy.get('v2g_adoption_percent', 0)
            if v2g_adoption_percent > 0:
                v2g_recharge_count = int(current_cars * v2g_adoption_percent / 100)
            
        # Calculate how many PV battery EVs we have (same as main simulation)
        pv_battery_count = 0
        if 'pv_battery' in active_strategies:
            pv_adoption_percent = st.session_state.optimization_strategy.get('pv_adoption_percent', 0)
            if pv_adoption_percent > 0:
                pv_battery_count = int(current_cars * pv_adoption_percent / 100)
        
        # Generate arrival times for multiple peaks (same as main simulation)
        arrival_times = []
        
        # Check if Time of Use is enabled
        time_of_use_enabled = ('smart_charging' in active_strategies and 
                              st.session_state.optimization_strategy.get('smart_charging_percent', 0) > 0)
        
        if time_of_use_enabled:
            # Use TOU logic (same as main simulation)
            for day in range(2):  # 2 days like main simulation
                day_offset_minutes = day * 24 * 60
                for i in range(current_cars):
                    # Use same arrival time generation as main simulation
                    arrival_time = np.random.normal(12 * 60, 4 * 60)  # 12h mean, 4h span
                    arrival_time = max(0, min(24 * 60 - 1, arrival_time))  # Clip to 0-24h
                    arrival_time += day_offset_minutes
                    arrival_times.append(arrival_time)
        else:
            # Use multiple peaks logic (same as main simulation)
            for day in range(2):  # 2 days like main simulation
                day_offset_minutes = day * 24 * 60
                
                # Generate cars for each peak
                for peak in time_peaks:
                    peak_quantity = peak.get('quantity', 0)
                    if peak_quantity > 0:
                        peak_mean = peak['time'] * 60
                        peak_span = peak['span'] * 60
                        sigma = peak_span
                        
                        # Calculate charging duration to shift arrival times
                        ev_capacity = ev_config['capacity']
                        ev_charging_rate = ev_config['AC']
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
        
        arrival_times.sort()
        
        # Create EVs with same logic as main simulation
        for i, arrival_time in enumerate(arrival_times):
            ev_name = f"Capacity_EV_{i+1}"
            
            # Determine EV types (same as main simulation)
            is_v2g_recharge = i >= (len(arrival_times) - v2g_recharge_count)
            is_pv_battery = i >= (len(arrival_times) - v2g_recharge_count - pv_battery_count) and i < (len(arrival_times) - v2g_recharge_count)
            
            # Set SOC based on EV type (same as main simulation)
            ev_capacity = ev_config['capacity']
            if is_v2g_recharge:
                # V2G EVs have lower SOC since they discharged earlier
                v2g_discharge_duration = st.session_state.optimization_strategy.get('v2g_discharge_duration', 3)
                v2g_max_discharge_rate = st.session_state.optimization_strategy.get('v2g_max_discharge_rate', 7)
                full_discharge_time = ev_capacity / v2g_max_discharge_rate if v2g_max_discharge_rate > 0 else 1.0
                
                if v2g_discharge_duration >= full_discharge_time:
                    actual_discharge_rate = ev_capacity / v2g_discharge_duration
                    total_discharged_kwh = ev_capacity
                else:
                    actual_discharge_rate = v2g_max_discharge_rate
                    total_discharged_kwh = v2g_discharge_duration * v2g_max_discharge_rate
                
                initial_soc = 0.8
                discharged_soc = total_discharged_kwh / ev_capacity
                soc = max(0.1, initial_soc - discharged_soc)
            elif is_pv_battery:
                # PV battery EVs start with higher SOC since they charge during the day
                soc = st.session_state.get('ev_soc', 0.2) + 0.3
                soc = min(0.9, soc)
            else:
                # Regular EVs start at configured SOC
                soc = st.session_state.get('ev_soc', 0.2)
                
            ev = EV(name=ev_name, battery_capacity=ev_capacity,
                   max_charging_power=ev_config['AC'], soc=soc)
            ev.ev_type = 'dynamic_ev'
            ev.is_pv_battery = is_pv_battery
            simulation.evs.append(ev)
        
        # Schedule EVs with arrival times (same as main simulation)
        for i, ev in enumerate(simulation.evs):
            simulation.env.process(simulation._ev_arrival(ev, arrival_times[i]))
        
        # Run the simulation using the same engine (EXACTLY like normal simulation)
        simulation.env.run(until=72 * 60)  # Run for 72 hours like main simulation
        
        # Get the EV load curve from simulation
        ev_load_curve = simulation.load_curve.copy()
        
        # Ensure both curves have the same length
        min_length = min(len(ev_load_curve), len(adjusted_margin_curve))
        ev_load_curve = ev_load_curve[:min_length]
        margin_curve = adjusted_margin_curve[:min_length]  # Use the actual margin curve, not the red curve
        
        # Find the point where EV load is closest to hitting the margin curve
        # We want the point where EV load / margin is highest (closest to 1.0)
        # This is the bottleneck point that limits our capacity
        ratios = ev_load_curve / margin_curve
        max_ratio_idx = np.argmax(ratios)  # Point where ratio is highest (closest to 1.0)
        
        # Get the values at this bottleneck point
        margin_at_bottleneck = margin_curve[max_ratio_idx]
        ev_load_at_bottleneck = ev_load_curve[max_ratio_idx]
        ratio_at_bottleneck = ratios[max_ratio_idx]
        
        if ev_load_at_bottleneck > 0:
            # Calculate how many more cars we can fit based on the bottleneck ratio
            # If ratio is 0.5, we can fit 2x more cars (1/0.5 = 2)
            # If ratio is 0.8, we can fit 1.25x more cars (1/0.8 = 1.25)
            factor = 1.0 / ratio_at_bottleneck
            max_cars = int(current_cars * factor)
        else:
            # If no EV load, we can fit many more cars
            max_cars = current_cars * 10  # Arbitrary large number
        
        # Ensure we don't return an unreasonable number
        max_cars = max(5, min(max_cars, current_cars * 20))  # Between 5 and 20x current cars
        
        return max_cars
            
    except Exception as e:
        st.error(f"❌ Error during capacity analysis: {e}")
        return current_cars
    finally:
        # Restore original models (EXACTLY like normal simulation)
        EV_MODELS.clear()
        EV_MODELS.update(original_ev_models)
        CHARGER_MODELS.clear()
        CHARGER_MODELS.update(original_charger_models)