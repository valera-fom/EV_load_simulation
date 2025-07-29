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
    
    # Try to get current cars from time_peaks first
    if time_peaks and len(time_peaks) > 0:
        current_cars = time_peaks[0].get('quantity', 20)
    
    # If not found in time_peaks, try to get from session state
    if current_cars == 20:
        current_cars = st.session_state.get('total_evs', 20)
    
    # If still default, try to get from charger config
    if current_cars == 20:
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
        
        # Upsample 15-minute data to 1-minute intervals (EXACTLY like main simulation)
        grid_profile_15min = power_values
        adjusted_margin_curve = np.repeat(grid_profile_15min, 15).astype(float) * capacity_margin
        
        # Extend grid profile for 60-hour simulation (EXACTLY like main simulation)
        daily_minutes = 24 * 60
        sim_duration_min = 60 * 60  # Run for 60 hours
        
        # Check if we're using synthetic data
        using_synthetic_data = ('synthetic_load_curve' in st.session_state and 
                              st.session_state.synthetic_load_curve is not None)
        
        if len(adjusted_margin_curve) < sim_duration_min:
            if using_synthetic_data:
                # For synthetic data, repeat the generated profile to cover the simulation duration
                num_days_needed = sim_duration_min // daily_minutes + (1 if sim_duration_min % daily_minutes > 0 else 0)
                extended_profile = []
                
                for day in range(num_days_needed):
                    # Use the same synthetic profile for each day
                    extended_profile.extend(adjusted_margin_curve)
                
                adjusted_margin_curve = np.array(extended_profile[:sim_duration_min])
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
                                # Use FULL capacity (no margin yet) - upsample to 1-minute intervals
                                day_power_values = day_data.iloc[:, 2].values
                                day_profile = np.repeat(day_power_values, 15).astype(float) * capacity_margin
                                extended_profile.extend(day_profile)
                            else:
                                # If no data for this day, use the last available day
                                extended_profile.extend(adjusted_margin_curve)
                            
                            # Move to next day
                            current_date = current_date + pd.Timedelta(days=1)
                        
                        adjusted_margin_curve = np.array(extended_profile[:sim_duration_min])
                    except Exception as e:
                        st.error(f"❌ Error loading dataset for capacity analysis: {e}")
                        # Fallback to repeating the available profile
                        num_repeats = sim_duration_min // len(adjusted_margin_curve) + 1
                        adjusted_margin_curve = np.tile(adjusted_margin_curve, num_repeats)[:sim_duration_min]
                else:
                    # No dataset selected, repeat the available profile
                    num_repeats = sim_duration_min // len(adjusted_margin_curve) + 1
                    adjusted_margin_curve = np.tile(adjusted_margin_curve, num_repeats)[:sim_duration_min]
        
        # Apply V2G optimization if enabled (EXACTLY like main simulation)
        adjusted_grid_profile = adjusted_margin_curve.copy()
        v2g_discharge_curve = np.zeros_like(adjusted_margin_curve)
        if 'v2g' in active_strategies:
            v2g_adoption_percent = st.session_state.optimization_strategy.get('v2g_adoption_percent', 0)
            v2g_discharge_duration = st.session_state.optimization_strategy.get('v2g_discharge_duration', 3)
            v2g_max_discharge_rate = st.session_state.optimization_strategy.get('v2g_max_discharge_rate', 7)
            
            if v2g_adoption_percent > 0 and v2g_discharge_duration > 0 and v2g_max_discharge_rate > 0:
                total_evs_for_v2g = current_cars
                v2g_evs = int(total_evs_for_v2g * v2g_adoption_percent / 100)
                
                # Calculate discharge rate and total power
                ev_capacity = ev_config['capacity']
                full_discharge_time = ev_capacity / v2g_max_discharge_rate if v2g_max_discharge_rate > 0 else 1.0
                
                if v2g_discharge_duration >= full_discharge_time:
                    # Fully discharged - use slower rate if needed
                    actual_discharge_rate = ev_capacity / v2g_discharge_duration
                else:
                    # Partially discharged
                    actual_discharge_rate = v2g_max_discharge_rate
                
                total_v2g_power = v2g_evs * actual_discharge_rate
                
                # Apply V2G discharge during peak hours (EXACTLY like main simulation)
                v2g_start_hour = st.session_state.optimization_strategy.get('v2g_start_hour', 18)
                v2g_end_hour = v2g_start_hour + v2g_discharge_duration
                
                for minute in range(len(adjusted_grid_profile)):
                    time_of_day = minute % (24 * 60)
                    hour_of_day = time_of_day / 60
                    
                    if v2g_start_hour <= hour_of_day < v2g_end_hour:
                        v2g_discharge_curve[minute] = total_v2g_power
                        adjusted_grid_profile[minute] += total_v2g_power  # Increase available capacity
        
        # Apply PV Battery optimization if enabled (EXACTLY like main simulation)
        pv_battery_charge_curve = np.zeros_like(adjusted_margin_curve)
        pv_battery_discharge_curve = np.zeros_like(adjusted_margin_curve)
        pv_direct_support_curve = np.zeros_like(adjusted_margin_curve)
        if 'pv_battery' in active_strategies:
            pv_adoption_percent = st.session_state.optimization_strategy.get('pv_adoption_percent', 0)
            pv_battery_capacity = st.session_state.optimization_strategy.get('pv_battery_capacity', 0)
            pv_battery_max_rate = st.session_state.optimization_strategy.get('pv_battery_max_rate', 0)
            pv_charge_start_hour = st.session_state.optimization_strategy.get('pv_charge_start_hour', 7)
            pv_charge_duration = st.session_state.optimization_strategy.get('pv_charge_duration', 8)
            pv_discharge_start_hour = st.session_state.optimization_strategy.get('pv_discharge_start_hour', 18)
            pv_discharge_duration = st.session_state.optimization_strategy.get('pv_discharge_duration', 4)
            
            if pv_adoption_percent > 0 and pv_battery_capacity > 0 and pv_battery_max_rate > 0:
                total_evs_for_pv = current_cars
                pv_evs = int(total_evs_for_pv * pv_adoption_percent / 100)
                
                # Calculate charge and discharge rates
                charge_rate = pv_battery_capacity / pv_charge_duration
                discharge_rate = pv_battery_capacity / pv_discharge_duration
                total_charge_power = pv_evs * charge_rate
                total_discharge_power = pv_evs * discharge_rate
                
                # Calculate system support power (total PV power available)
                total_system_support_power = total_charge_power + total_discharge_power
                
                # Apply PV battery optimization (EXACTLY like main simulation)
                for minute in range(len(adjusted_grid_profile)):
                    time_of_day = minute % (24 * 60)
                    hour_of_day = time_of_day / 60
                    
                    # Day Phase: Simultaneous battery charging and grid support
                    if pv_charge_start_hour <= hour_of_day < (pv_charge_start_hour + pv_charge_duration):
                        # Calculate total PV power available for this EV
                        ev_total_pv_power = total_system_support_power / pv_evs if pv_evs > 0 else 0
                        
                        # Calculate remaining power for grid support
                        ev_grid_support_power = max(0, ev_total_pv_power - charge_rate)
                        
                        # Apply battery charging (no grid effect - PV charges batteries directly)
                        pv_battery_charge_curve[minute] += total_charge_power
                                    
                        # Apply grid support (increases available capacity)
                        if ev_grid_support_power > 0:
                            pv_direct_support_curve[minute] += ev_grid_support_power * pv_evs
                            adjusted_grid_profile[minute] += ev_grid_support_power * pv_evs
                    
                    # Evening Phase: Battery discharge
                    if pv_discharge_start_hour <= hour_of_day < (pv_discharge_start_hour + pv_discharge_duration):
                        pv_battery_discharge_curve[minute] += total_discharge_power
                        adjusted_grid_profile[minute] += total_discharge_power  # Increase available capacity
        
        # Apply Grid-Charged Batteries optimization if enabled (EXACTLY like main simulation)
        grid_battery_charge_curve = np.zeros_like(adjusted_margin_curve)
        grid_battery_discharge_curve = np.zeros_like(adjusted_margin_curve)
        if 'grid_battery' in active_strategies:
            grid_battery_adoption_percent = st.session_state.optimization_strategy.get('grid_battery_adoption_percent', 0)
            grid_battery_capacity = st.session_state.optimization_strategy.get('grid_battery_capacity', 0)
            grid_battery_max_rate = st.session_state.optimization_strategy.get('grid_battery_max_rate', 0)
            grid_battery_charge_start_hour = st.session_state.optimization_strategy.get('grid_battery_charge_start_hour', 7)
            grid_battery_charge_duration = st.session_state.optimization_strategy.get('grid_battery_charge_duration', 8)
            grid_battery_discharge_start_hour = st.session_state.optimization_strategy.get('grid_battery_discharge_start_hour', 18)
            grid_battery_discharge_duration = st.session_state.optimization_strategy.get('grid_battery_discharge_duration', 4)
            
            if grid_battery_adoption_percent > 0 and grid_battery_capacity > 0 and grid_battery_max_rate > 0:
                total_evs_for_grid_battery = current_cars
                grid_battery_evs = int(total_evs_for_grid_battery * grid_battery_adoption_percent / 100)
                
                # Use the same simple logic as PV battery
                charge_rate = grid_battery_capacity / grid_battery_charge_duration
                discharge_rate = grid_battery_capacity / grid_battery_discharge_duration
                total_charge_power = grid_battery_evs * charge_rate
                total_discharge_power = grid_battery_evs * discharge_rate
                
                # Apply grid battery optimization (EXACTLY like main simulation)
                for minute in range(len(adjusted_grid_profile)):
                    time_of_day = minute % (24 * 60)
                    hour_of_day = time_of_day / 60
                    
                    # Day Phase: Battery charging (consumes grid power)
                    if grid_battery_charge_start_hour <= hour_of_day < (grid_battery_charge_start_hour + grid_battery_charge_duration):
                        grid_battery_charge_curve[minute] += total_charge_power
                        adjusted_grid_profile[minute] -= total_charge_power  # Decrease available capacity
                    
                    # Evening Phase: Battery discharge (provides grid power)
                    if grid_battery_discharge_start_hour <= hour_of_day < (grid_battery_discharge_start_hour + grid_battery_discharge_duration):
                        grid_battery_discharge_curve[minute] += total_discharge_power
                        adjusted_grid_profile[minute] += total_discharge_power  # Increase available capacity
        
        # NOW USE THE ACTUAL SIMULATION ENGINE (EXACTLY like normal simulation)
        # Apply the same margin as main simulation
        final_grid_profile = adjusted_grid_profile * available_load_fraction
        
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
        simulation.grid_power_profile = adjusted_grid_profile
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
        
        # Generate arrival times (same as main simulation)
        arrival_times = []
        for day in range(2):  # 2 days like main simulation
            day_offset_minutes = day * 24 * 60
            for i in range(current_cars):
                # Use same arrival time generation as main simulation
                arrival_time = np.random.normal(12 * 60, 4 * 60)  # 12h mean, 4h span
                arrival_time = max(0, min(24 * 60 - 1, arrival_time))  # Clip to 0-24h
                arrival_time += day_offset_minutes
                arrival_times.append(arrival_time)
        
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