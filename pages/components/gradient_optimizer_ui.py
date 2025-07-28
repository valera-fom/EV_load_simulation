#!/usr/bin/env python3
"""
Gradient Optimizer UI for Streamlit
Replaces Neural Network UI with Gradient-based Optimization interface
"""

import streamlit as st
import numpy as np
import pandas as pd
from pages.components.gradient_optimizer import GradientOptimizer
from pages.components.capacity_analyzer import find_max_cars_capacity
from EV import EV, EV_MODELS
from charger import CHARGER_MODELS
from sim_setup import SimulationSetup

def simulation_function(params):
    """Simulation function that matches the main simulation exactly."""
    try:
        # Get current configuration from session state (EXACTLY like main simulation)
        ev_config = st.session_state.dynamic_ev
        charger_config = st.session_state.charger_config
        time_peaks = st.session_state.get('time_peaks', [])
        active_strategies = st.session_state.get('active_strategies', [])
        grid_mode = st.session_state.get('grid_mode', 'Reference Only')
        available_load_fraction = st.session_state.get('available_load_fraction', 0.8)
        data_source = st.session_state.get('data_source', 'Real Dataset')
        power_values = st.session_state.get('power_values', None)
        
        # Handle data loading/generation (EXACTLY like main simulation)
        if data_source == "Synthetic Generation":
            if 'synthetic_load_curve' in st.session_state and st.session_state.synthetic_load_curve is not None and len(st.session_state.synthetic_load_curve) > 0:
                power_values = st.session_state.synthetic_load_curve
                st.session_state.power_values = power_values
            elif power_values is None or len(power_values) == 0:
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
                except Exception as e:
                    print(f"❌ Error generating synthetic data: {e}")
                    return np.zeros(2880)
        elif data_source == "Real Dataset":
            # For Real Dataset, use existing data if available
            if power_values is None or len(power_values) == 0:
                print("📊 No real dataset loaded. Falling back to synthetic generation...")
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
                except Exception as e:
                    print(f"❌ Error generating synthetic data: {e}")
                    return np.zeros(2880)
            else:
                # Use the existing real dataset power values
                print(f"📊 Using real dataset with {len(power_values)} data points")
        
        # Ensure power_values has correct length for 48-hour simulation (2880 minutes)
        if len(power_values) != 2880:
            if len(power_values) == 96:  # 24-hour hourly data
                # Interpolate to 15-minute intervals and repeat for 48 hours
                power_values_15min = np.repeat(power_values, 4)  # Convert to 15-min intervals (96*4=384)
                power_values = np.tile(power_values_15min, 8)  # Repeat for 48 hours (384*8=3072)
                power_values = power_values[:2880]  # Trim to exact length
            elif len(power_values) == 48:  # 24-hour hourly data
                # Interpolate to 15-minute intervals and repeat for 48 hours
                power_values_15min = np.repeat(power_values, 4)  # Convert to 15-min intervals (48*4=192)
                power_values = np.tile(power_values_15min, 15)  # Repeat for 48 hours (192*15=2880)
            elif len(power_values) == 1440:  # 24-hour minute data
                # Repeat for 48 hours
                power_values = np.tile(power_values, 2)
            else:
                # Generic approach: repeat or tile to get 2880 points
                if len(power_values) < 2880:
                    repeats_needed = int(np.ceil(2880 / len(power_values)))
                    power_values = np.tile(power_values, repeats_needed)
                power_values = power_values[:2880]  # Trim to exact length
        
        # Create dynamic EV model (EXACTLY like main simulation)
        dynamic_ev_model = {
            'name': 'Custom EV',
            'capacity': ev_config.get('capacity', 75),
            'AC': ev_config.get('AC', 11)
        }
        
        # Temporarily update EV_MODELS with our dynamic EV
        original_ev_models = EV_MODELS.copy()
        EV_MODELS['dynamic_ev'] = dynamic_ev_model
        
        # Calculate total EVs from the single peak (EXACTLY like main simulation)
        if time_peaks:
            total_evs = time_peaks[0]['quantity']
        else:
            print("❌ Error: No time peaks configured")
            return np.zeros(2880)
        
        # Define EV and charger counts (EXACTLY like main simulation)
        car_count = int(params['car_count'])  # Get from params (provided by wrapper)
        total_evs = car_count
        ev_counts = {'dynamic_ev': car_count}
        charger_counts = {'ac': car_count}  # Equal number of chargers
        
        # Create custom charger models with user-defined rates (EXACTLY like main simulation)
        custom_charger_models = {
            'ac': {'type': 'AC', 'power_kW': charger_config.get('ac_rate', 11)}
        }
        
        # Temporarily update CHARGER_MODELS with custom rates
        original_charger_models = CHARGER_MODELS.copy()
        CHARGER_MODELS.update(custom_charger_models)
        
        # Set grid power limit based on mode (EXACTLY like main simulation)
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
                        print(f"Error extending real dataset: {e}")
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
        
        # Apply optimization strategies to grid profile (EXACTLY like main simulation)
        adjusted_grid_profile = grid_profile_full.copy()
        
        # Apply PV + Battery optimization if enabled (charging during day, discharging during evening)
        pv_battery_support_adjusted = np.zeros_like(grid_profile_full)
        pv_battery_charge_curve = np.zeros_like(grid_profile_full)
        pv_direct_support_curve = np.zeros_like(grid_profile_full)
        pv_battery_discharge_curve = np.zeros_like(grid_profile_full)
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
        if 'grid_battery' in active_strategies:
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
                
                # Calculate rates based on battery capacity and duration
                # Charge rate = capacity / charge_duration
                # Discharge rate = capacity / discharge_duration
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
        if 'v2g' in active_strategies:
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
        
        # Apply margin to the adjusted grid profile (EXACTLY like main simulation)
        final_grid_profile = adjusted_grid_profile * available_load_fraction
        
        # Pre-calculate all arrival times for optimized simulation (EXACTLY like main simulation)
        arrival_times = []
        
        # Check if Time of Use is enabled
        time_of_use_enabled = ('smart_charging' in active_strategies and 
                              params.get('tou_super_offpeak_adoption', 0) > 0)
        
        if time_of_use_enabled:
            # Use dynamic TOU periods from session state instead of hardcoded ones
            time_of_use_timeline = st.session_state.get('time_of_use_timeline', None)
            
            if time_of_use_timeline and 'periods' in time_of_use_timeline:
                # Use the actual timeline periods from UI setup
                timeline_periods = time_of_use_timeline['periods']
                tou_periods = []
                
                # Convert timeline periods to simulation format
                for period in timeline_periods:
                    if period['hours']:  # Only include periods with assigned hours
                        # Find continuous ranges of hours
                        sorted_hours = sorted(period['hours'])
                        ranges = []
                        start = sorted_hours[0]
                        end = start
                        
                        for hour in sorted_hours[1:]:
                            if hour == end + 1:
                                end = hour
                            else:
                                ranges.append((start, end + 1))
                                start = hour
                                end = hour
                        
                        # Add the last range
                        ranges.append((start, end + 1))
                        
                        # Create a period for each range
                        for start_hour, end_hour in ranges:
                            # Map period names to optimization parameters
                            if period['name'] == 'Super Off-Peak':
                                adoption = params.get('tou_super_offpeak_adoption', 25)
                            elif period['name'] == 'Off-Peak':
                                adoption = params.get('tou_offpeak_adoption', 25)
                            elif period['name'] == 'Mid-Peak':
                                adoption = params.get('tou_midpeak_adoption', 25)
                            elif period['name'] == 'Peak':
                                adoption = params.get('tou_peak_adoption', 25)
                            else:
                                adoption = 25  # Default
                            
                            tou_periods.append({
                                'name': period['name'].lower().replace('-', '_'),
                                'start': start_hour - 1,  # Convert from 1-24 to 0-23 format
                                'end': end_hour - 1,      # Convert from 1-24 to 0-23 format
                                'adoption': adoption
                            })
            else:
                # Fallback to hardcoded periods if timeline not available
                tou_periods = [
                    {'name': 'super_offpeak', 'start': 0, 'end': 6, 'adoption': params.get('tou_super_offpeak_adoption', 25)},
                    {'name': 'offpeak', 'start': 6, 'end': 17, 'adoption': params.get('tou_offpeak_adoption', 25)},
                    {'name': 'midpeak', 'start': 17, 'end': 20, 'adoption': params.get('tou_midpeak_adoption', 25)},
                    {'name': 'peak', 'start': 20, 'end': 24, 'adoption': params.get('tou_peak_adoption', 25)}
                ]
            
            # Group periods by name for proper car distribution
            period_groups = {}
            for period in tou_periods:
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
            peak = time_peaks[0] if time_peaks else {'time': 12, 'span': 4, 'quantity': total_evs}
            peak_mean = peak['time'] * 60
            peak_span = peak['span'] * 60
            sigma = peak_span
            
            peak_arrivals = np.random.normal(peak_mean, sigma, peak['quantity'])
            arrival_times.extend(peak_arrivals)
        
        # Finalize arrival times (clip and sort once)
        arrival_times = np.array(arrival_times)
        arrival_times = np.clip(arrival_times, 0, 48 * 60 - 60)  # Always 48 hours for simulation
        arrival_times.sort()
        
        # Add V2G recharge EVs to arrival times (EXACTLY like main simulation)
        if 'v2g' in active_strategies:
            v2g_adoption_percent = st.session_state.optimization_strategy.get('v2g_adoption_percent', 0)
            v2g_recharge_arrival_hour = st.session_state.optimization_strategy.get('v2g_recharge_arrival_hour', 26)
            
            if v2g_adoption_percent > 0:
                total_v2g_evs = int(total_evs * v2g_adoption_percent / 100)
                
                # Calculate realistic discharge and recharge EVs
                ev_capacity = ev_config.get('capacity', 75)
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
                
                # V2G recharge EVs arrive the next day
                v2g_recharge_arrival_minute = v2g_recharge_arrival_hour * 60
                
                # Generate V2G recharge arrival times (spread over 1 hour)
                v2g_arrival_times = np.random.normal(v2g_recharge_arrival_minute, 30, recharge_evs)
                v2g_arrival_times = np.clip(v2g_arrival_times, v2g_recharge_arrival_minute - 30, v2g_recharge_arrival_minute + 30)
                
                # Add V2G recharge EVs to arrival times
                arrival_times = np.concatenate([arrival_times, v2g_arrival_times])
                arrival_times.sort()
                
                # Update total_evs to include V2G recharge EVs (same as main simulation)
                total_evs = len(arrival_times)
                ev_counts = {'dynamic_ev': total_evs}
                charger_counts = {'ac': total_evs}
        
        # Ensure we have the correct number of arrival times (same as main simulation)
        # The total_evs now includes V2G recharge EVs, so we don't need to trim
        
        # Set grid power limit for simulation (EXACTLY like main simulation)
        if grid_mode == "Grid Constrained":
            grid_power_limit = final_grid_profile
        else:
            grid_power_limit = None  # No constraint for Reference Only mode
        
        # Create simulation setup (EXACTLY like main simulation)
        sim = SimulationSetup(
            ev_counts=ev_counts,
            charger_counts=charger_counts,
            sim_duration=48 * 60,  # Always 48 hours for simulation
            arrival_time_mean=12 * 60,
            arrival_time_span=4 * 60,
            grid_power_limit=grid_power_limit,  # Pass grid constraint to simulation
            verbose=False,  # Disable verbose output for faster simulation
            silent=True     # Disable progress output during optimization
        )
        
        # Clear EVs and create them manually (EXACTLY like main simulation)
        sim.evs = []
        
        # Calculate how many V2G recharge EVs we have
        v2g_recharge_count = 0
        if 'v2g' in active_strategies:
            v2g_adoption_percent = st.session_state.optimization_strategy.get('v2g_adoption_percent', 0)
            if v2g_adoption_percent > 0:
                v2g_recharge_count = int(total_evs * v2g_adoption_percent / 100)
        
        # Calculate how many PV battery EVs we have
        pv_battery_count = 0
        if 'pv_battery' in active_strategies:
            pv_adoption_percent = st.session_state.optimization_strategy.get('pv_adoption_percent', 0)
            if pv_adoption_percent > 0:
                pv_battery_count = int(total_evs * pv_adoption_percent / 100)
        
        # Create EVs manually (EXACTLY like main simulation)
        for i, arrival_time in enumerate(arrival_times):
            ev_name = f"Custom_EV_{i+1}"
            
            # Determine EV types
            is_v2g_recharge = i >= (len(arrival_times) - v2g_recharge_count)
            is_pv_battery = i >= (len(arrival_times) - v2g_recharge_count - pv_battery_count) and i < (len(arrival_times) - v2g_recharge_count)
            
            # Set SOC based on EV type (EXACTLY like main simulation)
            if is_v2g_recharge:
                # V2G EVs have lower SOC since they discharged earlier
                # Calculate realistic discharge based on battery capacity constraints
                v2g_discharge_duration = st.session_state.optimization_strategy.get('v2g_discharge_duration', 3)
                v2g_max_discharge_rate = st.session_state.optimization_strategy.get('v2g_max_discharge_rate', 7)
                ev_capacity = ev_config.get('capacity', 75)
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
            
            ev = EV(name=ev_name, battery_capacity=ev_config.get('capacity', 75),
                   max_charging_power=ev_config.get('AC', 11), soc=soc)
            ev.ev_type = 'dynamic_ev'
            ev.is_pv_battery = is_pv_battery  # Mark PV battery EVs
            sim.evs.append(ev)
        
        # Schedule EVs manually (EXACTLY like main simulation)
        for i, ev in enumerate(sim.evs):
            sim.env.process(sim._ev_arrival(ev, arrival_times[i]))
        
        # Set optimization strategy (EXACTLY like main simulation)
        sim.optimization_strategy = {
            'smart_charging_percent': 100 if 'smart_charging' in active_strategies else 0,
            'time_of_use_periods': tou_periods if time_of_use_enabled else [],
            'pv_adoption_percent': 0,  # Simplified for now
            'grid_battery_adoption_percent': 0,  # Simplified for now
            'v2g_adoption_percent': 0,  # Simplified for now
        }
        
        # Also set the TOU adoption percentages directly in the optimization strategy
        if time_of_use_enabled and tou_periods:
            sim.optimization_strategy['tou_super_offpeak_adoption_percent'] = params.get('tou_super_offpeak_adoption', 25)
            sim.optimization_strategy['tou_offpeak_adoption_percent'] = params.get('tou_offpeak_adoption', 25)
            sim.optimization_strategy['tou_midpeak_adoption_percent'] = params.get('tou_midpeak_adoption', 25)
            sim.optimization_strategy['tou_peak_adoption_percent'] = params.get('tou_peak_adoption', 25)
        
        # Run simulation (EXACTLY like main simulation)
        sim.env.run(until=48 * 60)  # Always run for 48 hours
        
        # Get load curve (EXACTLY like main simulation)
        load_curve = np.array(sim.load_curve) if hasattr(sim, 'load_curve') else np.zeros(2880)
        
        # Check if load curve is all zeros
        if np.all(load_curve == 0):
            print(f"❌ CRITICAL: Load curve is all zeros! Simulation failed to generate any load.")
        
        # Restore original EV_MODELS (EXACTLY like main simulation)
        EV_MODELS.clear()
        EV_MODELS.update(original_ev_models)
        
        # Restore original CHARGER_MODELS
        CHARGER_MODELS.clear()
        CHARGER_MODELS.update(original_charger_models)
        
        return load_curve
        
    except Exception as e:
        print(f"❌ ERROR in simulation function: {e}")
        import traceback
        traceback.print_exc()
        
        # Restore EV_MODELS even if simulation fails
        try:
            EV_MODELS.clear()
            EV_MODELS.update(original_ev_models)
            CHARGER_MODELS.clear()
            CHARGER_MODELS.update(original_charger_models)
        except:
            pass
        
        return np.zeros(2880)  # Return zero load curve on error

def create_gradient_optimizer_ui():
    """Create the Gradient-based optimizer UI."""
    
    st.write("--------------------------------")
    # Gradient-based Capacity Optimizer
    st.write("🚀 **Gradient-Based Capacity Optimizer**")
    
    # Optimization quality dropdown
    optimization_quality = st.selectbox(
        "🎯 Optimization Quality",
        options=["Quick Test", "Standard", "High Quality"],
        index=1,
        help="Choose optimization quality and iteration count"
    )
    
    # Set iterations and learning rate based on quality
    if optimization_quality == "Quick Test":
        n_iterations = 20
        learning_rate = 2.0  # Much more aggressive
    elif optimization_quality == "Standard":
        n_iterations = 30
        learning_rate = 2.0  # Much more aggressive
    else:  # High Quality
        n_iterations = 50
        learning_rate = 1.2  # Much more aggressive
    
    if st.button("🚀 Run Gradient Optimization", type="primary"):
        # Check if smart charging is enabled
        active_strategies = st.session_state.get('active_strategies', [])
        
        if 'smart_charging' not in active_strategies:
            st.warning("⚠️ **Please enable Time-of-Use optimization in the Optimization Strategies section to run Gradient optimization.**")
            return
    
        # Build active parameters based on active strategies
        active_params = {}
        
        # Only add TOU parameters (no car_count) - FULL RANGE 0-50%
        if 'smart_charging' in active_strategies:
            active_params.update({
                'tou_super_offpeak_adoption': (0, 50),   # Full range 0-50%
                'tou_offpeak_adoption': (0, 50),         # Full range 0-50%
                'tou_midpeak_adoption': (0, 50),         # Full range 0-50%
                'tou_peak_adoption': (0, 50)             # Full range 0-50%
            })
        
        # Fixed car count for optimization - TOU percentages will scale the effective car count
        fixed_car_count = 25  # Start with 25 cars, TOU percentages will scale this up
        
        print(f"🔍 Car count setup:")
        print(f"  Base car count: {fixed_car_count} cars")
        print(f"  TOU percentages will scale effective car count")
        print(f"  Example: 25 cars × 200% TOU = 50 effective cars")
        
        # Create a wrapper function that captures the fixed car count
        def simulation_wrapper(params):
            # Add the fixed car count to the parameters
            params_with_cars = params.copy()
            params_with_cars['car_count'] = fixed_car_count
            return simulation_function(params_with_cars)
        
        # Create Gradient optimizer
        try:
            optimizer = GradientOptimizer(
                simulation_function=simulation_wrapper,
                parameter_bounds=active_params,
                n_iterations=n_iterations,
                learning_rate=learning_rate
            )
            
            # Print optimization setup info
            print(f"🚀 Starting Gradient Optimization:")
            print(f"  📊 Iterations: {n_iterations}")
            print(f"  📈 Learning Rate: {learning_rate}")
            print(f"  🚗 Fixed Car Count: {fixed_car_count}")
            print(f"  ⚙️ Parameter Bounds: {active_params}")
            print(f"  " + "="*50)
            
            # Force refresh margin curve with current session state (in case data changed)
            optimizer._setup_margin_curve()
            
        except Exception as e:
            st.error(f"❌ Error creating Gradient Optimizer: {e}")
            import traceback
            st.write(f"**Error details:**")
            st.code(traceback.format_exc())
            return
        
        # Add progress tracking
        st.write("Running Gradient-based optimization...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Track progress
        def progress_callback(iteration, total_iterations, best_reward):
            progress = min(1.0, (iteration + 1) / total_iterations)  # Ensure progress doesn't exceed 1.0
            progress_bar.progress(progress)
            status_text.write(f"🔄 **Iteration {iteration + 1}/{total_iterations}** ({progress*100:.1f}% complete)")
            status_text.write(f"🎯 **Current best reward: {best_reward:.2f}**")
        
        # Run optimization with progress tracking
        try:
            results = optimizer.optimize(progress_callback=progress_callback)
        except Exception as e:
            st.error(f"❌ Error during Gradient Optimization: {e}")
            import traceback
            st.write(f"**Error details:**")
            st.code(traceback.format_exc())
            return
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state for persistence
        st.session_state.gradient_optimization_completed = True
        st.session_state.gradient_results = results
        st.session_state.gradient_active_strategies = active_strategies.copy()
        
        # Automatically apply the optimized parameters
        if results and 'best_parameters' in results:
            best_params = results['best_parameters'].copy()
            
            # Apply TOU adoption percentages to the optimization strategy
            if 'smart_charging' in active_strategies:
                # Get raw TOU values from optimization
                tou_super_offpeak = best_params.get('tou_super_offpeak_adoption', 25)
                tou_offpeak = best_params.get('tou_offpeak_adoption', 25)
                tou_midpeak = best_params.get('tou_midpeak_adoption', 25)
                tou_peak = best_params.get('tou_peak_adoption', 25)
                
                # Store original total for scaling factor calculation
                original_total_tou = tou_super_offpeak + tou_offpeak + tou_midpeak + tou_peak
                
                # Normalize TOU values to 100% (required for proper distribution)
                if original_total_tou > 0:
                    tou_super_offpeak = (tou_super_offpeak / original_total_tou) * 100
                    tou_offpeak = (tou_offpeak / original_total_tou) * 100
                    tou_midpeak = (tou_midpeak / original_total_tou) * 100
                    tou_peak = (tou_peak / original_total_tou) * 100
                else:
                    tou_super_offpeak = tou_offpeak = tou_midpeak = tou_peak = 25
                
                # Round to 2 decimal places
                tou_super_offpeak = round(tou_super_offpeak, 2)
                tou_offpeak = round(tou_offpeak, 2)
                tou_midpeak = round(tou_midpeak, 2)
                tou_peak = round(tou_peak, 2)
                
                # Calculate scaling factor based on original total TOU values
                scaling_factor = original_total_tou / 100 if original_total_tou > 0 else 1.0
                final_car_count = int(fixed_car_count * scaling_factor)
                

                
                # Apply normalized values
                st.session_state.optimization_strategy['tou_super_offpeak_adoption_percent'] = tou_super_offpeak
                st.session_state.optimization_strategy['tou_offpeak_adoption_percent'] = tou_offpeak
                st.session_state.optimization_strategy['tou_midpeak_adoption_percent'] = tou_midpeak
                st.session_state.optimization_strategy['tou_peak_adoption_percent'] = tou_peak
                
                # Update the timeline periods with the optimized values
                if 'time_of_use_timeline' in st.session_state:
                    timeline = st.session_state.time_of_use_timeline
                    
                    # Map optimization parameters to timeline periods
                    param_mapping = {
                        'tou_super_offpeak_adoption': 0,  # First period
                        'tou_offpeak_adoption': 1,        # Second period  
                        'tou_midpeak_adoption': 2,        # Third period
                        'tou_peak_adoption': 3            # Fourth period
                    }
                    
                    # Update timeline periods with optimized values
                    for i, period in enumerate(timeline['periods']):
                        if i < 4:  # Only handle first 4 periods
                            param_key = list(param_mapping.keys())[i]
                            adoption_value = round(float(best_params.get(param_key, 25)), 2)
                            
                            # Update the adoption value in the timeline period
                            period['adoption'] = adoption_value
                    
                    # Store the optimized TOU values to be applied after rerun
                    st.session_state.optimized_tou_values = {
                        'tou_super_offpeak': tou_super_offpeak,
                        'tou_offpeak': tou_offpeak,
                        'tou_midpeak': tou_midpeak,
                        'tou_peak': tou_peak
                    }
            
            # Update EV and charger counts from scaling factor
            st.session_state.total_evs = final_car_count
            st.session_state.charger_config['ac_count'] = final_car_count
            
            # Update time control EV count
            if 'time_peaks' in st.session_state and len(st.session_state.time_peaks) > 0:
                st.session_state.time_peaks[0]['quantity'] = final_car_count
            
            # Display Gradient optimization results table
            st.write("**📊 Gradient Optimization Results:**")
            
            # Create results dictionary for display
            gradient_results = {
                'Car Count': final_car_count,
                'Super Off-Peak Adoption': f"{tou_super_offpeak:.1f}%",
                'Off-Peak Adoption': f"{tou_offpeak:.1f}%",
                'Mid-Peak Adoption': f"{tou_midpeak:.1f}%",
                'Peak Adoption': f"{tou_peak:.1f}%"
            }
            
            # Display as dataframe
            import pandas as pd
            df = pd.DataFrame(list(gradient_results.items()), columns=['Parameter', 'Value'])
            st.dataframe(df, use_container_width=True)
            
            # Store results table in session state for persistence
            st.session_state.gradient_results_table = gradient_results.copy()
        
        # Rerun to apply the optimized parameters to the configuration
        st.rerun()
    
    # Display stored Gradient optimization results if available
    if st.session_state.get('gradient_optimization_completed', False):
        results = st.session_state.get('gradient_results', {})
        active_strategies = st.session_state.get('gradient_active_strategies', [])
        
        if results and 'best_parameters' in results:
            st.write("**📊 Gradient Optimization Results:**")
            
            # Get the stored results table
            gradient_results = st.session_state.get('gradient_results_table', {})
            
            if gradient_results:
                # Display as dataframe
                import pandas as pd
                df = pd.DataFrame(list(gradient_results.items()), columns=['Parameter', 'Value'])
                st.dataframe(df, use_container_width=True)
            else:
                # Fallback to calculating from best parameters
                best_params = results['best_parameters'].copy()
                
                # Get raw TOU values from optimization
                tou_super_offpeak = best_params.get('tou_super_offpeak_adoption', 25)
                tou_offpeak = best_params.get('tou_offpeak_adoption', 25)
                tou_midpeak = best_params.get('tou_midpeak_adoption', 25)
                tou_peak = best_params.get('tou_peak_adoption', 25)
                
                # Normalize TOU values to sum to 100%
                total_tou = tou_super_offpeak + tou_offpeak + tou_midpeak + tou_peak
                if total_tou > 0:
                    tou_super_offpeak = (tou_super_offpeak / total_tou) * 100
                    tou_offpeak = (tou_offpeak / total_tou) * 100
                    tou_midpeak = (tou_midpeak / total_tou) * 100
                    tou_peak = (tou_peak / total_tou) * 100
                else:
                    tou_super_offpeak = tou_offpeak = tou_midpeak = tou_peak = 25
                
                # Round to 2 decimal places
                tou_super_offpeak = round(tou_super_offpeak, 2)
                tou_offpeak = round(tou_offpeak, 2)
                tou_midpeak = round(tou_midpeak, 2)
                tou_peak = round(tou_peak, 2)
                
                # Calculate scaling factor based on normalization
                scaling_factor = 100 / total_tou if total_tou > 0 else 1.0
                final_car_count = int(100 * scaling_factor)  # Fixed car count was 100
                
                # Create results dictionary for display
                gradient_results = {
                    'Car Count': final_car_count,
                    'Super Off-Peak Adoption': f"{tou_super_offpeak:.1f}%",
                    'Off-Peak Adoption': f"{tou_offpeak:.1f}%",
                    'Mid-Peak Adoption': f"{tou_midpeak:.1f}%",
                    'Peak Adoption': f"{tou_peak:.1f}%"
                }
                
                # Display as dataframe
                import pandas as pd
                df = pd.DataFrame(list(gradient_results.items()), columns=['Parameter', 'Value'])
                st.dataframe(df, use_container_width=True)

 