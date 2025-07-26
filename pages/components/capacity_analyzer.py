import numpy as np
import streamlit as st
from sim_setup import SimulationSetup
from EV import EV, EV_MODELS
from charger import CHARGER_MODELS

def find_max_cars_capacity(ev_config, charger_config, time_peaks, active_strategies, 
                          grid_mode, available_load_fraction, power_values, sim_duration=24, num_steps=1):
    """
    Find the maximum number of cars that can fit under the margin curve.
    Uses the exact same approach as the normal simulation.
    """
    # Multi-step capacity analysis
    current_cars = 20
    final_max_cars = current_cars
    
    for step in range(num_steps):
        st.write(f"**Step {step+1}/{num_steps}: Calculating maximum EV capacity...**")
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
            
            # Calculate margin curve with less conservative margin for capacity analysis
            # If available_load_fraction is 0.8, use 0.9; if 0.6, use 0.8, etc.
            capacity_margin = min(0.95, available_load_fraction + 0.1)  # Add 0.1 but cap at 0.95
            margin_curve = power_values * capacity_margin
            
            # Extend grid profile for multi-day simulations (same as main simulation)
            daily_minutes = 24 * 60
            sim_duration_min = sim_duration * 60
            
            if len(margin_curve) < sim_duration_min:
                # Extend the profile to cover the full simulation duration
                num_days_needed = sim_duration_min // daily_minutes + (1 if sim_duration_min % daily_minutes > 0 else 0)
                extended_profile = []
                for day in range(num_days_needed):
                    extended_profile.extend(margin_curve)
                margin_curve = np.array(extended_profile[:sim_duration_min])
            
            # Apply optimization strategies to grid profile (EXACTLY like normal simulation)
            adjusted_grid_profile = margin_curve.copy()
            
            # Apply PV + Battery optimization if enabled (charging during day, discharging during evening)
            pv_battery_support_adjusted = np.zeros_like(margin_curve)
            pv_battery_charge_curve = np.zeros_like(margin_curve)
            pv_direct_support_curve = np.zeros_like(margin_curve)
            pv_battery_discharge_curve = np.zeros_like(margin_curve)
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
                    total_evs_for_pv = current_cars
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
                    
                    for minute in range(len(margin_curve)):
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
            grid_battery_charge_curve = np.zeros_like(margin_curve)
            grid_battery_discharge_curve = np.zeros_like(margin_curve)
            if 'grid_battery' in active_strategies:
                grid_battery_adoption_percent = st.session_state.optimization_strategy.get('grid_battery_adoption_percent', 0)
                grid_battery_capacity = st.session_state.optimization_strategy.get('grid_battery_capacity', 0)
                grid_battery_max_rate = st.session_state.optimization_strategy.get('grid_battery_max_rate', 0)
                grid_battery_charge_start_hour = st.session_state.optimization_strategy.get('grid_battery_charge_start_hour', 7)
                grid_battery_charge_duration = st.session_state.optimization_strategy.get('grid_battery_charge_duration', 8)
                grid_battery_discharge_start_hour = st.session_state.optimization_strategy.get('grid_battery_discharge_start_hour', 20)
                grid_battery_discharge_duration = st.session_state.optimization_strategy.get('grid_battery_discharge_duration', 4)
                
                if grid_battery_adoption_percent > 0 and grid_battery_capacity > 0 and grid_battery_max_rate > 0:
                    total_evs_for_grid_battery = current_cars
                    grid_battery_evs = int(total_evs_for_grid_battery * grid_battery_adoption_percent / 100)
                    
                    # Use the same simple logic as PV battery
                    total_charge_power = grid_battery_evs * grid_battery_max_rate
                    total_discharge_power = grid_battery_evs * grid_battery_max_rate
                    
                    # Create charging curve (reduces available capacity) - only during charging hours
                    charge_start_minute = grid_battery_charge_start_hour * 60
                    charge_duration_minutes = int(grid_battery_charge_duration * 60)
                    charge_end_minute = charge_start_minute + charge_duration_minutes
                    
                    # Create discharging curve (increases available capacity) - only during discharging hours
                    discharge_start_minute = grid_battery_discharge_start_hour * 60
                    discharge_duration_minutes = int(grid_battery_discharge_duration * 60)
                    discharge_end_minute = discharge_start_minute + discharge_duration_minutes
                    
                    for minute in range(len(margin_curve)):
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
            v2g_discharge_curve = np.zeros_like(margin_curve)
            if 'v2g' in active_strategies:
                v2g_adoption_percent = st.session_state.optimization_strategy.get('v2g_adoption_percent', 0)
                v2g_max_discharge_rate = st.session_state.optimization_strategy.get('v2g_max_discharge_rate', 0)
                v2g_discharge_start_hour = st.session_state.optimization_strategy.get('v2g_discharge_start_hour', 20)
                v2g_discharge_duration = st.session_state.optimization_strategy.get('v2g_discharge_duration', 2)
                
                if v2g_adoption_percent > 0 and v2g_max_discharge_rate > 0:
                    total_v2g_evs = int(current_cars * v2g_adoption_percent / 100)
                    total_v2g_discharge_power = total_v2g_evs * v2g_max_discharge_rate
                    discharge_start_minute = v2g_discharge_start_hour * 60
                    discharge_duration_minutes = int(v2g_discharge_duration * 60)
                    discharge_end_minute = discharge_start_minute + discharge_duration_minutes
                    
                    for minute in range(len(margin_curve)):
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
            
            # Apply less conservative margin to the adjusted grid profile for capacity analysis
            final_grid_profile = adjusted_grid_profile * capacity_margin
            
            # Set grid power limit for simulation (EXACTLY like normal simulation)
            if grid_mode == "Grid Constrained":
                grid_power_limit = final_grid_profile
            else:
                grid_power_limit = None
            
            # Calculate arrival times (same as main simulation)
            arrival_times = []
            
            # Check if Time of Use is enabled
            time_of_use_enabled = ('smart_charging' in active_strategies and 
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
                    
                    # Calculate how many days the simulation runs
                    daily_minutes = 24 * 60
                    num_days = sim_duration * 60 // daily_minutes + (1 if sim_duration * 60 % daily_minutes > 0 else 0)
                    
                    # Pre-calculate all arrival times for Time of Use (repeating daily)
                    for day in range(num_days):
                        day_offset_minutes = day * daily_minutes
                        
                        for period_name, periods in period_groups.items():
                            if periods:
                                # Get adoption for this period type
                                period_adoption = periods[0]['adoption']
                                
                                if period_adoption > 0:
                                    # Calculate total EVs for this period type
                                    total_evs_for_type = int(current_cars * period_adoption / 100)
                                    
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
                if time_peaks and len(time_peaks) > 0:
                    peak = time_peaks[0]  # Use first peak
                    peak_mean = peak['time'] * 60  # Convert hours to minutes
                    peak_span = peak['span'] * 60  # Convert hours to minutes
                    sigma = peak_span
                    
                    peak_arrivals = np.random.normal(peak_mean, sigma, current_cars)
                    arrival_times.extend(peak_arrivals)
                else:
                    # Fallback to default values
                    peak_arrivals = np.random.normal(12 * 60, 4 * 60, current_cars)
                    arrival_times.extend(peak_arrivals)
            
            # Finalize arrival times (same as main simulation)
            arrival_times = np.array(arrival_times)
            arrival_times = np.clip(arrival_times, 0, sim_duration * 60 - 60)
            arrival_times.sort()
            
            # Add V2G recharge EVs to arrival times (same as main simulation)
            if 'v2g' in active_strategies:
                v2g_adoption_percent = st.session_state.optimization_strategy.get('v2g_adoption_percent', 0)
                v2g_recharge_arrival_hour = st.session_state.optimization_strategy.get('v2g_recharge_arrival_hour', 26)
                
                if v2g_adoption_percent > 0:
                    total_v2g_evs = int(current_cars * v2g_adoption_percent / 100)
                    
                    # Calculate realistic discharge and recharge EVs
                    ev_capacity = ev_config['capacity']
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
            
            sim = SimulationSetup(
                ev_counts=ev_counts,
                charger_counts=charger_counts,
                sim_duration=sim_duration * 60,
                arrival_time_mean=12 * 60,  # Not used since we manually schedule
                arrival_time_span=4 * 60,   # Not used since we manually schedule
                grid_power_limit=grid_power_limit
            )
            
            # Manually create EVs and schedule with pre-calculated arrival times (EXACTLY like normal simulation)
            sim.evs = []
            ev_capacity = ev_config['capacity']
            ev_ac = ev_config['AC']
            
            # Calculate how many V2G recharge EVs we have
            v2g_recharge_count = 0
            if 'v2g' in active_strategies:
                v2g_adoption_percent = st.session_state.optimization_strategy.get('v2g_adoption_percent', 0)
                if v2g_adoption_percent > 0:
                    v2g_recharge_count = int(len(arrival_times) * v2g_adoption_percent / 100)
            
            # Calculate how many PV battery EVs we have
            pv_battery_count = 0
            if 'pv_battery' in active_strategies:
                pv_adoption_percent = st.session_state.optimization_strategy.get('pv_adoption_percent', 0)
                if pv_adoption_percent > 0:
                    pv_battery_count = int(len(arrival_times) * pv_adoption_percent / 100)
            
            for i, arrival_time in enumerate(arrival_times):
                ev_name = f"Capacity_EV_{i+1}"
                
                # Determine EV types
                is_v2g_recharge = i >= (len(arrival_times) - v2g_recharge_count)
                is_pv_battery = i >= (len(arrival_times) - v2g_recharge_count - pv_battery_count) and i < (len(arrival_times) - v2g_recharge_count)
                
                # Set SOC based on EV type (same as main simulation)
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
                    soc = 0.2 + 0.3  # 30% higher SOC due to daytime charging
                    soc = min(0.9, soc)
                else:
                    # Regular EVs start at configured SOC
                    soc = 0.2
                
                ev = EV(name=ev_name, battery_capacity=ev_capacity, max_charging_power=ev_ac, soc=soc)
                ev.ev_type = 'dynamic_ev'
                ev.is_pv_battery = is_pv_battery
                sim.evs.append(ev)
            
            # Schedule EVs with pre-calculated arrival times (same as main simulation)
            for i, ev in enumerate(sim.evs):
                sim.env.process(sim._ev_arrival(ev, arrival_times[i]))
            
            sim.env.run(until=sim_duration * 60)
            
            # Process results
            load_curve = sim.load_curve
            
            # Convert final grid profile to 1-minute intervals to match load curve
            final_grid_profile_1min = np.repeat(final_grid_profile, 15)  # Convert 15-min to 1-min
            min_points = min(len(load_curve), len(final_grid_profile_1min))
            load_curve = load_curve[:min_points]
            margin_curve = final_grid_profile_1min[:min_points]
            
            available_loads = margin_curve - load_curve
            if available_loads.ndim > 1:
                available_loads = available_loads.flatten()
            min_available = float(np.min(available_loads))
            
            if np.sum(load_curve) <= 0:
                st.error("❌ No load generated by current cars - simulation may have failed")
                return None
            
            bottleneck_index = np.argmin(available_loads)
            ev_load_at_bottleneck = load_curve[bottleneck_index]
            scaling_factor = (min_available + ev_load_at_bottleneck) / ev_load_at_bottleneck if ev_load_at_bottleneck > 0 else 0
            max_cars = int(current_cars * scaling_factor)
            

            
            current_cars = max_cars
            final_max_cars = max_cars
            
            st.write(f"**Maximum EVs found: {final_max_cars}**")
            
        except Exception as e:
            st.error(f"❌ Error during capacity analysis: {e}")
            raise
        finally:
            # Restore original models (EXACTLY like normal simulation)
            EV_MODELS.clear()
            EV_MODELS.update(original_ev_models)
            CHARGER_MODELS.clear()
            CHARGER_MODELS.update(original_charger_models)
    
    return final_max_cars 