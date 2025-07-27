#!/usr/bin/env python3
"""
Bayesian Optimizer UI for Streamlit
Replaces RL UI with Bayesian Optimization interface
"""

import streamlit as st
import numpy as np
import pandas as pd
from pages.components.bayesian_optimizer import BayesianOptimizer
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
            # For Real Dataset, try to use existing data or fall back to synthetic
            if power_values is None or len(power_values) == 0:
                print("🔍 Debug: No real dataset loaded. Falling back to synthetic generation...")
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
                    print("✅ Synthetic load curve generated!")
                except Exception as e:
                    print(f"❌ Error generating synthetic data: {e}")
                    return np.zeros(2880)
        
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
        car_count = int(params['car_count'])
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
        
        # Pre-calculate all arrival times for optimized simulation (EXACTLY like main simulation)
        arrival_times = []
        
        # Check if Time of Use is enabled
        time_of_use_enabled = ('smart_charging' in active_strategies and 
                              params.get('tou_super_offpeak_adoption', 0) > 0)
        
        if time_of_use_enabled:
            # Create Time of Use periods based on optimized parameters
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
        
        # Ensure we don't have more arrival times than EVs
        if len(arrival_times) > total_evs:
            print(f"🔍 Debug: Too many arrival times ({len(arrival_times)}) for {total_evs} EVs, truncating...")
            arrival_times = arrival_times[:total_evs]
        elif len(arrival_times) < total_evs:
            print(f"🔍 Debug: Too few arrival times ({len(arrival_times)}) for {total_evs} EVs, adding random times...")
            # Add random arrival times to reach total_evs
            additional_times = np.random.uniform(0, 48 * 60 - 60, total_evs - len(arrival_times))
            arrival_times = np.concatenate([arrival_times, additional_times])
            arrival_times.sort()
        
        print(f"🔍 Debug: Generated {len(arrival_times)} arrival times for {total_evs} EVs")
        
        # Debug TOU values
        if time_of_use_enabled and tou_periods:
            print(f"🔍 Debug: TOU periods being used:")
            for period in tou_periods:
                print(f"  {period['name']}: {period['adoption']}% ({period['start']}-{period['end']}h)")
        
        # Create simulation setup (EXACTLY like main simulation)
        sim = SimulationSetup(
            ev_counts=ev_counts,
            charger_counts=charger_counts,
            sim_duration=48*60,  # 48 hours in minutes
            time_step=1,
            arrival_time_mean=12*60,
            arrival_time_span=4*60,
            grid_power_limit=power_values * available_load_fraction,  # Margin curve
            verbose=False,  # Disable verbose to reduce terminal output
            silent=True     # Enable silent to suppress all simulation logs
        )
        
        # Set grid power profile (base grid data)
        sim.grid_power_profile = power_values
        
        # Clear EVs and create them manually (EXACTLY like main simulation)
        sim.evs = []
        
        # Create EVs manually (EXACTLY like main simulation)
        for i, arrival_time in enumerate(arrival_times):
            ev_name = f"Custom_EV_{i+1}"
            
            # Determine EV types
            is_v2g_recharge = False  # Simplified for now
            is_pv_battery = False    # Simplified for now
            
            # Set SOC based on EV type (EXACTLY like main simulation)
            if is_v2g_recharge:
                # V2G EVs have lower SOC since they discharged earlier
                soc = 0.1  # Simplified: 10% SOC for V2G recharge EVs
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

def create_bayesian_optimizer_ui():
    """Create the Bayesian optimizer UI."""
    
    st.write("--------------------------------")
    # Bayesian Capacity Optimizer
    st.write("🤖 **Bayesian Capacity Optimizer**")
    
    # Optimization quality dropdown
    optimization_quality = st.selectbox(
        "🎯 Optimization Quality",
        options=["Quick Test", "Standard", "High Quality"],
        index=1,
        help="Choose optimization quality and iteration count"
    )
    
    # Set iterations and initial points based on quality
    if optimization_quality == "Quick Test":
        n_iterations = 60  # Doubled from 30
        initial_points = 30  # Doubled from 15
    elif optimization_quality == "Standard":
        n_iterations = 120  # Doubled from 60
        initial_points = 60  # Doubled from 30
    else:  # High Quality
        n_iterations = 240  # Doubled from 120
        initial_points = 120  # Doubled from 60
    
    if st.button("🎯 Run Bayesian Optimization", type="primary"):
        # Check if smart charging is enabled
        active_strategies = st.session_state.get('active_strategies', [])
        
        if 'smart_charging' not in active_strategies:
            st.warning("⚠️ **Please enable Time-of-Use optimization in the Optimization Strategies section to run Bayesian optimization.**")
            return
    
        # Build active parameters based on active strategies
        active_params = {}
        
        # Only add parameters that are actually being optimized
        if 'smart_charging' in active_strategies:
            active_params.update({
                'tou_super_offpeak_adoption': (5, 40),
                'tou_offpeak_adoption': (10, 50),
                'tou_midpeak_adoption': (5, 40),
                'tou_peak_adoption': (10, 50)
            })
        
        # Always include car_count
        active_params['car_count'] = (1, 200)
        
        # Create Bayesian optimizer
        optimizer = BayesianOptimizer(
            simulation_function=simulation_function,
            parameter_bounds=active_params,
            n_iterations=n_iterations,
            initial_points=initial_points
        )
        
        # Add progress tracking
        st.write("🚀 **Starting Bayesian Optimization...**")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Track progress
        def progress_callback(iteration, total_iterations, best_reward):
            progress = min(1.0, (iteration + 1) / total_iterations)  # Ensure progress doesn't exceed 1.0
            progress_bar.progress(progress)
            status_text.write(f"🔄 **Iteration {iteration + 1}/{total_iterations}** ({progress*100:.1f}% complete)")
            status_text.write(f"🎯 **Current best reward: {best_reward:.2f}**")
        
        # Run optimization with progress tracking
        results = optimizer.optimize(progress_callback=progress_callback)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state for display after rerun
        if results and 'best_parameters' in results:
            st.session_state.bayesian_results = results
            st.session_state.bayesian_active_strategies = active_strategies.copy()
            st.session_state.bayesian_optimization_completed = True
        
        # Display results
        if results and 'best_parameters' in results:
            st.write("🎯 **Best Parameters**")
            
            # Get the best parameters and normalize TOU values for display
            best_params = results['best_parameters'].copy()
            
            # Normalize TOU values for display (same logic as in simulation)
            if 'smart_charging' in active_strategies:
                tou_super_offpeak = best_params.get('tou_super_offpeak_adoption', 25)
                tou_offpeak = best_params.get('tou_offpeak_adoption', 25)
                tou_midpeak = best_params.get('tou_midpeak_adoption', 25)
                tou_peak = best_params.get('tou_peak_adoption', 25)
                
                # Ensure all values are positive
                tou_super_offpeak = max(0, tou_super_offpeak)
                tou_offpeak = max(0, tou_offpeak)
                tou_midpeak = max(0, tou_midpeak)
                tou_peak = max(0, tou_peak)
                
                # Normalize to ensure sum = 100%
                total_tou = tou_super_offpeak + tou_offpeak + tou_midpeak + tou_peak
                if total_tou > 0:
                    tou_super_offpeak = (tou_super_offpeak / total_tou) * 100
                    tou_offpeak = (tou_offpeak / total_tou) * 100
                    tou_midpeak = (tou_midpeak / total_tou) * 100
                    tou_peak = (tou_peak / total_tou) * 100
                else:
                    tou_super_offpeak = tou_offpeak = tou_midpeak = tou_peak = 25
                
                # Round to 2 decimal places to avoid floating point precision issues
                tou_super_offpeak = round(tou_super_offpeak, 2)
                tou_offpeak = round(tou_offpeak, 2)
                tou_midpeak = round(tou_midpeak, 2)
                tou_peak = round(tou_peak, 2)
                
                # Update display values with normalized percentages
                best_params['tou_super_offpeak_adoption'] = f"{tou_super_offpeak:.2f}%"
                best_params['tou_offpeak_adoption'] = f"{tou_offpeak:.2f}%"
                best_params['tou_midpeak_adoption'] = f"{tou_midpeak:.2f}%"
                best_params['tou_peak_adoption'] = f"{tou_peak:.2f}%"
                
                # Update the time_of_use_timeline that the UI sliders read from
                if 'time_of_use_timeline' in st.session_state:
                    timeline = st.session_state.time_of_use_timeline
                    if len(timeline['periods']) >= 4:
                        # Update each period's adoption percentage
                        timeline['periods'][0]['adoption'] = tou_super_offpeak  # Super Off-Peak
                        timeline['periods'][1]['adoption'] = tou_offpeak        # Off-Peak
                        timeline['periods'][2]['adoption'] = tou_midpeak        # Mid-Peak
                        timeline['periods'][3]['adoption'] = tou_peak          # Peak
            
            # Display results in a table
            st.write("**📊 Optimization Results:**")
            
            # Format other parameters with 2 decimal places
            best_params['car_count'] = int(best_params.get('car_count', 100))
            
            # Display as dataframe (better format)
            import pandas as pd
            df = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
            st.dataframe(df, use_container_width=True)
            
            # Show actual reward value
            actual_reward = results.get('best_reward', -1000.0)
            st.write(f"🎯 **Best reward: {actual_reward:.2f}**")
            
            # Add debug info if reward is -1000
            if actual_reward <= -1000:
                st.warning("⚠️ **Warning: Optimization may not be working properly. Reward of -1000 suggests simulation issues.**")
                st.info("💡 **Debug Info:** Check if smart charging is enabled and simulation parameters are valid.")
            
            # Show iterations info
            st.write(f"🔄 **Total Iterations:** {results.get('n_iterations', 'N/A')}")
            
            # Automatically apply the optimized parameters
            if 'smart_charging' in active_strategies:
                # Apply TOU adoption percentages to the optimization strategy
                st.session_state.optimization_strategy['tou_super_offpeak_adoption_percent'] = round(float(best_params['tou_super_offpeak_adoption'].rstrip('%')), 2)
                st.session_state.optimization_strategy['tou_offpeak_adoption_percent'] = round(float(best_params['tou_offpeak_adoption'].rstrip('%')), 2)
                st.session_state.optimization_strategy['tou_midpeak_adoption_percent'] = round(float(best_params['tou_midpeak_adoption'].rstrip('%')), 2)
                st.session_state.optimization_strategy['tou_peak_adoption_percent'] = round(float(best_params['tou_peak_adoption'].rstrip('%')), 2)
                
                # Also update the time_of_use_periods with the optimized values
                optimized_tou_periods = [
                    {'name': 'super_offpeak', 'start': 0, 'end': 6, 'adoption': round(float(best_params['tou_super_offpeak_adoption'].rstrip('%')), 2)},
                    {'name': 'offpeak', 'start': 6, 'end': 17, 'adoption': round(float(best_params['tou_offpeak_adoption'].rstrip('%')), 2)},
                    {'name': 'midpeak', 'start': 17, 'end': 20, 'adoption': round(float(best_params['tou_midpeak_adoption'].rstrip('%')), 2)},
                    {'name': 'peak', 'start': 20, 'end': 24, 'adoption': round(float(best_params['tou_peak_adoption'].rstrip('%')), 2)}
                ]
                st.session_state.optimization_strategy['time_of_use_periods'] = optimized_tou_periods
                
                # Update the time_of_use_timeline that the UI sliders read from
                if 'time_of_use_timeline' in st.session_state:
                    timeline = st.session_state.time_of_use_timeline
                    if len(timeline['periods']) >= 4:
                        # Update each period's adoption percentage
                        timeline['periods'][0]['adoption'] = round(float(best_params['tou_super_offpeak_adoption'].rstrip('%')), 2)  # Super Off-Peak
                        timeline['periods'][1]['adoption'] = round(float(best_params['tou_offpeak_adoption'].rstrip('%')), 2)      # Off-Peak
                        timeline['periods'][2]['adoption'] = round(float(best_params['tou_midpeak_adoption'].rstrip('%')), 2)      # Mid-Peak
                        timeline['periods'][3]['adoption'] = round(float(best_params['tou_peak_adoption'].rstrip('%')), 2)        # Peak
            
            # Update EV and charger counts
            car_count = best_params['car_count']
            st.session_state.total_evs = car_count
            st.session_state.charger_config['ac_count'] = car_count
            
            # Update time control EV count
            if 'time_peaks' in st.session_state and len(st.session_state.time_peaks) > 0:
                st.session_state.time_peaks[0]['quantity'] = car_count
            
            st.success("✅ Optimized parameters applied automatically!")
            st.rerun()
    
    # Display results after rerun if optimization was completed
    if st.session_state.get('bayesian_optimization_completed', False):
        results = st.session_state.get('bayesian_results', {})
        active_strategies = st.session_state.get('bayesian_active_strategies', [])
        
        if results and 'best_parameters' in results:
            st.write("🎯 **Optimization Results**")
            
            # Get the best parameters and normalize TOU values for display
            best_params = results['best_parameters'].copy()
            
            # Normalize TOU values for display (same logic as in simulation)
            if 'smart_charging' in active_strategies:
                tou_super_offpeak = best_params.get('tou_super_offpeak_adoption', 25)
                tou_offpeak = best_params.get('tou_offpeak_adoption', 25)
                tou_midpeak = best_params.get('tou_midpeak_adoption', 25)
                tou_peak = best_params.get('tou_peak_adoption', 25)
                
                # Ensure all values are positive
                tou_super_offpeak = max(0, tou_super_offpeak)
                tou_offpeak = max(0, tou_offpeak)
                tou_midpeak = max(0, tou_midpeak)
                tou_peak = max(0, tou_peak)
                
                # Normalize to ensure sum = 100%
                total_tou = tou_super_offpeak + tou_offpeak + tou_midpeak + tou_peak
                if total_tou > 0:
                    tou_super_offpeak = (tou_super_offpeak / total_tou) * 100
                    tou_offpeak = (tou_offpeak / total_tou) * 100
                    tou_midpeak = (tou_midpeak / total_tou) * 100
                    tou_peak = (tou_peak / total_tou) * 100
                else:
                    tou_super_offpeak = tou_offpeak = tou_midpeak = tou_peak = 25
                
                # Round to 2 decimal places to avoid floating point precision issues
                tou_super_offpeak = round(tou_super_offpeak, 2)
                tou_offpeak = round(tou_offpeak, 2)
                tou_midpeak = round(tou_midpeak, 2)
                tou_peak = round(tou_peak, 2)
                
                # Update display values with normalized percentages
                best_params['tou_super_offpeak_adoption'] = f"{tou_super_offpeak:.2f}%"
                best_params['tou_offpeak_adoption'] = f"{tou_offpeak:.2f}%"
                best_params['tou_midpeak_adoption'] = f"{tou_midpeak:.2f}%"
                best_params['tou_peak_adoption'] = f"{tou_peak:.2f}%"
                
                # Update the time_of_use_timeline that the UI sliders read from
                if 'time_of_use_timeline' in st.session_state:
                    timeline = st.session_state.time_of_use_timeline
                    if len(timeline['periods']) >= 4:
                        # Update each period's adoption percentage
                        timeline['periods'][0]['adoption'] = tou_super_offpeak  # Super Off-Peak
                        timeline['periods'][1]['adoption'] = tou_offpeak        # Off-Peak
                        timeline['periods'][2]['adoption'] = tou_midpeak        # Mid-Peak
                        timeline['periods'][3]['adoption'] = tou_peak          # Peak
            
            # Display results in a table
            st.write("**📊 Optimization Results:**")
            
            # Format other parameters with 2 decimal places
            best_params['car_count'] = int(best_params.get('car_count', 100))
            
            # Display as dataframe (better format)
            import pandas as pd
            df = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
            st.dataframe(df, use_container_width=True)
            
            # Show actual reward value
            actual_reward = results.get('best_reward', -1000.0)
            st.write(f"🎯 **Best reward: {actual_reward:.2f}**")
            
            # Add debug info if reward is -1000
            if actual_reward <= -1000:
                st.warning("⚠️ **Warning: Optimization may not be working properly. Reward of -1000 suggests simulation issues.**")
                st.info("💡 **Debug Info:** Check if smart charging is enabled and simulation parameters are valid.")
            
            # Show iterations info
            st.write(f"🔄 **Total Iterations:** {results.get('n_iterations', 'N/A')}")
            
            # Clear the completion flag
            st.session_state.bayesian_optimization_completed = False 

if __name__ == "__main__":
    create_bayesian_optimizer_ui() 