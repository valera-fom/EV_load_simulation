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
        st.error("❌ Please configure Time of Use periods first")
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
    """Run the dynamic capacity optimization."""
    # Get configuration from session state
    ev_config = st.session_state.dynamic_ev
    charger_config = st.session_state.charger_config
    tou_periods = st.session_state.optimization_strategy.get('time_of_use_periods', [])
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
        st.error("❌ No valid dataset available. Please load a dataset or generate synthetic data first.")
        raise ValueError("No valid dataset available")
    

    
    # Calculate margin curve
    try:
        power_values = np.array(power_values, dtype=float)
        margin_curve = np.repeat(power_values, 15).astype(float) * available_load_fraction
    except Exception as e:
        print(f"Error converting power_values: {e}")
        print(f"power_values: {power_values}")
        raise ValueError(f"Invalid power_values data: {e}")
    
    # Extend to 48 hours if needed
    sim_duration_min = 48 * 60
    if len(margin_curve) < sim_duration_min:
        num_repeats = sim_duration_min // len(margin_curve) + 1
        margin_curve = np.tile(margin_curve, num_repeats)[:sim_duration_min]
    
    # Convert to time step intervals
    step_duration = time_step
    num_steps = sim_duration_min // step_duration
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
    
    return {
        'total_cars': total_cars,
        'cars_added_per_step': cars_added_per_step,
        'active_cars_per_step': active_cars_per_step,
        'period_results': period_results,
        'margin_curve': margin_curve_steps,
        'final_load': current_load
    }

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
    
    # Calculate percentages
    period_data = []
    total_points = sum(results['period_results'].values())
    
    for period_name, points in results['period_results'].items():
        percentage = (points / total_points * 100) if total_points > 0 else 0
        period_data.append({
            'Period': period_name,
            'Percentage': f"{percentage:.1f}%"
        })
    
    # Add total cars row
    total_cars = sum(results['cars_added_per_step'])
    # Divide by 2 since optimizer runs for 48 hours but normal simulation uses 24 hours
    total_cars_24h = total_cars // 2
    period_data.append({
        'Period': 'Total Cars',
        'Percentage': f"{total_cars_24h}"
    })
    
    # Create DataFrame and display
    df = pd.DataFrame(period_data)
    st.dataframe(df, use_container_width=True)
    
    # Display current optimization results if just completed
    if st.session_state.get('dynamic_optimization_just_completed', False):
        st.session_state.dynamic_optimization_just_completed = False
    
    # Now apply the optimization results
    # Update TOU periods with optimized percentages
    tou_periods = st.session_state.optimization_strategy.get('time_of_use_periods', [])
    
    for period in tou_periods:
        period_name = period['name']
        if period_name in results['period_results']:
            points = results['period_results'][period_name]
            percentage = (points / total_points * 100) if total_points > 0 else 0
            period['adoption'] = round(percentage, 1)
    
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
    
    st.session_state.dynamic_optimization_results = {
        'total_cars': total_cars_24h,
        'period_results': results['period_results'],
        'tou_periods': tou_periods,
        'margin_curve': safe_tolist(results['margin_curve']),
        'final_load': safe_tolist(results['final_load']),
        'cars_added_per_step': safe_tolist(results['cars_added_per_step']),
        'active_cars_per_step': safe_tolist(results['active_cars_per_step']),
        'time_step': time_step
    }
    
    # Store optimized TOU values in session state (like Bayesian optimizer)
    optimized_tou_values = {}
    for period in tou_periods:
        if period['name'] == 'Super Off-Peak':
            optimized_tou_values['tou_super_offpeak'] = period['adoption']
        elif period['name'] == 'Off-Peak':
            optimized_tou_values['tou_offpeak'] = period['adoption']
        elif period['name'] == 'Mid-Peak':
            optimized_tou_values['tou_midpeak'] = period['adoption']
        elif period['name'] == 'Peak':
            optimized_tou_values['tou_peak'] = period['adoption']
    
    st.session_state.optimized_tou_values = optimized_tou_values
    
    # Apply car count results (like max cars optimizer)
    st.session_state.total_evs = total_cars_24h
    st.session_state.charger_config['ac_count'] = total_cars_24h
    if 'time_peaks' in st.session_state and len(st.session_state.time_peaks) > 0:
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

 