#!/usr/bin/env python3
"""
Dataset Manager for EV Simulation
Handles both real dataset loading and synthetic data generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def render_dataset_selection_ui():
    """Render the dataset selection UI with 100% preserved functionality and design."""
    
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
        st.write("---")
        st.write("**⚡ Available Load Fraction:**")
        
        # Get the current value from session state, ensuring it's a single integer
        current_fraction = st.session_state.get('available_load_fraction', 80)
        if isinstance(current_fraction, list):
            current_fraction = current_fraction[0] if current_fraction else 80
        elif not isinstance(current_fraction, (int, float)):
            current_fraction = 80
            
        available_load_fraction = st.slider(
            "Available Load Fraction (%)",
            min_value=10,
            max_value=100,
            value=int(current_fraction),
            step=5,
            help="Percentage of total load that can be used for EV charging (10-100%)"
        )
        st.session_state.available_load_fraction = available_load_fraction
        
        # Grid Constraint Mode
        st.write("**Grid Constraint Mode:**")
        grid_mode = st.radio("Mode", ["Reference Only", "Grid Constrained"], 
                            help="Reference Only: Show actual load with grid limit as reference line. Grid Constrained: Load cannot exceed grid limit.")
        st.session_state.grid_mode = grid_mode
        
        # Return power_values, data_source, and grid_mode for use in main simulation
        return power_values, data_source, grid_mode 