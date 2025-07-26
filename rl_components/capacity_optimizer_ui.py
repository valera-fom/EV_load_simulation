import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

def create_capacity_optimizer_ui():
    """Create the capacity optimizer UI in its own section."""
    
    st.write("Use trained Reinforcement Learning model to optimize EV charging parameters for maximum capacity.")
    
    # Use existing margin slider from session state (consistent with other parameters)
    margin_ratio = st.session_state.get('available_load_fraction', 0.8)
    
    # Only check RL availability when button is clicked
    if st.button("🤖 Predict with RL Model", type="primary"):
        # Lazy import heavy RL components only when needed
        try:
            from stable_baselines3 import PPO
            st.success("✅ Stable-Baselines3 is available")
            
            # Try to import the optimizer
            try:
                from rl_components.models.ppo_capacity_optimizer import create_capacity_optimizer
                optimizer = create_capacity_optimizer()
                model_info = optimizer.get_model_info()
                
                # Model status
                if model_info['is_trained']:
                    st.success("✅ Model is trained and ready")
                    st.info("🚧 RL Prediction feature coming soon!")
                    st.write("This will use the trained RL model to predict optimal parameters.")
                    st.write("The model will suggest parameters to maximize capacity while following the margin curve.")
                else:
                    st.warning("⚠️ Model needs training")
                    st.info("💡 To train the model, run: `python train_rl_model.py`")
                    
            except Exception as e:
                st.warning("⚠️ RL components not fully loaded")
                st.info(f"💡 Error: {e}")
                
        except ImportError:
            st.error("❌ Stable-Baselines3 is not available. Please install it first.")
            st.code("pip install stable-baselines3")

def _get_current_simulation_config() -> Dict[str, Any]:
    """Get current simulation configuration from session state."""
    config = {
        'ev_counts': st.session_state.get('ev_counts', {'dynamic_ev': 100}),
        'charger_counts': st.session_state.get('charger_counts', {'AC': 50, 'DC': 10}),
        'sim_duration': 24 * 60,  # 24 hours in minutes
        'arrival_time_mean': 12 * 60,  # 12 hours in minutes
        'arrival_time_span': 4 * 60,  # 4 hours in minutes
        'grid_power_limit': None
    }
    
    # Add optimization strategy parameters
    if 'optimization_strategy' in st.session_state:
        config.update(st.session_state.optimization_strategy)
    
    return config

def _get_margin_curve_from_ratio(margin_ratio: float) -> np.ndarray:
    """Get margin curve based on margin ratio (0-1)."""
    time_points = 1440  # 24 hours * 60 minutes
    
    # Get current load curve to base margin on
    # For now, create a simple base load curve
    base_load = 100  # Base load in kW
    
    # Create margin curve based on ratio
    # margin_ratio = 0.8 means use 80% of load as margin (20% offset)
    margin_curve = np.ones(time_points) * (base_load * margin_ratio)
    
    # Add some variation to make it more realistic
    time_hours = np.arange(time_points) / 60
    margin_curve += 20 * np.sin(2 * np.pi * time_hours / 24)  # Daily variation
    
    return margin_curve

def _apply_optimized_parameters(params: Dict[str, float]):
    """Apply optimized parameters to session state."""
    if 'optimization_strategy' not in st.session_state:
        st.session_state.optimization_strategy = {}
    
    # Update optimization strategy with optimized parameters
    for param, value in params.items():
        st.session_state.optimization_strategy[param] = value
    
    # Clear simulation results to force re-run with new parameters
    st.session_state.simulation_just_run = False 