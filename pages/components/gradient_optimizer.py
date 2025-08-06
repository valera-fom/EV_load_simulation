#!/usr/bin/env python3
"""
Gradient-Based Optimizer for EV Load Curve Optimization
Uses finite difference gradients to directly optimize TOU parameters
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class GradientOptimizer:
    """Gradient-based optimization for EV load curve optimization."""
    
    def __init__(self, simulation_function, parameter_bounds, n_iterations=50, learning_rate=0.1):
        self.simulation_function = simulation_function
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())
        self.n_parameters = len(self.parameter_names)
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._iteration_count = 0
        
        # Initialize margin curve for reward calculation
        self._setup_margin_curve()
        
        # Storage for observations
        self.X_observed = []
        self.y_observed = []
        self.best_reward = float('-inf')
        self.best_parameters = None
        
        # Optimization history
        self.optimization_history = []
    
    def _setup_margin_curve(self):
        """Setup the margin curve for reward calculation."""
        try:
            # Use the SAME approach as main simulation's extract_margin_curve_from_current_data()
            # Get current configuration
            available_load_fraction = st.session_state.get('available_load_fraction', 0.8)
            capacity_margin = min(0.95, available_load_fraction + 0.1)  # Add 0.1 but cap at 0.95
            
            # Get power values from session state
            power_values = st.session_state.get('power_values', None)
            
            if power_values is None or len(power_values) == 0:
                print("❌ CRITICAL: No power values available for margin curve setup")
                self.margin_curve = np.zeros(2880)
                return
            
            # Convert to numpy array if needed
            power_values = np.array(power_values)
            
            # Upsample 15-minute data to 1-minute intervals (EXACTLY like main simulation)
            margin_curve = np.repeat(power_values, 15).astype(float) * capacity_margin
            
            # Extend for 48 hours (2880 minutes) - EXACTLY like main simulation
            sim_duration_min = 48 * 60  # 48 hours
            
            if len(margin_curve) < sim_duration_min:
                # Repeat the profile to cover 48 hours
                num_repeats = sim_duration_min // len(margin_curve) + 1
                margin_curve = np.tile(margin_curve, num_repeats)[:sim_duration_min]
            else:
                margin_curve = margin_curve[:sim_duration_min]
            
            # Ensure margin_curve matches expected length
            if len(margin_curve) != 2880:
                print(f"❌ CRITICAL: Margin curve length {len(margin_curve)} != 2880 after processing")
                self.margin_curve = np.zeros(2880)
                return
            
            # Use the margin curve as calculated
            self.margin_curve = np.array(margin_curve)
            
            # Debug output removed for production
            
        except Exception as e:
            print(f"❌ CRITICAL: Error setting up margin curve: {e}")
            self.margin_curve = np.zeros(2880)
    
    def _calculate_reward(self, load_curve: np.ndarray, params: Dict[str, float]) -> float:
        """
        Calculate reward based on RMSE of non-normalized curves.
        Simplified reward function focusing on actual shape learning.
        """
        try:
            # Ignore first 5 hours (300 minutes) to avoid initialization effects
            start_minute = 300  # 5 hours * 60 minutes
            end_minute = 2880   # 48 hours * 60 minutes
            
            # Extract the relevant portion of the load curve (hours 5-48)
            load_curve_filtered = load_curve[start_minute:end_minute]
            margin_curve_filtered = self.margin_curve[start_minute:end_minute]
            
            # Debug output removed for production
            
            # Ensure we have valid data
            if len(load_curve_filtered) == 0 or len(margin_curve_filtered) == 0:
                print(f"❌ CRITICAL: Filtered curves are empty!")
                return -1000.0
            
            # Check if filtered load curve is all zeros
            if np.all(load_curve_filtered == 0):
                print(f"❌ CRITICAL: Filtered load curve is all zeros!")
                return -1000.0
            
            # Check for violations (hard constraints) FIRST
            initial_line_violation_count = 0
            margin_violation_count = 0
            
            # Check initial line violations (red curve)
            original_grid_profile = st.session_state.get('power_values', [])
            if len(original_grid_profile) > 0:
                if len(original_grid_profile) < len(load_curve_filtered):
                    original_grid_profile = np.tile(original_grid_profile, int(np.ceil(len(load_curve_filtered) / len(original_grid_profile))))
                original_grid_profile = original_grid_profile[:len(load_curve_filtered)]
                
                initial_line_violations = load_curve_filtered > original_grid_profile
                initial_line_violation_count = np.sum(initial_line_violations)
                
                # HARD CONSTRAINT: If ANY red curve violations, return terrible reward
                if initial_line_violation_count > 0:
                    print(f"❌ CRITICAL: {initial_line_violation_count} red curve violations detected!")
                    return -1000.0
            
            # Check margin violations
            margin_violations = load_curve_filtered > margin_curve_filtered
            margin_violation_count = np.sum(margin_violations)
            
            # SIMPLIFIED SHAPE LEARNING - Focus on what actually matters
            
            # 1. RMSE between curves (direct shape matching)
            raw_rmse = np.sqrt(np.mean((load_curve_filtered - margin_curve_filtered) ** 2))
            max_margin = np.max(margin_curve_filtered)
            
            # Normalize RMSE by max margin value to get 0-1 score (lower RMSE = better)
            rmse_score = max(0, 1.0 - (raw_rmse / max_margin)) if max_margin > 0 else 0
            
            # 2. Simple shape similarity (normalized curves correlation)
            # Normalize both curves to 0-1 range for fair comparison
            load_normalized = (load_curve_filtered - np.min(load_curve_filtered)) / (np.max(load_curve_filtered) - np.min(load_curve_filtered) + 1e-8)
            margin_normalized = (margin_curve_filtered - np.min(margin_curve_filtered)) / (np.max(margin_curve_filtered) - np.min(margin_curve_filtered) + 1e-8)
            
            # Calculate correlation of normalized curves (shape similarity)
            shape_correlation = np.corrcoef(load_normalized, margin_normalized)[0, 1]
            if np.isnan(shape_correlation):
                shape_correlation = 0.0
            
            # 3. Utilization (how much of the margin we're using)
            safe_utilization = np.mean(load_curve_filtered) / np.mean(margin_curve_filtered) if np.mean(margin_curve_filtered) > 0 else 0
            
            # Calculate expected utilization based on car count vs margin capacity
            # If we have 100 cars and margin can handle 500 cars, we expect 20% utilization
            # If we have 200 cars and margin can handle 200 cars, we expect 100% utilization
            expected_utilization = min(1.0, safe_utilization * 2.0)  # Allow up to 100% utilization
            
            # SIMPLIFIED REWARD FUNCTION - Focus on shape learning
            # 1. RMSE reward (direct curve matching)
            rmse_reward = rmse_score * 15.0  # Moderate weight
            
            # 2. Shape similarity reward (normalized curve correlation) - MOST IMPORTANT
            shape_reward = max(0, shape_correlation) * 100.0  # MUCH higher weight for shape learning
            
            # 3. Utilization reward (encourage using more capacity) - CRITICAL
            utilization_reward = expected_utilization * 30.0  # Use expected utilization
            
            # 4. Penalty for margin violations
            violation_penalty = margin_violation_count * 2.0  # Stronger penalty
            
            # 5. PENALTY for low utilization (force higher utilization)
            low_utilization_penalty = max(0, (0.3 - expected_utilization)) * 20.0  # Penalty if utilization < 30%
            
            # 6. PENALTY for poor shape learning (force shape matching)
            poor_shape_penalty = max(0, (0.2 - shape_correlation)) * 25.0  # Penalty if shape correlation < 0.2
            
            # 7. BONUS for high TOU percentages (encourage scaling up effective car count)
            total_tou = (params.get('tou_super_offpeak_adoption', 0) + 
                        params.get('tou_offpeak_adoption', 0) + 
                        params.get('tou_midpeak_adoption', 0) + 
                        params.get('tou_peak_adoption', 0))
            
            # Reward for higher total TOU (more effective cars)
            tou_scaling_bonus = max(0, (total_tou - 100) / 100) * 15.0  # Bonus for TOU > 100%
            
            # Calculate final reward with much stronger incentives
            reward = (rmse_reward + shape_reward + utilization_reward + tou_scaling_bonus - 
                     violation_penalty - low_utilization_penalty - poor_shape_penalty + 10.0)
            
            # Update iteration counter
            if hasattr(self, '_iteration_count'):
                self._iteration_count += 1
            else:
                self._iteration_count = 1
            
            # Enhanced progress output with shape learning metrics
            if self._iteration_count <= 5 or self._iteration_count % 5 == 0:
                print(f"🔄 Iteration {self._iteration_count}: Reward = {reward:.3f}")
                print(f"  📊 RMSE Score: {rmse_score:.3f}, Shape Corr: {shape_correlation:.3f}, Utilization: {safe_utilization*100:.1f}% (Expected: {expected_utilization*100:.1f}%)")
                print(f"  🎯 Rewards: RMSE={rmse_reward:.1f}, Shape={shape_reward:.1f}, Util={utilization_reward:.1f}, TOU_Bonus={tou_scaling_bonus:.1f}")
                print(f"  ⚠️ Penalties: Violations={violation_penalty:.1f}, LowUtil={low_utilization_penalty:.1f}, PoorShape={poor_shape_penalty:.1f}")
                print(f"  📈 Total TOU: {total_tou:.1f}% (Effective cars: {total_tou/100*25:.0f})")
                print(f"  ⚙️ TOU: Period1={params.get('tou_super_offpeak_adoption', 0):.1f}, Period2={params.get('tou_offpeak_adoption', 0):.1f}, Period3={params.get('tou_midpeak_adoption', 0):.1f}, Period4={params.get('tou_peak_adoption', 0):.1f}")
                if self._iteration_count <= 5:
                    print(f"  " + "="*50)
            
            return reward
            
        except Exception as e:
            print(f"❌ Reward calculation error: {e}")
            return -1000.0
    
    def _evaluate_point(self, params):
        """Evaluate a point by running simulation and calculating reward."""
        try:
            # Run simulation with these parameters
            load_curve = self.simulation_function(params)
            
            # Calculate reward
            reward = self._calculate_reward(load_curve, params)
            
            return reward
            
        except Exception as e:
            print(f"❌ Error evaluating point: {e}")
            return -1000.0
    
    def _finite_difference_gradient(self, params, epsilon=0.1):
        """Calculate gradients using finite differences."""
        gradients = []
        
        for param_name in self.parameter_names:
            # Create perturbed parameters
            params_plus = params.copy()
            params_minus = params.copy()
            
            # Perturb one parameter at a time
            params_plus[param_name] += epsilon
            params_minus[param_name] -= epsilon
            
            # Clip to bounds
            min_val, max_val = self.parameter_bounds[param_name]
            params_plus[param_name] = np.clip(params_plus[param_name], min_val, max_val)
            params_minus[param_name] = np.clip(params_minus[param_name], min_val, max_val)
            
            # Calculate finite difference
            f_plus = self._evaluate_point(params_plus)
            f_minus = self._evaluate_point(params_minus)
            
            gradient = (f_plus - f_minus) / (2 * epsilon)
            gradients.append(gradient)
            
            # Debug output removed for production
        
        # Debug output removed for production
        
        return np.array(gradients)
    
    def _clip_to_bounds(self, params):
        """Clip parameters to their bounds."""
        clipped_params = params.copy()
        for param_name, (min_val, max_val) in self.parameter_bounds.items():
            clipped_params[param_name] = np.clip(clipped_params[param_name], min_val, max_val)
        return clipped_params
    
    def optimize(self, progress_callback=None):
        """
        Run gradient-based optimization.
        
        Args:
            progress_callback: Optional callback function(iteration, total_iterations, best_reward)
            
        Returns:
            Dict with best parameters and reward
        """
        print(f"🚀 Starting Gradient-based optimization ({self.n_iterations} iterations)")
        
        # Initialize parameters (start with equal values)
        current_params = {}
        for param_name, (min_val, max_val) in self.parameter_bounds.items():
            # Start with 25% for each TOU parameter (equal starting point)
            current_params[param_name] = 25.0
        
        # Evaluate initial point
        initial_reward = self._evaluate_point(current_params)
        self.best_reward = initial_reward
        self.best_parameters = current_params.copy()
        
        print(f"📊 Initial reward: {initial_reward:.3f}")
        print(f"📊 Initial parameters: {current_params}")
        
        # Gradient ascent loop
        for iteration in range(self.n_iterations):
            # Calculate gradients using finite differences
            gradients = self._finite_difference_gradient(current_params)
            
            # Update parameters using gradient ascent
            param_names = list(self.parameter_bounds.keys())
            for i, param_name in enumerate(param_names):
                current_params[param_name] += self.learning_rate * gradients[i]
            
            # Clip to bounds
            current_params = self._clip_to_bounds(current_params)
            
            # Evaluate new point
            reward = self._evaluate_point(current_params)
            
            # Store observation
            param_values = [current_params[name] for name in self.parameter_names]
            self.X_observed.append(param_values)
            self.y_observed.append(reward)
            
            # Update best if better
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_parameters = current_params.copy()
                print(f"🎯 New best reward: {reward:.3f} with params: {current_params}")
            
            # Store optimization history
            self.optimization_history.append({
                'iteration': iteration,
                'params': current_params.copy(),
                'reward': reward,
                'gradients': gradients.copy(),
                'best_reward': self.best_reward
            })
            
            # Print progress (less frequent)
            if iteration % 10 == 0 or iteration == self.n_iterations - 1:
                print(f"📊 Iteration {iteration + 1}/{self.n_iterations}: Reward = {reward:.3f}, Best = {self.best_reward:.3f}")
                print(f"  ⚙️ TOU: Period1={current_params.get('tou_super_offpeak_adoption', 0):.1f}, Period2={current_params.get('tou_offpeak_adoption', 0):.1f}, Period3={current_params.get('tou_midpeak_adoption', 0):.1f}, Period4={current_params.get('tou_peak_adoption', 0):.1f}")
            
            if progress_callback:
                progress_callback(iteration, self.n_iterations, self.best_reward)
        
        print(f"✅ Gradient-based optimization completed!")
        print(f"🎯 Best reward: {self.best_reward:.3f}")
        print(f"📊 Final parameters: {self.best_parameters}")
        print(f"🔄 Total iterations: {self.n_iterations}")
        print(f"  " + "="*50)
        
        return {
            'best_parameters': self.best_parameters,
            'best_reward': self.best_reward,
            'optimization_history': self.optimization_history,
            'n_iterations': self.n_iterations
        }

def create_gradient_optimizer(num_periods: int = 4) -> GradientOptimizer:
    """Create a Gradient-based optimizer with parameter bounds for the specified number of periods."""
    
    # Parameter bounds (only TOU parameters, no car_count)
    # Use equal bounds for all periods to ensure equal adoption percentages
    # The bounds are set to ensure the optimizer searches within an equal distribution range
    
    # Define parameter mappings for different numbers of periods
    period_parameters = {
        'Period 1': 'tou_super_offpeak_adoption',
        'Period 2': 'tou_offpeak_adoption',
        'Period 3': 'tou_midpeak_adoption', 
        'Period 4': 'tou_peak_adoption',
        'Period 5': 'tou_peak_adoption'  # Map Period 5 to peak for compatibility
    }
    
    # Create parameter bounds based on the number of periods
    parameter_bounds = {}
    for i in range(1, min(num_periods + 1, 6)):  # Support up to 5 periods
        period_name = f'Period {i}'
        if period_name in period_parameters:
            param_name = period_parameters[period_name]
            parameter_bounds[param_name] = (20, 30)  # Equal bounds for all periods
    
    # Debug output removed for production
    
    return GradientOptimizer(parameter_bounds)

if __name__ == "__main__":
    # Test the Gradient Optimizer with different numbers of periods
    for num_periods in [2, 3, 4, 5]:
        # Debug output removed for production
        optimizer = create_gradient_optimizer(num_periods)
        # Debug output removed for production
        
        # Example simulation function (placeholder)
        def test_simulation(params):
            # Placeholder - would be replaced with actual simulation
            return np.random.normal(100, 20, 2880)  # 48-hour load curve
        
        # Run optimization
        results = optimizer.optimize(test_simulation, n_iterations=5)
        print(f"🎯 Best parameters: {results['best_parameters']}") 