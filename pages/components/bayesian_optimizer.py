#!/usr/bin/env python3
"""
Bayesian Optimization for EV Load Curve Optimization
Replaces RL approach with more suitable Bayesian Optimization
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BayesianOptimizer:
    """Bayesian Optimization for EV load curve optimization."""
    
    def __init__(self, simulation_function, parameter_bounds, n_iterations=50, initial_points=15):
        self.simulation_function = simulation_function
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())  # Add this line
        self.n_parameters = len(self.parameter_names)  # Add this line
        self.n_iterations = n_iterations
        self.initial_points = initial_points
        self._iteration_count = 0
        
        # Initialize margin curve for reward calculation
        self._setup_margin_curve()
        
        # Initialize Gaussian Process
        self.gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)),
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
        
        # Storage for observations
        self.X_observed = []
        self.y_observed = []
        self.best_reward = float('-inf')
        self.best_parameters = None
        
        # Parameter importance tracking
        self.parameter_importance = {name: 0.0 for name in self.parameter_names}
    
    def _setup_margin_curve(self):
        """Setup the margin curve for reward calculation."""
        try:
            # Get margin ratio from session state
            margin_ratio = st.session_state.get('available_load_fraction', 0.8)
            
            # Get power values from session state
            power_values = st.session_state.get('power_values', None)
            
            if power_values is None or len(power_values) == 0:
                print("❌ CRITICAL: No power values available for margin curve setup")
                self.margin_curve = np.zeros(2880)
                return
            
            # Convert to numpy array if needed
            power_values = np.array(power_values)
            
            # Check if power_values needs to be multiplied by available_load_fraction (like in main simulation)
            data_source = st.session_state.get('data_source', 'Real Dataset')
            if data_source == "Real Dataset":
                power_values = power_values * margin_ratio
            
            # Extend power_values to match load_curve length (48 hours = 2880 minutes)
            if len(power_values) == 96:  # 24-hour data (96 15-minute intervals)
                # Repeat for 48 hours (96 * 2 = 192, but we need 2880)
                # 2880 / 96 = 30 repetitions needed for 48 hours
                power_values = np.tile(power_values, 30)
            elif len(power_values) == 48:  # 24-hour data (48 hourly intervals)
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
            
            # Ensure power_values matches expected length
            if len(power_values) != 2880:
                print(f"❌ CRITICAL: Power values length {len(power_values)} != 2880 after processing")
                self.margin_curve = np.zeros(2880)
                return
            
            # Use the actual grid profile as the margin curve
            self.margin_curve = np.array(power_values)
            
        except Exception as e:
            print(f"❌ CRITICAL: Error setting up margin curve: {e}")
            self.margin_curve = np.zeros(2880)
    
    def objective_function(self, params: np.ndarray, simulation_func) -> float:
        """
        Objective function to minimize.
        Returns negative reward (we want to maximize reward, so minimize negative).
        """
        # Convert array to dict
        param_dict = {name: params[i] for i, name in enumerate(self.parameter_names)}
        
        try:
            # Run simulation
            load_curve = simulation_func(param_dict)
            
            # Calculate reward (same as RL environment)
            reward = self._calculate_reward(load_curve, param_dict)
            
            return -reward  # Negative because we minimize
            
        except Exception as e:
            print(f"Simulation error: {e}")
            return 1000.0  # High penalty for failed simulations
    
    def _calculate_reward(self, load_curve: np.ndarray, params: Dict[str, float]) -> float:
        """
        Calculate reward based on how well the load curve follows the margin curve.
        Only considers hours 5-48 (minutes 300-2880) to avoid initialization effects.
        """
        try:
            # Ignore first 5 hours (300 minutes) to avoid initialization effects
            start_minute = 300  # 5 hours * 60 minutes
            end_minute = 2880   # 48 hours * 60 minutes
            
            # Extract the relevant portion of the load curve (hours 5-48)
            load_curve_filtered = load_curve[start_minute:end_minute]
            margin_curve_filtered = self.margin_curve[start_minute:end_minute]
            
            # Ensure we have valid data
            if len(load_curve_filtered) == 0 or len(margin_curve_filtered) == 0:
                print(f"❌ CRITICAL: Filtered curves are empty! Load: {len(load_curve_filtered)}, Margin: {len(margin_curve_filtered)}")
                return -1000.0
            
            # Check if filtered load curve is all zeros
            if np.all(load_curve_filtered == 0):
                print(f"❌ CRITICAL: Filtered load curve is all zeros! Simulation may have failed.")
                return -1000.0
            
            # Calculate RMSE between load curve and margin curve (hours 5-48 only)
            rmse = np.sqrt(np.mean((load_curve_filtered - margin_curve_filtered) ** 2))
            
            # Split RMSE into overuse and underuse components
            overuse_mask = load_curve_filtered > margin_curve_filtered
            underuse_mask = load_curve_filtered < margin_curve_filtered
            
            overuse_rmse = np.sqrt(np.mean((load_curve_filtered[overuse_mask] - margin_curve_filtered[overuse_mask]) ** 2)) if np.any(overuse_mask) else 0
            underuse_rmse = np.sqrt(np.mean((margin_curve_filtered[underuse_mask] - load_curve_filtered[underuse_mask]) ** 2)) if np.any(underuse_mask) else 0
            
            # Calculate violation metrics (hours 5-48 only)
            violations = load_curve_filtered > margin_curve_filtered
            violation_frequency = np.sum(violations) / len(load_curve_filtered)
            
            # Calculate max violation duration
            max_violation_duration = 0
            current_violation = 0
            for violation in violations:
                if violation:
                    current_violation += 1
                    max_violation_duration = max(max_violation_duration, current_violation)
                else:
                    current_violation = 0
            
            # Calculate load curve quality metrics (hours 5-48 only)
            load_variance = np.var(load_curve_filtered)
            load_mean = np.mean(load_curve_filtered)
            load_max = np.max(load_curve_filtered)
            load_min = np.min(load_curve_filtered)
            
            # Peak-to-valley ratio
            peak_valley_ratio = load_max / load_min if load_min > 0 else float('inf')
            
            # Ramp rate (rate of change)
            ramp_rate = np.mean(np.abs(np.diff(load_curve_filtered)))
            
            # Utilization (how much of the margin is used on average)
            utilization = np.mean(load_curve_filtered) / np.mean(margin_curve_filtered) if np.mean(margin_curve_filtered) > 0 else 0
            
            # Load smoothness (inverse of variance)
            load_smoothness = 1.0 / (1.0 + load_variance)
            
            # Peak shaving effectiveness (how much we reduce peaks)
            peak_shaving_effectiveness = max(0, (np.max(margin_curve_filtered) - load_max) / np.max(margin_curve_filtered)) if np.max(margin_curve_filtered) > 0 else 0
            
            # Load distribution variance (how evenly distributed the load is)
            load_distribution_variance = np.var(load_curve_filtered / np.max(load_curve_filtered)) if np.max(load_curve_filtered) > 0 else 0
            
            # TOU efficiency (if TOU parameters are being optimized)
            tou_efficiency = 0
            if any('tou_' in key for key in params.keys()):
                # Get actual TOU periods from session state (dynamic)
                time_of_use_timeline = st.session_state.get('time_of_use_timeline', None)
                if time_of_use_timeline and 'periods' in time_of_use_timeline:
                    periods = time_of_use_timeline['periods']
                    
                    # Define off-peak hours (Super Off-Peak + Off-Peak periods)
                    off_peak_hours = []
                    peak_hours = []
                    
                    for period in periods:
                        if period['name'] in ['Super Off-Peak', 'Off-Peak']:
                            off_peak_hours.extend(period['hours'])
                        elif period['name'] == 'Peak':
                            peak_hours.extend(period['hours'])
                    
                    # Calculate average load during off-peak vs peak hours
                    if off_peak_hours and peak_hours:
                        off_peak_load = np.mean([load_curve_filtered[i*60:(i+1)*60] for i in off_peak_hours if i*60 < len(load_curve_filtered)])
                        peak_load = np.mean([load_curve_filtered[i*60:(i+1)*60] for i in peak_hours if i*60 < len(load_curve_filtered)])
                        
                        if peak_load > 0:
                            tou_efficiency = max(0, (off_peak_load - peak_load) / peak_load)
                else:
                    # Fallback to hardcoded periods if timeline not available
                    off_peak_hours = list(range(1, 2)) + list(range(2, 6)) + list(range(6, 8)) + list(range(22, 25))
                    peak_hours = list(range(9, 11)) + list(range(18, 21))
                    
                    off_peak_load = np.mean([load_curve_filtered[i*60:(i+1)*60] for i in off_peak_hours if i*60 < len(load_curve_filtered)])
                    peak_load = np.mean([load_curve_filtered[i*60:(i+1)*60] for i in peak_hours if i*60 < len(load_curve_filtered)])
                    
                    if peak_load > 0:
                        tou_efficiency = max(0, (off_peak_load - peak_load) / peak_load)
            
            # Combine metrics into reward (weighted sum)
            # Primary: Grid compliance (most important)
            rmse_weight = -8.0          # Heavy penalty for overall RMSE (critical)
            overuse_weight = -6.4       # Heavy penalty for overuse (critical)
            underuse_weight = -5.0      # Light penalty for underuse (less critical)
            violation_freq_weight = -2.0  # Penalty for frequent violations
            violation_duration_weight = -1.0  # Penalty for long violations
            
            # Secondary: Load quality (moderate importance)
            variance_weight = -0.2       # Light penalty for high variance
            peak_valley_weight = -0.33    # Light penalty for high peak-valley ratio
            ramp_weight = -0.05          # Very light penalty for high ramp rate
            smoothness_weight = -0.1     # Light penalty for lack of smoothness
            
            # Tertiary: Efficiency and utilization (bonuses)
            utilization_weight = 2.0     # Good bonus for utilization (50-90%)
            peak_shaving_weight = 1.5    # Bonus for effective peak shaving
            distribution_weight = -0.05  # Light penalty for uneven distribution
            tou_efficiency_weight = 1.0  # Bonus for TOU efficiency
            
            # Car count bonus (encourage using more cars)
            car_count_bonus_weight = 1.8  # Good bonus for using more cars
            car_count = params.get('car_count', 100)
            car_count_bonus = min(car_count / 200.0, 1.0)  # Normalize to 0-1, max at 200 cars
            
            
            
            # TOU diversity penalty (encourage balanced percentages)
            tou_diversity_penalty_weight = -0.5  # Penalty for very diverse TOU percentages
            tou_percentages = []
            if any('tou_' in key for key in params.keys()):
                tou_percentages = [
                    params.get('tou_super_offpeak_adoption', 25),
                    params.get('tou_offpeak_adoption', 25),
                    params.get('tou_midpeak_adoption', 25),
                    params.get('tou_peak_adoption', 25)
                ]
                # Calculate diversity as standard deviation of percentages
                if len(tou_percentages) == 4:
                    tou_mean = np.mean(tou_percentages)
                    tou_std = np.std(tou_percentages)
                    tou_diversity_penalty = min(tou_std / 25.0, 1.0)  # Normalize to 0-1, max penalty at 25% std
                else:
                    tou_diversity_penalty = 0.0
            else:
                tou_diversity_penalty = 0.0
            
            # Normalize metrics to similar scales (0-1 range)
            max_margin = np.max(margin_curve_filtered)
            if max_margin > 0:
                normalized_variance = min(load_variance / (max_margin ** 2), 1.0)
                normalized_peak_valley = min(peak_valley_ratio / 10.0, 1.0)
                normalized_ramp = min(ramp_rate / max_margin, 1.0)
                normalized_smoothness = min(load_smoothness / max_margin, 1.0)
            else:
                normalized_variance = normalized_peak_valley = normalized_ramp = normalized_smoothness = 0.0
            
            normalized_distribution = min(load_distribution_variance, 1.0)
            
            # Utilization bonus (optimal range: 50-90%)
            utilization_bonus = 0
            if 0.5 <= utilization <= 0.9:
                utilization_bonus = 1.0  # Full bonus for optimal utilization
            elif utilization < 0.5:
                utilization_bonus = utilization / 0.5  # Partial bonus for under-utilization
            else:
                utilization_bonus = max(0, (1.0 - utilization) / 0.1)  # Penalty for over-utilization
            
            # Additional bonus for high utilization (70-90%)
            if 0.7 <= utilization <= 0.9:
                utilization_bonus += 0.5  # Extra bonus for high utilization
            
            # Calculate final reward with logical scaling
            reward = (rmse_weight * min(rmse / max_margin, 1.0) +
                     overuse_weight * min(overuse_rmse / max_margin, 1.0) + 
                     underuse_weight * min(underuse_rmse / max_margin, 1.0) +
                     violation_freq_weight * violation_frequency +
                     violation_duration_weight * min(max_violation_duration / len(load_curve_filtered), 1.0) +
                     variance_weight * normalized_variance +
                     peak_valley_weight * normalized_peak_valley +
                     ramp_weight * normalized_ramp +
                     smoothness_weight * normalized_smoothness +
                     utilization_weight * utilization_bonus +
                     peak_shaving_weight * peak_shaving_effectiveness +
                     distribution_weight * normalized_distribution +
                     tou_efficiency_weight * min(tou_efficiency, 1.0) +
                     car_count_bonus_weight * car_count_bonus +
                     tou_diversity_penalty_weight * tou_diversity_penalty) # Add TOU diversity penalty
            
            # Add debug info for first few iterations
            if hasattr(self, '_iteration_count'):
                self._iteration_count += 1
            else:
                self._iteration_count = 1
            
            # Add warning for -1000 rewards
            if reward <= -1000:
                print(f"❌ CRITICAL: Reward {reward:.2f} in iteration {self._iteration_count}")
                print(f"  Load curve stats: min={np.min(load_curve_filtered):.2f}, max={np.max(load_curve_filtered):.2f}, mean={np.mean(load_curve_filtered):.2f}")
                print(f"  Margin curve stats: min={np.min(margin_curve_filtered):.2f}, max={np.max(margin_curve_filtered):.2f}, mean={np.mean(margin_curve_filtered):.2f}")
                print(f"  Power values length: {len(power_values) if 'power_values' in locals() else 'N/A'}")
            
            # Special handling for zero load curve
            if np.all(load_curve_filtered == 0):
                print(f"❌ CRITICAL: Load curve is all zeros! Simulation may have failed.")
                print(f"  This will cause reward to be -1000. Check simulation setup.")
                return -1000.0
            
            return reward
        except Exception as e:
            print(f"❌ Reward calculation error: {e}")
            return -1000.0
    
    def optimize(self, progress_callback=None):
        """
        Run Bayesian optimization.
        
        Args:
            progress_callback: Optional callback function(iteration, total_iterations, best_reward)
            
        Returns:
            Dict with best parameters and reward
        """
        # Initial random sampling
        print(f"🚀 Starting Bayesian optimization ({self.n_iterations} iterations, {self.initial_points} initial points)")
        
        for i in range(self.initial_points):
            # Generate diverse initial parameters
            params = {}
            for param_name, (min_val, max_val) in self.parameter_bounds.items():
                # Use Latin Hypercube sampling for better coverage
                if i == 0:
                    # First point: use middle values
                    params[param_name] = (min_val + max_val) / 2
                elif i == 1:
                    # Second point: use extreme values
                    params[param_name] = min_val if i % 2 == 0 else max_val
                else:
                    # Other points: use stratified sampling
                    stratum = (i - 2) % 4  # 4 strata
                    stratum_size = (max_val - min_val) / 4
                    params[param_name] = min_val + stratum * stratum_size + np.random.uniform(0, stratum_size)
            
            # Evaluate this point
            reward = self._evaluate_point(params)
            
            # Store results
            param_values = [params[name] for name in self.parameter_names]
            self.X_observed.append(param_values)
            self.y_observed.append(reward)
            
            # Update best if better
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_parameters = params.copy()
                print(f"🎯 New best reward: {reward:.2f} with params: {params}")
            
            if progress_callback:
                progress_callback(i, self.n_iterations + self.initial_points, self.best_reward)
        
        # Fit GP to initial observations
        if len(self.X_observed) > 0:
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
        
        # Bayesian optimization loop
        print(f"🧠 Starting Bayesian optimization loop...")
        for iteration in range(self.n_iterations):
            # Find next point to evaluate
            next_params = self._next_point()
            
            # Evaluate this point
            reward = self._evaluate_point(next_params)
            
            # Store results
            param_values = [next_params[name] for name in self.parameter_names]
            self.X_observed.append(param_values)
            self.y_observed.append(reward)
            
            # Update best if better
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_parameters = next_params.copy()
                print(f"🎯 New best reward: {reward:.2f} with params: {next_params}")
            
            # Retrain GP
            if len(self.X_observed) > 0:
                self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
            
            if progress_callback:
                progress_callback(self.initial_points + iteration, self.n_iterations + self.initial_points, self.best_reward)
        
        # Calculate parameter importance
        parameter_importance = self._calculate_parameter_importance()
        
        print(f"✅ Optimization completed! Best reward: {self.best_reward:.2f}")
        
        return {
            'best_parameters': self.best_parameters,
            'best_reward': self.best_reward,
            'parameter_importance': parameter_importance,
            'n_iterations': self.n_iterations,
            'n_initial_points': self.initial_points
        }
    
    def _evaluate_point(self, params):
        """
        Evaluate a point by running simulation and calculating reward.
        
        Args:
            params: Dict of parameter values
            
        Returns:
            Reward value (higher is better)
        """
        try:
            # Run simulation with these parameters
            load_curve = self.simulation_function(params)
            
            # Calculate reward
            reward = self._calculate_reward(load_curve, params)
            
            return reward
            
        except Exception as e:
            print(f"❌ Error evaluating point: {e}")
            return -1000.0
    
    def _next_point(self):
        """Find next point to evaluate using acquisition function."""
        # Use Expected Improvement acquisition function
        def acquisition_function(x):
            if len(self.X_observed) == 0:
                return 0.0
            
            x = x.reshape(1, -1)
            mean, std = self.gp.predict(x, return_std=True)
            
            # Expected Improvement with exploration bonus
            best_f = max(self.y_observed)
            ei = (mean - best_f) / (std + 1e-8)
            ei = (mean - best_f) * norm.cdf(ei) + std * norm.pdf(ei)
            
            # Add exploration bonus to encourage trying new parameter combinations
            exploration_bonus = std * 0.1  # Small bonus for high uncertainty
            
            return -(ei + exploration_bonus)  # Negative because we minimize
        
        # Optimize acquisition function
        bounds = [self.parameter_bounds[name] for name in self.parameter_bounds.keys()]
        
        # Use best parameters found so far as starting point
        if len(self.X_observed) > 0:
            best_idx = np.argmax(self.y_observed)
            x0 = self.X_observed[best_idx]
        else:
            # Generate random starting point
            x0 = []
            for param_name, (min_val, max_val) in self.parameter_bounds.items():
                x0.append(np.random.uniform(min_val, max_val))
        
        result = minimize(
            acquisition_function,
            x0=np.array(x0),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        
        # Convert result back to parameter dict
        param_names = list(self.parameter_bounds.keys())
        next_params = {}
        for i, param_name in enumerate(param_names):
            next_params[param_name] = result.x[i]
        
        return next_params
    
    def _calculate_parameter_importance(self):
        """Calculate parameter importance based on GP kernel."""
        if len(self.X_observed) < 2:
            return {name: 1.0 for name in self.parameter_names}
        
        try:
            # Get kernel parameters
            kernel_params = self.gp.kernel_.get_params()
            length_scales = kernel_params.get('k2__length_scale', np.ones(self.n_parameters))
            
            # Importance is inverse of length scale (shorter = more important)
            for i, name in enumerate(self.parameter_names):
                self.parameter_importance[name] = 1.0 / (length_scales[i] + 1e-8)
            
            return self.parameter_importance
        except Exception as e:
            print(f"Warning: Could not calculate parameter importance: {e}")
            return {name: 1.0 for name in self.parameter_names}

def create_bayesian_optimizer() -> BayesianOptimizer:
    """Create a Bayesian optimizer with parameter bounds."""
    
    # Parameter bounds (same as UI)
    parameter_bounds = {
        'tou_super_offpeak_adoption': (15, 40),   # 10-40% instead of 0-25%
        'tou_offpeak_adoption': (15, 40),        # 10-40% instead of 0-25%
        'tou_midpeak_adoption': (30, 40),         # 30-40% instead of 0-25%
        'tou_peak_adoption': (15, 40),           # 10-40% instead of 0-25%
        'car_count': (1, 200)
    }
    
    return BayesianOptimizer(parameter_bounds)

if __name__ == "__main__":
    # Test the Bayesian Optimizer
    optimizer = create_bayesian_optimizer()
    print(f"✅ Bayesian Optimizer created with {optimizer.n_parameters} parameters")
    
    # Example simulation function (placeholder)
    def test_simulation(params):
        # Placeholder - would be replaced with actual simulation
        return np.random.normal(100, 20, 2880)  # 48-hour load curve
    
    # Run optimization
    results = optimizer.optimize(test_simulation, n_iterations=5, initial_points=3)
    print(f"🎯 Best parameters: {results['best_parameters']}") 