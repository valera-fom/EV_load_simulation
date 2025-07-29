#!/usr/bin/env python3
"""
Time of Use (TOU) Optimizer
A simplified optimizer that creates optimal TOU periods for 24 hours only.
The periods are then duplicated for the second day in the simulation.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def optimize_tou_periods_24h(margin_curve: List[float]) -> Dict:
    """
    Create optimal TOU periods based on capacity levels for exactly 24 hours.
    
    Args:
        margin_curve: List of capacity values over time (1-minute steps)
    
    Returns:
        Dictionary with optimal TOU timeline structure for 24 hours
    """
    # Extract first 24 hours (1440 minutes)
    margin_curve_24h = margin_curve[:24*60]
    
    # Calculate hourly average capacity for 24 hours
    hourly_capacity = []
    for hour in range(24):
        start_idx = hour * 60  # 60 steps per hour
        end_idx = start_idx + 60
        hour_avg = np.mean(margin_curve_24h[start_idx:end_idx])
        hourly_capacity.append(hour_avg)
    
    # Find capacity range
    min_capacity = min(hourly_capacity)
    max_capacity = max(hourly_capacity)
    capacity_range = max_capacity - min_capacity
    
    # Create 4 equal capacity buckets
    bucket_size = capacity_range / 4
    capacity_thresholds = [
        min_capacity + bucket_size,
        min_capacity + 2 * bucket_size,
        min_capacity + 3 * bucket_size,
        max_capacity
    ]
    
    # Assign each hour to a capacity bucket (0-3)
    period_assignments = []
    for hour in range(24):
        capacity = hourly_capacity[hour]
        if capacity <= capacity_thresholds[0]:
            period_assignments.append(3)  # Peak (lowest capacity)
        elif capacity <= capacity_thresholds[1]:
            period_assignments.append(2)  # Mid-Peak
        elif capacity <= capacity_thresholds[2]:
            period_assignments.append(1)  # Off-Peak
        else:
            period_assignments.append(0)  # Super Off-Peak (highest capacity)
    
    # Create periods with consecutive hour ranges
    period_names = ['Super Off-Peak', 'Off-Peak', 'Mid-Peak', 'Peak']
    period_colors = ['#87CEEB', '#90EE90', '#FFD700', '#FF6B6B']
    
    periods = []
    for period_idx, (name, color) in enumerate(zip(period_names, period_colors)):
        # Find all hours assigned to this period
        period_hours = [hour + 1 for hour, assignment in enumerate(period_assignments) if assignment == period_idx]
        
        if period_hours:
            # Group consecutive hours into ranges
            ranges = _group_consecutive_hours(period_hours)
            
            # Create one period per range, but track total hours for adoption calculation
            total_hours_for_period = len(period_hours)
            adoption_per_period = 25  # Default equal distribution
            
            for start_hour, end_hour in ranges:
                periods.append({
                    'name': name,
                    'color': color,
                    'adoption': adoption_per_period,  # Will be adjusted later
                    'hours': list(range(start_hour, end_hour)),
                    'total_hours': total_hours_for_period  # Track total hours for this TOU type
                })
        else:
            # If no hours assigned, create a default period
            periods.append({
                'name': name,
                'color': color,
                'adoption': 25,
                'hours': [1, 2, 3, 4, 5, 6],  # Default 6 hours
                'total_hours': 6
            })
    
    # Merge periods of the same type and adjust adoption percentages
    merged_periods = []
    seen_names = set()
    
    for period in periods:
        if period['name'] not in seen_names:
            # First occurrence of this TOU type
            merged_periods.append(period)
            seen_names.add(period['name'])
        else:
            # Merge with existing period of same name
            existing_period = next(p for p in merged_periods if p['name'] == period['name'])
            # Combine hours
            existing_period['hours'].extend(period['hours'])
            existing_period['hours'] = sorted(list(set(existing_period['hours'])))  # Remove duplicates
            # Average the adoption percentages
            existing_period['adoption'] = (existing_period['adoption'] + period['adoption']) / 2
    
    # Ensure total adoption is 100%
    total_adoption = sum(p['adoption'] for p in merged_periods)
    if total_adoption != 100:
        # Normalize adoption percentages
        for period in merged_periods:
            period['adoption'] = (period['adoption'] / total_adoption) * 100
    
    # Create visualization
    fig = _create_optimization_visualization(margin_curve_24h, hourly_capacity, period_assignments, 
                                           period_names, period_colors, capacity_thresholds)
    
    return {
        'periods': merged_periods,
        'selected_period': 0,
        'visualization': fig,
        'capacity_thresholds': capacity_thresholds,
        'hourly_capacity': hourly_capacity
    }

def _group_consecutive_hours(hours: List[int]) -> List[Tuple[int, int]]:
    """
    Group consecutive hours into ranges.
    
    Args:
        hours: List of hour numbers (1-24)
    
    Returns:
        List of (start_hour, end_hour) tuples
    """
    if not hours:
        return []
    
    ranges = []
    start = hours[0]
    end = start
    
    for i in range(1, len(hours)):
        if hours[i] == hours[i-1] + 1:
            end = hours[i]
        else:
            ranges.append((start, end + 1))  # +1 because end is exclusive
            start = hours[i]
            end = start
    
    ranges.append((start, end + 1))
    return ranges

def _create_optimization_visualization(margin_curve_24h: List[float], hourly_capacity: List[float], 
                                     period_assignments: List[int], period_names: List[str], 
                                     period_colors: List[str], capacity_thresholds: List[float]) -> plt.Figure:
    """
    Create visualization of the optimization results.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Plot margin curve (15-minute resolution for clarity)
    margin_curve_15min = margin_curve_24h[::15]  # Every 15 minutes
    time_15min = np.arange(len(margin_curve_15min)) * 15 / 60  # Convert to hours
    
    ax.plot(time_15min, margin_curve_15min, 'b-', linewidth=4, label='Available Capacity')
    
    # Add capacity threshold lines
    for i, threshold in enumerate(capacity_thresholds):
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.7, linewidth=2,
                  label=f'Threshold {i+1}: {threshold:.1f} kW')
    
    # Color background based on period assignments
    for hour, period_idx in enumerate(period_assignments):
        color = period_colors[period_idx]
        start_time = hour
        end_time = hour + 1
        ax.axvspan(start_time, end_time, alpha=0.3, color=color)
    
    ax.set_xlabel('Hour of Day', fontsize=24)
    ax.set_ylabel('Available Capacity (kW)', fontsize=24)
    ax.set_title('Optimal TOU Periods (24 Hours)', fontsize=28)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16)
    
    # Add period info
    period_info = []
    for i, name in enumerate(period_names):
        hours_count = period_assignments.count(i)
        period_info.append(f'{name}: {hours_count}h')
    
    ax.text(0.02, 0.98, f"Periods: {' | '.join(period_info)}", 
            transform=ax.transAxes, fontsize=16, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def convert_to_simulation_format(periods: List[Dict]) -> List[Dict]:
    """
    Convert TOU periods to simulation format (start/end hours).
    
    Args:
        periods: List of periods with 'hours' lists
    
    Returns:
        List of periods with 'start' and 'end' properties
    """
    simulation_periods = []
    
    for period in periods:
        if 'hours' in period and period['hours']:
            hours = sorted(period['hours'])
            # Filter to 1-24 range only
            day_hours = [h for h in hours if 1 <= h <= 24]
            
            if day_hours:
                min_hour = min(day_hours)
                max_hour = max(day_hours)
                
                simulation_periods.append({
                    'name': period['name'],
                    'start': min_hour,
                    'end': max_hour + 1,  # +1 because end is exclusive
                    'adoption': period['adoption']
                })
    
    # Ensure exactly 4 periods by merging duplicates
    unique_periods = []
    seen_names = set()
    
    for period in simulation_periods:
        if period['name'] not in seen_names:
            unique_periods.append(period)
            seen_names.add(period['name'])
        else:
            # Merge with existing period
            existing = next(p for p in unique_periods if p['name'] == period['name'])
            existing['start'] = min(existing['start'], period['start'])
            existing['end'] = max(existing['end'], period['end'])
            existing['adoption'] = (existing['adoption'] + period['adoption']) / 2
    
    return unique_periods

def validate_periods(periods: List[Dict]) -> bool:
    """
    Validate that periods are correct for simulation.
    
    Args:
        periods: List of periods to validate
    
    Returns:
        True if valid, False otherwise
    """
    if len(periods) < 4:
        return False
    
    expected_names = {'Super Off-Peak', 'Off-Peak', 'Mid-Peak', 'Peak'}
    actual_names = {p['name'] for p in periods}
    
    if actual_names != expected_names:
        return False
    
    # Check that periods cover 1-24 hours
    total_hours = 0
    for period in periods:
        hours = period['end'] - period['start']
        total_hours += hours
    
    if total_hours != 24:
        return False
    
    return True