# -*- coding: utf-8 -*-
"""
Portable Load Curve Generator
Standalone module for generating synthetic load curves.
Easy to integrate into other projects/apps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import pickle


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for load curve generation.
    Enhanced with diversity-promoting components.
    """
    
    def __init__(self, 
                 seq_len=192, 
                 hidden_dim=256, 
                 latent_dim=64, 
                 cond_dim=6):
        super().__init__()
        
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Condition processing
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(seq_len + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder with diversity components
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len)
        )
        
        # Diversity-promoting components
        # 1. Multiple pattern generators
        self.pattern_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, seq_len // 4)
            ) for _ in range(4)
        ])
        
        # 2. Local variation generator
        self.local_variation = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, seq_len),
            nn.Tanh()
        )
        
        # 3. Peak modulation network
        self.peak_modulation = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, seq_len),
            nn.Sigmoid()
        )
        
        # Strong condition injection
        self.cond_injection = nn.Sequential(
            nn.Linear(hidden_dim, seq_len),
            nn.Tanh()
        )
        
        self.cond_weight = nn.Parameter(torch.tensor(5.0))
        
        # Diversity weights
        self.pattern_weight = nn.Parameter(torch.tensor(0.3))
        self.local_weight = nn.Parameter(torch.tensor(0.1))
        self.peak_weight = nn.Parameter(torch.tensor(0.2))
        
    def encode(self, x, condition):
        """Encode input to latent space."""
        cond_feat = self.cond_proj(condition)
        combined = torch.cat([x, cond_feat], dim=1)
        hidden = self.encoder(combined)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, condition):
        """Decode from latent space with diversity components."""
        cond_feat = self.cond_proj(condition)
        combined = torch.cat([z, cond_feat], dim=1)
        
        # Base reconstruction
        base_output = self.decoder(combined)
        
        # 1. Add pattern variations
        patterns = []
        for pattern_gen in self.pattern_generators:
            pattern = pattern_gen(z)
            # Upsample pattern to full sequence length
            pattern = F.interpolate(pattern.unsqueeze(1), size=self.seq_len, mode='linear', align_corners=False).squeeze(1)
            patterns.append(pattern)
        
        pattern_sum = torch.stack(patterns, dim=1).mean(dim=1)
        
        # 2. Add local variations
        local_var = self.local_variation(z)
        
        # 3. Add peak modulation
        peak_mod = self.peak_modulation(z)
        
        # Combine all components
        output = (base_output + 
                 self.pattern_weight * pattern_sum +
                 self.local_weight * local_var +
                 self.peak_weight * peak_mod * base_output)
        
        # Strong condition injection
        cond_injection = self.cond_injection(cond_feat)
        output = output + self.cond_weight * cond_injection
        
        return output
    
    def forward(self, x, condition):
        """Forward pass."""
        mu, log_var = self.encode(x, condition)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z, condition)
        return output, mu, log_var
    
    def generate(self, condition, num_samples=1, diversity_mode='normal'):
        """
        Generate samples from condition with natural diversity.
        The diversity comes from the model's learned representations.
        
        Args:
            condition: Input condition tensor
            num_samples: Number of samples to generate
            diversity_mode: 'normal', 'high', 'extreme' for different diversity levels
        """
        batch_size = condition.shape[0]
        
        # Base latent sampling with controlled diversity
        if diversity_mode == 'normal':
            z = torch.randn(batch_size * num_samples, self.latent_dim).to(condition.device)
        elif diversity_mode == 'high':
            # Increase latent space exploration
            z = torch.randn(batch_size * num_samples, self.latent_dim).to(condition.device) * 1.2
        elif diversity_mode == 'extreme':
            # Maximum latent space exploration
            z = torch.randn(batch_size * num_samples, self.latent_dim).to(condition.device) * 1.5
        
        condition_expanded = condition.repeat(num_samples, 1)
        output = self.decode(z, condition_expanded)
        
        return output


def encode_condition(season, is_weekend, max_power, max_power_normalizer=630):
    """
    Encode condition parameters into a tensor.
    
    Args:
        season (str): 'winter', 'spring', 'summer', 'autumn'
        is_weekend (bool): True for weekend, False for weekday
        max_power (float): Maximum power in kW
        max_power_normalizer (float): Normalization factor for max_power
    
    Returns:
        torch.Tensor: Encoded condition vector
    """
    # Season encoding (one-hot)
    season_map = {'winter': [1, 0, 0, 0], 'spring': [0, 1, 0, 0], 
                  'summer': [0, 0, 1, 0], 'autumn': [0, 0, 0, 1]}
    
    season_encoding = season_map.get(season.lower(), [1, 0, 0, 0])
    
    # Weekend encoding
    weekend_encoding = [1] if is_weekend else [0]
    
    # Normalized max power
    max_power_norm = max_power / max_power_normalizer
    
    # Combine all features
    condition = season_encoding + weekend_encoding + [max_power_norm]
    
    return torch.tensor(condition, dtype=torch.float32)


class LoadCurveGenerator:
    """
    Main class for generating load curves.
    Handles model loading and generation.
    """
    
    def __init__(self, models_dir="portable_models"):
        """
        Initialize the generator.
        
        Args:
            models_dir (str): Directory containing trained models
        """
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models (will be loaded when needed)
        self.weekday_model = None
        self.weekend_model = None
        self.weekday_scaler = None
        self.weekend_scaler = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load trained models and scalers."""
        try:
            # Load weekday model
            self.weekday_model = ConditionalVAE().to(self.device)
            weekday_path = os.path.join(self.models_dir, "weekday", "trained_weekday_model.pth")
            self.weekday_model.load_state_dict(torch.load(weekday_path, map_location=self.device))
            self.weekday_model.eval()
            
            # Load weekend model
            self.weekend_model = ConditionalVAE().to(self.device)
            weekend_path = os.path.join(self.models_dir, "weekend", "trained_weekend_model.pth")
            self.weekend_model.load_state_dict(torch.load(weekend_path, map_location=self.device))
            self.weekend_model.eval()
            
            # Load scalers
            weekday_scaler_path = os.path.join(self.models_dir, "weekday", "scaler.pth")
            weekend_scaler_path = os.path.join(self.models_dir, "weekend", "scaler.pth")
            
            self.weekday_scaler = torch.load(weekday_scaler_path, map_location='cpu')
            self.weekend_scaler = torch.load(weekend_scaler_path, map_location='cpu')
            
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please ensure the models directory contains the trained models.")
            raise
    
    def generate(self, season, day_type, max_power, diversity_mode='high', 
                 return_timestamps=True, start_time=None):
        """
        Generate a load curve.
        
        Args:
            season (str): 'winter', 'spring', 'summer', 'autumn'
            day_type (str): 'weekday' or 'weekend'
            max_power (float): Maximum power in kW (100-1000)
            diversity_mode (str): 'normal', 'high', 'extreme'
            return_timestamps (bool): Whether to return timestamps
            start_time (datetime): Start time for timestamps (default: current time)
        
        Returns:
            dict: Contains 'load_curve', 'timestamps' (if requested), 'metadata'
        """
        # Validate inputs
        if season not in ['winter', 'spring', 'summer', 'autumn']:
            raise ValueError("Season must be 'winter', 'spring', 'summer', or 'autumn'")
        
        if day_type not in ['weekday', 'weekend']:
            raise ValueError("Day type must be 'weekday' or 'weekend'")
        
        if not (100 <= max_power <= 1000):
            raise ValueError("Max power must be between 100 and 1000 kW")
        
        if diversity_mode not in ['normal', 'high', 'extreme']:
            diversity_mode = 'high'
        
        # Generate condition
        is_weekend = (day_type == 'weekend')
        condition = encode_condition(season, is_weekend, max_power)
        
        # Select model and scaler
        if is_weekend:
            model = self.weekend_model
            scaler = self.weekend_scaler
        else:
            model = self.weekday_model
            scaler = self.weekday_scaler
        
        # Generate load curve
        with torch.no_grad():
            condition = condition.to(self.device).unsqueeze(0)
            generated = model.generate(condition, num_samples=1, diversity_mode=diversity_mode)
            load_curve = scaler.inverse_transform(generated.cpu().numpy()).flatten()
        
        # Create timestamps if requested
        timestamps = None
        if return_timestamps:
            if start_time is None:
                start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            timestamps = [start_time + timedelta(minutes=15*i) for i in range(192)]
        
        # Metadata
        metadata = {
            'season': season,
            'day_type': day_type,
            'max_power': max_power,
            'diversity_mode': diversity_mode,
            'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_points': len(load_curve),
            'time_interval_minutes': 15
        }
        
        result = {
            'load_curve': load_curve,
            'metadata': metadata
        }
        
        if timestamps is not None:
            result['timestamps'] = timestamps
        
        return result
    
    def generate_batch(self, conditions, diversity_mode='high'):
        """
        Generate multiple load curves.
        
        Args:
            conditions (list): List of dicts with 'season', 'day_type', 'max_power'
            diversity_mode (str): 'normal', 'high', 'extreme'
        
        Returns:
            list: List of generation results
        """
        results = []
        for condition in conditions:
            result = self.generate(
                season=condition['season'],
                day_type=condition['day_type'],
                max_power=condition['max_power'],
                diversity_mode=diversity_mode
            )
            results.append(result)
        return results


# Convenience function for quick generation
def generate_load_curve(season, day_type, max_power, diversity_mode='high', 
                       models_dir="portable_models", return_timestamps=True):
    """
    Quick function to generate a single load curve.
    
    Args:
        season (str): 'winter', 'spring', 'summer', 'autumn'
        day_type (str): 'weekday' or 'weekend'
        max_power (float): Maximum power in kW (100-1000)
        diversity_mode (str): 'normal', 'high', 'extreme'
        models_dir (str): Directory containing trained models
        return_timestamps (bool): Whether to return timestamps
    
    Returns:
        dict: Generation result
    """
    generator = LoadCurveGenerator(models_dir)
    return generator.generate(season, day_type, max_power, diversity_mode, return_timestamps)


# Example usage
if __name__ == "__main__":
    # Example 1: Quick generation
    result = generate_load_curve('winter', 'weekday', 300)
    print(f"Generated load curve with mean: {np.mean(result['load_curve']):.2f} kW")
    
    # Example 2: Using the generator class
    generator = LoadCurveGenerator()
    
    # Single generation
    result = generator.generate('summer', 'weekend', 500)
    print(f"Generated load curve with mean: {np.mean(result['load_curve']):.2f} kW")
    
    # Batch generation
    conditions = [
        {'season': 'winter', 'day_type': 'weekday', 'max_power': 300},
        {'season': 'summer', 'day_type': 'weekend', 'max_power': 500},
        {'season': 'spring', 'day_type': 'weekday', 'max_power': 400}
    ]
    
    results = generator.generate_batch(conditions)
    for i, result in enumerate(results):
        print(f"Curve {i+1}: {result['metadata']['season']} {result['metadata']['day_type']} "
              f"mean={np.mean(result['load_curve']):.2f} kW") 