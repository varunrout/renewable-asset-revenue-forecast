"""
Monte Carlo Simulation Module

Functions for simulating revenue distributions under uncertainty.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Monte Carlo simulation for revenue uncertainty."""
    
    def __init__(self, n_simulations: int = 1000, random_seed: int = 42):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of simulations to run
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def simulate_price_scenarios(
        self,
        base_prices: pd.Series,
        volatility: float = 0.2,
        drift: float = 0.0,
        method: str = 'multiplicative'
    ) -> pd.DataFrame:
        """
        Simulate price scenarios.
        
        Args:
            base_prices: Base price time series
            volatility: Price volatility (std dev as fraction of price)
            drift: Mean drift (as fraction per period)
            method: Simulation method ('multiplicative', 'additive', 'percentile')
            
        Returns:
            DataFrame with simulated price scenarios
        """
        logger.info(f"Simulating {self.n_simulations} price scenarios")
        
        scenarios = pd.DataFrame(index=base_prices.index)
        
        if method == 'multiplicative':
            # Multiplicative noise: price * (1 + noise)
            for i in range(self.n_simulations):
                noise = np.random.normal(drift, volatility, len(base_prices))
                scenarios[f'scenario_{i}'] = base_prices * (1 + noise)
        
        elif method == 'additive':
            # Additive noise: price + noise
            price_std = base_prices.std()
            for i in range(self.n_simulations):
                noise = np.random.normal(0, volatility * price_std, len(base_prices))
                scenarios[f'scenario_{i}'] = base_prices + noise
        
        elif method == 'percentile':
            # Bootstrap from historical distribution
            for i in range(self.n_simulations):
                # Sample from historical price distribution
                sampled = np.random.choice(base_prices, size=len(base_prices), replace=True)
                scenarios[f'scenario_{i}'] = sampled
        
        # Ensure no negative prices
        scenarios = scenarios.clip(lower=0)
        
        return scenarios
    
    def simulate_generation_scenarios(
        self,
        base_generation: pd.Series,
        cf_uncertainty: float = 0.1,
        method: str = 'multiplicative'
    ) -> pd.DataFrame:
        """
        Simulate generation scenarios.
        
        Args:
            base_generation: Base generation time series
            cf_uncertainty: Capacity factor uncertainty (as fraction)
            method: Simulation method ('multiplicative', 'seasonal')
            
        Returns:
            DataFrame with simulated generation scenarios
        """
        logger.info(f"Simulating {self.n_simulations} generation scenarios")
        
        scenarios = pd.DataFrame(index=base_generation.index)
        
        if method == 'multiplicative':
            # Apply random scaling factors
            for i in range(self.n_simulations):
                # Sample a CF adjustment factor
                cf_factor = np.random.normal(1.0, cf_uncertainty)
                scenarios[f'scenario_{i}'] = base_generation * cf_factor
        
        elif method == 'seasonal':
            # Apply seasonal variation
            for i in range(self.n_simulations):
                # Random walk for each season
                cf_factor = 1.0 + np.random.normal(0, cf_uncertainty)
                scenarios[f'scenario_{i}'] = base_generation * cf_factor
        
        # Ensure non-negative generation
        scenarios = scenarios.clip(lower=0)
        
        return scenarios
    
    def simulate_revenue_distribution(
        self,
        base_prices: pd.Series,
        base_generation: pd.Series,
        price_volatility: float = 0.2,
        generation_uncertainty: float = 0.1,
        simulate_price: bool = True,
        simulate_generation: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Simulate revenue distribution.
        
        Args:
            base_prices: Base price time series
            base_generation: Base generation time series
            price_volatility: Price volatility parameter
            generation_uncertainty: Generation uncertainty parameter
            simulate_price: Whether to simulate price scenarios
            simulate_generation: Whether to simulate generation scenarios
            
        Returns:
            Dictionary with revenue statistics
        """
        logger.info("Simulating revenue distribution")
        
        revenues = []
        
        for i in range(self.n_simulations):
            # Generate price scenario
            if simulate_price:
                noise = np.random.normal(0, price_volatility, len(base_prices))
                prices = base_prices * (1 + noise)
                prices = prices.clip(lower=0)
            else:
                prices = base_prices
            
            # Generate generation scenario
            if simulate_generation:
                cf_factor = np.random.normal(1.0, generation_uncertainty)
                generation = base_generation * cf_factor
                generation = generation.clip(lower=0)
            else:
                generation = base_generation
            
            # Calculate revenue
            revenue = (prices * generation).sum()
            revenues.append(revenue)
        
        revenues = np.array(revenues)
        
        # Calculate statistics
        results = {
            'revenues': revenues,
            'mean': revenues.mean(),
            'std': revenues.std(),
            'p10': np.percentile(revenues, 10),
            'p50': np.percentile(revenues, 50),
            'p90': np.percentile(revenues, 90),
            'min': revenues.min(),
            'max': revenues.max()
        }
        
        logger.info(f"Revenue distribution:")
        logger.info(f"  Mean: £{results['mean']:,.0f}")
        logger.info(f"  P10:  £{results['p10']:,.0f}")
        logger.info(f"  P50:  £{results['p50']:,.0f}")
        logger.info(f"  P90:  £{results['p90']:,.0f}")
        logger.info(f"  Std:  £{results['std']:,.0f}")
        
        return results
    
    def value_at_risk(
        self,
        revenues: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            revenues: Array of revenue simulations
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR value
        """
        var = np.percentile(revenues, (1 - confidence_level) * 100)
        logger.info(f"VaR at {confidence_level*100}% confidence: £{var:,.0f}")
        return var
    
    def conditional_value_at_risk(
        self,
        revenues: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Also known as Expected Shortfall.
        
        Args:
            revenues: Array of revenue simulations
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            CVaR value
        """
        var = self.value_at_risk(revenues, confidence_level)
        cvar = revenues[revenues <= var].mean()
        logger.info(f"CVaR at {confidence_level*100}% confidence: £{cvar:,.0f}")
        return cvar


def monte_carlo_revenue_analysis(
    df: pd.DataFrame,
    asset_type: str = 'offshore_wind',
    n_simulations: int = 1000,
    price_volatility: float = 0.2,
    generation_uncertainty: float = 0.1,
    scenarios: Optional[Dict[str, Dict]] = None
) -> Dict:
    """
    Complete Monte Carlo revenue analysis.
    
    Args:
        df: DataFrame with price and generation data
        asset_type: Type of asset
        n_simulations: Number of Monte Carlo simulations
        price_volatility: Price volatility parameter
        generation_uncertainty: Generation uncertainty parameter
        scenarios: Additional scenario definitions
        
    Returns:
        Dictionary with Monte Carlo results
    """
    logger.info(f"Running Monte Carlo revenue analysis for {asset_type}")
    
    simulator = MonteCarloSimulator(n_simulations=n_simulations)
    
    # Get base data
    base_prices = df['price']
    base_generation = df[asset_type]
    
    # Run base case simulation
    base_results = simulator.simulate_revenue_distribution(
        base_prices,
        base_generation,
        price_volatility=price_volatility,
        generation_uncertainty=generation_uncertainty
    )
    
    results = {
        'base_case': base_results,
        'simulations': n_simulations,
        'parameters': {
            'price_volatility': price_volatility,
            'generation_uncertainty': generation_uncertainty
        }
    }
    
    # Run additional scenarios if provided
    if scenarios:
        results['scenarios'] = {}
        
        for name, params in scenarios.items():
            logger.info(f"\nRunning scenario: {name}")
            
            # Adjust parameters for scenario
            adj_prices = base_prices * params.get('price_adjustment', 1.0)
            adj_generation = base_generation * params.get('generation_adjustment', 1.0)
            adj_volatility = price_volatility * params.get('volatility_adjustment', 1.0)
            
            scenario_results = simulator.simulate_revenue_distribution(
                adj_prices,
                adj_generation,
                price_volatility=adj_volatility,
                generation_uncertainty=generation_uncertainty
            )
            
            results['scenarios'][name] = scenario_results
    
    # Calculate risk metrics
    revenues = base_results['revenues']
    results['risk_metrics'] = {
        'var_95': simulator.value_at_risk(revenues, 0.95),
        'cvar_95': simulator.conditional_value_at_risk(revenues, 0.95),
        'var_99': simulator.value_at_risk(revenues, 0.99),
        'cvar_99': simulator.conditional_value_at_risk(revenues, 0.99)
    }
    
    return results
