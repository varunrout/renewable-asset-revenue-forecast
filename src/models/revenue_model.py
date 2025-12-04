"""
Revenue Modelling Module

Functions for calculating renewable asset revenue under different scenarios.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RevenueCalculator:
    """Calculate revenue for renewable assets."""
    
    def __init__(self, asset_capacity_mw: float = 100):
        """
        Initialize revenue calculator.
        
        Args:
            asset_capacity_mw: Asset capacity in MW
        """
        self.asset_capacity_mw = asset_capacity_mw
    
    def calculate_merchant_revenue(
        self,
        prices: pd.Series,
        generation: pd.Series
    ) -> pd.Series:
        """
        Calculate merchant (spot market) revenue.
        
        Revenue = Price × Generation
        
        Args:
            prices: Electricity prices (£/MWh)
            generation: Generation output (MWh)
            
        Returns:
            Revenue time series (£)
        """
        revenue = prices * generation
        logger.info(f"Merchant revenue: £{revenue.sum():,.0f}")
        return revenue
    
    def calculate_ppa_revenue(
        self,
        generation: pd.Series,
        ppa_price: float
    ) -> pd.Series:
        """
        Calculate fixed-price PPA revenue.
        
        Revenue = PPA Price × Generation
        
        Args:
            generation: Generation output (MWh)
            ppa_price: Fixed PPA price (£/MWh)
            
        Returns:
            Revenue time series (£)
        """
        revenue = generation * ppa_price
        logger.info(f"PPA revenue (£{ppa_price}/MWh): £{revenue.sum():,.0f}")
        return revenue
    
    def calculate_cfd_revenue(
        self,
        prices: pd.Series,
        generation: pd.Series,
        strike_price: float
    ) -> pd.Series:
        """
        Calculate Contract for Difference (CfD) revenue.
        
        Revenue = Strike Price × Generation
        (with implicit top-up/payback vs market price)
        
        Args:
            prices: Market electricity prices (£/MWh)
            generation: Generation output (MWh)
            strike_price: CfD strike price (£/MWh)
            
        Returns:
            Revenue time series (£)
        """
        # CfD gives fixed strike price
        cfd_revenue = generation * strike_price
        
        # Calculate implicit top-up/payback
        market_revenue = prices * generation
        top_up = cfd_revenue - market_revenue
        
        logger.info(f"CfD revenue (strike £{strike_price}/MWh): £{cfd_revenue.sum():,.0f}")
        logger.info(f"Average top-up/payback: £{top_up.mean():,.0f}/period")
        
        return cfd_revenue
    
    def calculate_blended_revenue(
        self,
        prices: pd.Series,
        generation: pd.Series,
        ppa_price: Optional[float] = None,
        ppa_fraction: float = 0.5
    ) -> pd.Series:
        """
        Calculate blended merchant/PPA revenue.
        
        Args:
            prices: Electricity prices (£/MWh)
            generation: Generation output (MWh)
            ppa_price: Fixed PPA price (£/MWh), if None use mean market price
            ppa_fraction: Fraction of generation under PPA (0-1)
            
        Returns:
            Revenue time series (£)
        """
        if ppa_price is None:
            ppa_price = prices.mean()
        
        # Split generation
        ppa_generation = generation * ppa_fraction
        merchant_generation = generation * (1 - ppa_fraction)
        
        # Calculate revenues
        ppa_rev = ppa_generation * ppa_price
        merchant_rev = prices * merchant_generation
        
        total_revenue = ppa_rev + merchant_rev
        
        logger.info(f"Blended revenue ({ppa_fraction*100:.0f}% PPA @ £{ppa_price:.0f}/MWh): "
                   f"£{total_revenue.sum():,.0f}")
        
        return total_revenue
    
    def calculate_capacity_factor(self, generation: pd.Series) -> float:
        """
        Calculate capacity factor.
        
        CF = Actual Generation / Maximum Possible Generation
        
        Args:
            generation: Generation output (MWh)
            
        Returns:
            Capacity factor (0-1)
        """
        # Assuming hourly data
        hours = len(generation)
        max_generation = self.asset_capacity_mw * hours
        actual_generation = generation.sum()
        
        cf = actual_generation / max_generation
        
        logger.info(f"Capacity factor: {cf*100:.1f}%")
        
        return cf
    
    def calculate_capture_price(
        self,
        prices: pd.Series,
        generation: pd.Series
    ) -> float:
        """
        Calculate capture price (volume-weighted average price).
        
        Capture Price = Total Revenue / Total Generation
        
        Args:
            prices: Electricity prices (£/MWh)
            generation: Generation output (MWh)
            
        Returns:
            Capture price (£/MWh)
        """
        total_revenue = (prices * generation).sum()
        total_generation = generation.sum()
        
        if total_generation == 0:
            return 0
        
        capture_price = total_revenue / total_generation
        
        avg_price = prices.mean()
        capture_ratio = capture_price / avg_price if avg_price > 0 else 0
        
        logger.info(f"Capture price: £{capture_price:.2f}/MWh")
        logger.info(f"Average market price: £{avg_price:.2f}/MWh")
        logger.info(f"Capture ratio: {capture_ratio*100:.1f}%")
        
        return capture_price
    
    def calculate_annual_metrics(
        self,
        revenue: pd.Series,
        generation: pd.Series,
        prices: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate annual revenue metrics.
        
        Args:
            revenue: Revenue time series (£)
            generation: Generation output (MWh)
            prices: Electricity prices (£/MWh)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'total_revenue': revenue.sum(),
            'total_generation': generation.sum(),
            'capacity_factor': self.calculate_capacity_factor(generation),
            'capture_price': self.calculate_capture_price(prices, generation),
            'avg_market_price': prices.mean(),
            'revenue_per_mw': revenue.sum() / self.asset_capacity_mw
        }
        
        return metrics


class RevenueScenario:
    """Define and run revenue scenarios."""
    
    def __init__(
        self,
        asset_capacity_mw: float = 100,
        asset_type: str = 'offshore_wind'
    ):
        """
        Initialize revenue scenario.
        
        Args:
            asset_capacity_mw: Asset capacity in MW
            asset_type: Type of asset ('offshore_wind', 'onshore_wind', 'solar')
        """
        self.asset_capacity_mw = asset_capacity_mw
        self.asset_type = asset_type
        self.calculator = RevenueCalculator(asset_capacity_mw)
    
    def run_scenario(
        self,
        df: pd.DataFrame,
        scenario_type: str = 'merchant',
        **kwargs
    ) -> pd.DataFrame:
        """
        Run a revenue scenario.
        
        Args:
            df: DataFrame with price and generation data
            scenario_type: Type of scenario ('merchant', 'ppa', 'cfd', 'blended')
            **kwargs: Additional parameters for scenario
            
        Returns:
            DataFrame with revenue calculations
        """
        logger.info(f"Running {scenario_type} scenario for {self.asset_type}")
        
        result = df.copy()
        
        # Get generation column
        gen_col = self.asset_type
        if gen_col not in df.columns:
            raise ValueError(f"Generation column {gen_col} not found")
        
        # Calculate revenue based on scenario type
        if scenario_type == 'merchant':
            result['revenue'] = self.calculator.calculate_merchant_revenue(
                df['price'], df[gen_col]
            )
        
        elif scenario_type == 'ppa':
            ppa_price = kwargs.get('ppa_price', df['price'].mean())
            result['revenue'] = self.calculator.calculate_ppa_revenue(
                df[gen_col], ppa_price
            )
            result['ppa_price'] = ppa_price
        
        elif scenario_type == 'cfd':
            strike_price = kwargs.get('strike_price', df['price'].mean())
            result['revenue'] = self.calculator.calculate_cfd_revenue(
                df['price'], df[gen_col], strike_price
            )
            result['strike_price'] = strike_price
        
        elif scenario_type == 'blended':
            ppa_price = kwargs.get('ppa_price', df['price'].mean())
            ppa_fraction = kwargs.get('ppa_fraction', 0.5)
            result['revenue'] = self.calculator.calculate_blended_revenue(
                df['price'], df[gen_col], ppa_price, ppa_fraction
            )
            result['ppa_price'] = ppa_price
            result['ppa_fraction'] = ppa_fraction
        
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        # Calculate metrics
        metrics = self.calculator.calculate_annual_metrics(
            result['revenue'], 
            result[gen_col],
            result['price']
        )
        
        result.attrs['metrics'] = metrics
        result.attrs['scenario_type'] = scenario_type
        
        return result
    
    def compare_scenarios(
        self,
        df: pd.DataFrame,
        scenarios: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Compare multiple revenue scenarios.
        
        Args:
            df: DataFrame with price and generation data
            scenarios: Dictionary of scenario definitions
                      e.g., {'merchant': {}, 'ppa': {'ppa_price': 50}}
            
        Returns:
            DataFrame with scenario comparison
        """
        logger.info(f"Comparing {len(scenarios)} scenarios")
        
        results = {}
        
        for name, params in scenarios.items():
            scenario_type = params.pop('scenario_type', name)
            result = self.run_scenario(df, scenario_type, **params)
            results[name] = result.attrs['metrics']
        
        comparison = pd.DataFrame(results).T
        
        logger.info("\nScenario Comparison:")
        logger.info(comparison)
        
        return comparison


def backcast_historical_revenue(
    df: pd.DataFrame,
    asset_capacity_mw: float = 100,
    asset_type: str = 'offshore_wind'
) -> pd.DataFrame:
    """
    Backcast historical revenue for an asset.
    
    Args:
        df: DataFrame with price and generation data
        asset_capacity_mw: Asset capacity in MW
        asset_type: Type of asset
        
    Returns:
        DataFrame with revenue backcast
    """
    logger.info(f"Backcasting revenue for {asset_capacity_mw}MW {asset_type} asset")
    
    scenario = RevenueScenario(asset_capacity_mw, asset_type)
    result = scenario.run_scenario(df, 'merchant')
    
    return result
