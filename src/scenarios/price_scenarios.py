"""
Price Scenarios Module

Functions for creating price scenarios (gas prices, RES cannibalisation, etc.).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceScenarioBuilder:
    """Build price scenarios."""
    
    def __init__(self):
        """Initialize price scenario builder."""
        pass
    
    def apply_price_shock(
        self,
        prices: pd.Series,
        shock_factor: float = 1.5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Apply a price shock over a period.
        
        Args:
            prices: Base price series
            shock_factor: Price multiplier
            start_date: Start date for shock (if None, apply to all)
            end_date: End date for shock (if None, apply to end)
            
        Returns:
            Price series with shock applied
        """
        shocked_prices = prices.copy()
        
        if start_date or end_date:
            mask = pd.Series(True, index=prices.index)
            if start_date:
                mask &= prices.index >= pd.to_datetime(start_date)
            if end_date:
                mask &= prices.index <= pd.to_datetime(end_date)
            
            shocked_prices.loc[mask] *= shock_factor
        else:
            shocked_prices *= shock_factor
        
        logger.info(f"Applied price shock (×{shock_factor})")
        
        return shocked_prices
    
    def apply_gas_price_scenario(
        self,
        prices: pd.Series,
        gas_price_change: float = 0.5
    ) -> pd.Series:
        """
        Adjust electricity prices based on gas price changes.
        
        Simplified model: electricity price changes proportionally to gas
        
        Args:
            prices: Base electricity prices
            gas_price_change: Fractional change in gas prices (e.g., 0.5 = 50% increase)
            
        Returns:
            Adjusted price series
        """
        # Simple linear relationship: assume 60% of price is driven by gas
        gas_sensitivity = 0.6
        price_adjustment = 1 + (gas_price_change * gas_sensitivity)
        
        adjusted_prices = prices * price_adjustment
        
        logger.info(f"Applied gas price scenario ({gas_price_change:+.1%} gas → "
                   f"{(price_adjustment-1):.1%} electricity)")
        
        return adjusted_prices
    
    def apply_res_cannibalisation(
        self,
        df: pd.DataFrame,
        cannibalisation_factor: float = 0.5
    ) -> pd.Series:
        """
        Apply RES cannibalisation effect to prices.
        
        Higher RES share → lower prices (cannibalisation effect)
        
        Args:
            df: DataFrame with price and res_share columns
            cannibalisation_factor: Strength of cannibalisation effect
            
        Returns:
            Adjusted price series
        """
        if 'res_share' not in df.columns:
            logger.warning("res_share not found, cannot apply cannibalisation")
            return df['price']
        
        # Price reduction proportional to RES share
        # New price = Base price × (1 - cannibalisation_factor × res_share)
        price_adjustment = 1 - (cannibalisation_factor * df['res_share'])
        price_adjustment = price_adjustment.clip(lower=0.1)  # Floor at 10% of base price
        
        adjusted_prices = df['price'] * price_adjustment
        
        avg_reduction = (1 - price_adjustment.mean()) * 100
        logger.info(f"Applied RES cannibalisation (avg reduction: {avg_reduction:.1f}%)")
        
        return adjusted_prices
    
    def create_high_res_scenario(
        self,
        df: pd.DataFrame,
        res_increase: float = 0.5
    ) -> pd.Series:
        """
        Create high RES penetration scenario.
        
        Args:
            df: DataFrame with generation and price data
            res_increase: Fractional increase in RES (e.g., 0.5 = 50% more)
            
        Returns:
            Adjusted price series
        """
        df_scenario = df.copy()
        
        # Increase RES generation
        for col in ['offshore_wind', 'onshore_wind', 'solar']:
            if col in df_scenario.columns:
                df_scenario[col] *= (1 + res_increase)
        
        # Recalculate RES share
        if 'demand' in df_scenario.columns:
            total_res = sum([df_scenario.get(col, 0) for col in 
                           ['offshore_wind', 'onshore_wind', 'solar']])
            df_scenario['res_share'] = total_res / df_scenario['demand']
        
        # Apply cannibalisation
        adjusted_prices = self.apply_res_cannibalisation(df_scenario, cannibalisation_factor=0.4)
        
        return adjusted_prices
    
    def create_demand_scenario(
        self,
        prices: pd.Series,
        df: pd.DataFrame,
        demand_change: float = 0.2
    ) -> pd.Series:
        """
        Create demand growth/reduction scenario.
        
        Args:
            prices: Base prices
            df: DataFrame with demand data
            demand_change: Fractional change in demand (e.g., 0.2 = 20% increase)
            
        Returns:
            Adjusted price series
        """
        # Higher demand → higher prices (simplified elastic relationship)
        demand_elasticity = 0.3  # 10% demand increase → 3% price increase
        price_change = demand_change * demand_elasticity
        
        adjusted_prices = prices * (1 + price_change)
        
        logger.info(f"Applied demand scenario ({demand_change:+.1%} demand → "
                   f"{price_change:+.1%} price)")
        
        return adjusted_prices
    
    def create_carbon_price_scenario(
        self,
        prices: pd.Series,
        carbon_price_gbp_per_tonne: float = 50,
        gas_carbon_intensity: float = 0.4  # tCO2/MWh
    ) -> pd.Series:
        """
        Add carbon price impact to electricity prices.
        
        Args:
            prices: Base prices
            carbon_price_gbp_per_tonne: Carbon price in £/tCO2
            gas_carbon_intensity: Carbon intensity of marginal generator (tCO2/MWh)
            
        Returns:
            Adjusted price series
        """
        carbon_adder = carbon_price_gbp_per_tonne * gas_carbon_intensity
        adjusted_prices = prices + carbon_adder
        
        logger.info(f"Applied carbon price (£{carbon_price_gbp_per_tonne}/tCO2 → "
                   f"+£{carbon_adder:.2f}/MWh)")
        
        return adjusted_prices
    
    def create_seasonal_pattern(
        self,
        prices: pd.Series,
        winter_factor: float = 1.3,
        summer_factor: float = 0.8
    ) -> pd.Series:
        """
        Apply seasonal price pattern.
        
        Args:
            prices: Base prices
            winter_factor: Winter price multiplier
            summer_factor: Summer price multiplier
            
        Returns:
            Prices with seasonal pattern
        """
        seasonal_prices = prices.copy()
        
        # Define seasons
        winter_months = [12, 1, 2]
        summer_months = [6, 7, 8]
        
        # Apply factors
        seasonal_prices.loc[seasonal_prices.index.month.isin(winter_months)] *= winter_factor
        seasonal_prices.loc[seasonal_prices.index.month.isin(summer_months)] *= summer_factor
        
        logger.info(f"Applied seasonal pattern (winter ×{winter_factor}, summer ×{summer_factor})")
        
        return seasonal_prices


def create_price_scenarios(
    df: pd.DataFrame,
    base_scenario: str = 'historical'
) -> Dict[str, pd.Series]:
    """
    Create a suite of price scenarios.
    
    Args:
        df: DataFrame with price and generation data
        base_scenario: Base scenario to use ('historical', 'mean', 'median')
        
    Returns:
        Dictionary of price scenarios
    """
    logger.info("Creating price scenarios")
    
    builder = PriceScenarioBuilder()
    
    # Get base prices
    if base_scenario == 'historical':
        base_prices = df['price']
    elif base_scenario == 'mean':
        base_prices = pd.Series(df['price'].mean(), index=df.index)
    elif base_scenario == 'median':
        base_prices = pd.Series(df['price'].median(), index=df.index)
    else:
        base_prices = df['price']
    
    scenarios = {
        'base': base_prices,
        'high_gas': builder.apply_gas_price_scenario(base_prices, gas_price_change=0.5),
        'low_gas': builder.apply_gas_price_scenario(base_prices, gas_price_change=-0.3),
        'high_demand': builder.create_demand_scenario(base_prices, df, demand_change=0.2),
        'low_demand': builder.create_demand_scenario(base_prices, df, demand_change=-0.2),
        'high_res': builder.create_high_res_scenario(df, res_increase=0.5),
        'carbon_price_high': builder.create_carbon_price_scenario(
            base_prices, 
            carbon_price_gbp_per_tonne=100
        ),
    }
    
    logger.info(f"Created {len(scenarios)} price scenarios")
    
    return scenarios


def create_combined_scenarios(
    df: pd.DataFrame,
    asset_type: str = 'offshore_wind'
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Create combined price and generation scenarios.
    
    Args:
        df: DataFrame with all data
        asset_type: Type of asset
        
    Returns:
        Dictionary of combined scenarios
    """
    logger.info("Creating combined scenarios")
    
    price_builder = PriceScenarioBuilder()
    
    scenarios = {
        'base_case': {
            'price': df['price'],
            'generation': df[asset_type]
        },
        
        'optimistic': {
            'price': price_builder.apply_gas_price_scenario(
                df['price'], gas_price_change=0.3
            ),
            'generation': df[asset_type] * 1.15
        },
        
        'pessimistic': {
            'price': price_builder.create_high_res_scenario(df, res_increase=1.0),
            'generation': df[asset_type] * 0.85
        },
        
        'high_price_low_gen': {
            'price': price_builder.apply_gas_price_scenario(
                df['price'], gas_price_change=0.5
            ),
            'generation': df[asset_type] * 0.85
        },
        
        'low_price_high_gen': {
            'price': price_builder.create_high_res_scenario(df, res_increase=0.8),
            'generation': df[asset_type] * 1.15
        },
    }
    
    logger.info(f"Created {len(scenarios)} combined scenarios")
    
    return scenarios
