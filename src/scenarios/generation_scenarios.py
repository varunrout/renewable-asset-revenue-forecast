"""
Generation Scenarios Module

Functions for creating generation scenarios (capacity factors, build-out, etc.).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerationScenarioBuilder:
    """Build generation scenarios for renewable assets."""
    
    def __init__(self):
        """Initialize generation scenario builder."""
        pass
    
    def scale_by_capacity_factor(
        self,
        base_generation: pd.Series,
        target_cf: float,
        asset_capacity_mw: float
    ) -> pd.Series:
        """
        Scale generation to achieve target capacity factor.
        
        Args:
            base_generation: Base generation time series (MWh)
            target_cf: Target capacity factor (0-1)
            asset_capacity_mw: Asset capacity (MW)
            
        Returns:
            Scaled generation series
        """
        # Calculate current CF
        hours = len(base_generation)
        max_generation = asset_capacity_mw * hours
        current_cf = base_generation.sum() / max_generation
        
        # Scale to target
        scaling_factor = target_cf / current_cf
        scaled_generation = base_generation * scaling_factor
        
        logger.info(f"Scaled generation from CF={current_cf:.2%} to CF={target_cf:.2%}")
        
        return scaled_generation
    
    def apply_seasonal_adjustment(
        self,
        generation: pd.Series,
        seasonal_factors: Dict[str, float]
    ) -> pd.Series:
        """
        Apply seasonal adjustment factors to generation.
        
        Args:
            generation: Generation time series
            seasonal_factors: Dictionary mapping season to factor
                            e.g., {'winter': 1.2, 'spring': 1.0, 'summer': 0.8, 'autumn': 1.0}
            
        Returns:
            Adjusted generation series
        """
        adjusted = generation.copy()
        
        # Define seasons
        season_months = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'autumn': [9, 10, 11]
        }
        
        # Apply factors
        for season, months in season_months.items():
            if season in seasonal_factors:
                mask = adjusted.index.month.isin(months)
                adjusted.loc[mask] = adjusted.loc[mask] * seasonal_factors[season]
        
        logger.info(f"Applied seasonal adjustments: {seasonal_factors}")
        
        return adjusted
    
    def create_high_wind_year(
        self,
        base_generation: pd.Series,
        uplift: float = 0.15
    ) -> pd.Series:
        """
        Create a high wind year scenario.
        
        Args:
            base_generation: Base wind generation
            uplift: Uplift factor (e.g., 0.15 = 15% increase)
            
        Returns:
            High wind generation series
        """
        high_wind = base_generation * (1 + uplift)
        logger.info(f"Created high wind scenario (+{uplift:.1%})")
        return high_wind
    
    def create_low_wind_year(
        self,
        base_generation: pd.Series,
        reduction: float = 0.15
    ) -> pd.Series:
        """
        Create a low wind year scenario.
        
        Args:
            base_generation: Base wind generation
            reduction: Reduction factor (e.g., 0.15 = 15% decrease)
            
        Returns:
            Low wind generation series
        """
        low_wind = base_generation * (1 - reduction)
        logger.info(f"Created low wind scenario (-{reduction:.1%})")
        return low_wind
    
    def create_fleet_buildout_scenario(
        self,
        base_generation: pd.Series,
        buildout_factor: float = 1.5,
        ramp_years: Optional[int] = None
    ) -> pd.Series:
        """
        Create a fleet build-out scenario.
        
        Args:
            base_generation: Base fleet generation
            buildout_factor: Final capacity factor (e.g., 1.5 = 50% increase)
            ramp_years: Years to ramp up (if None, apply immediately)
            
        Returns:
            Build-out generation series
        """
        if ramp_years is None:
            # Immediate build-out
            buildout = base_generation * buildout_factor
        else:
            # Gradual ramp-up
            buildout = base_generation.copy()
            years = buildout.index.year
            start_year = years.min()
            
            for year in years.unique():
                year_progress = min((year - start_year) / ramp_years, 1.0)
                year_factor = 1.0 + (buildout_factor - 1.0) * year_progress
                buildout.loc[years == year] *= year_factor
        
        logger.info(f"Created fleet build-out scenario (×{buildout_factor})")
        
        return buildout
    
    def create_degradation_scenario(
        self,
        generation: pd.Series,
        annual_degradation: float = 0.005
    ) -> pd.Series:
        """
        Apply degradation over time to generation.
        
        Args:
            generation: Generation time series
            annual_degradation: Annual degradation rate (e.g., 0.005 = 0.5%/year)
            
        Returns:
            Generation with degradation applied
        """
        degraded = generation.copy()
        
        years = degraded.index.year
        start_year = years.min()
        
        for year in years.unique():
            years_elapsed = year - start_year
            degradation_factor = (1 - annual_degradation) ** years_elapsed
            degraded.loc[years == year] *= degradation_factor
        
        logger.info(f"Applied degradation ({annual_degradation:.1%}/year)")
        
        return degraded
    
    def create_technology_mix_scenario(
        self,
        df: pd.DataFrame,
        offshore_share: float = 0.4,
        onshore_share: float = 0.3,
        solar_share: float = 0.3
    ) -> pd.Series:
        """
        Create combined generation from technology mix.
        
        Args:
            df: DataFrame with generation columns
            offshore_share: Share of offshore wind
            onshore_share: Share of onshore wind
            solar_share: Share of solar
            
        Returns:
            Combined generation series
        """
        total = (
            df.get('offshore_wind', 0) * offshore_share +
            df.get('onshore_wind', 0) * onshore_share +
            df.get('solar', 0) * solar_share
        )
        
        logger.info(f"Created tech mix: offshore={offshore_share:.1%}, "
                   f"onshore={onshore_share:.1%}, solar={solar_share:.1%}")
        
        return total


def create_generation_scenarios(
    df: pd.DataFrame,
    asset_type: str = 'offshore_wind',
    asset_capacity_mw: float = 100
) -> Dict[str, pd.Series]:
    """
    Create a suite of generation scenarios.
    
    Args:
        df: DataFrame with generation data
        asset_type: Type of asset
        asset_capacity_mw: Asset capacity in MW
        
    Returns:
        Dictionary of generation scenarios
    """
    logger.info(f"Creating generation scenarios for {asset_type}")
    
    builder = GenerationScenarioBuilder()
    base_generation = df[asset_type]
    
    scenarios = {
        'base': base_generation,
        'high_wind': builder.create_high_wind_year(base_generation, uplift=0.15),
        'low_wind': builder.create_low_wind_year(base_generation, reduction=0.15),
        'p90_cf': builder.scale_by_capacity_factor(base_generation, 0.35, asset_capacity_mw),
        'p10_cf': builder.scale_by_capacity_factor(base_generation, 0.55, asset_capacity_mw),
    }
    
    # Add seasonal scenarios if applicable
    if asset_type in ['offshore_wind', 'onshore_wind']:
        scenarios['winter_heavy'] = builder.apply_seasonal_adjustment(
            base_generation,
            {'winter': 1.3, 'spring': 1.0, 'summer': 0.7, 'autumn': 1.0}
        )
    
    if asset_type == 'solar':
        scenarios['summer_heavy'] = builder.apply_seasonal_adjustment(
            base_generation,
            {'winter': 0.6, 'spring': 1.0, 'summer': 1.4, 'autumn': 1.0}
        )
    
    logger.info(f"Created {len(scenarios)} generation scenarios")
    
    return scenarios
