#!/usr/bin/env python3
"""
Example Script: Wind/Solar Revenue Modelling Workflow

This script demonstrates the complete end-to-end workflow for renewable
asset revenue modelling using sample data.

Run: python example_workflow.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data import load_sample_data, clean_pipeline
from src.features import feature_engineering_pipeline
from src.models import (
    RevenueCalculator, 
    RevenueScenario, 
    MonteCarloSimulator,
    monte_carlo_revenue_analysis
)
from src.scenarios import create_generation_scenarios, create_price_scenarios
from src.utils import load_config

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def main():
    print("\n" + "#"*70)
    print("  Wind/Solar Revenue Modelling - Example Workflow")
    print("#"*70)
    
    # ==========================================
    # 1. Load Configuration
    # ==========================================
    print_section("1. LOADING CONFIGURATION")
    
    config = load_config('configs/modelling_config.yaml')
    asset_capacity_mw = config.get('asset.capacity_mw', 100)
    asset_type = config.get('asset.type', 'offshore_wind')
    
    print(f"✓ Configuration loaded")
    print(f"  Asset: {asset_capacity_mw}MW {asset_type}")
    
    # ==========================================
    # 2. Load and Clean Data
    # ==========================================
    print_section("2. DATA LOADING AND CLEANING")
    
    # Load sample data
    raw_data = load_sample_data()
    print(f"✓ Sample data loaded: {len(raw_data)} datasets")
    for name, df in raw_data.items():
        print(f"  {name}: {len(df)} records")
    
    # Clean data
    clean_data = clean_pipeline(raw_data, freq='h', missing_method='interpolate')
    print(f"✓ Data cleaned and aligned")
    print(f"  Shape: {clean_data.shape}")
    print(f"  Date range: {clean_data.index.min()} to {clean_data.index.max()}")
    
    # ==========================================
    # 3. Feature Engineering
    # ==========================================
    print_section("3. FEATURE ENGINEERING")
    
    df_features = feature_engineering_pipeline(
        clean_data,
        price_lags=[1, 24, 168],
        demand_lags=[24, 168],
        rolling_windows=[3, 24, 168]
    )
    print(f"✓ Features engineered")
    print(f"  Total features: {len(df_features.columns)}")
    print(f"  Key features: residual_demand, res_share, time features, lags, rolling stats")
    
    # ==========================================
    # 4. Historical Revenue Backcast
    # ==========================================
    print_section("4. HISTORICAL REVENUE BACKCAST")
    
    calculator = RevenueCalculator(asset_capacity_mw=asset_capacity_mw)
    revenue = calculator.calculate_merchant_revenue(
        clean_data['price'],
        clean_data[asset_type]
    )
    
    metrics = calculator.calculate_annual_metrics(
        revenue,
        clean_data[asset_type],
        clean_data['price']
    )
    
    print(f"✓ Historical revenue calculated")
    print(f"  Total Revenue: £{metrics['total_revenue']:,.0f}")
    print(f"  Total Generation: {metrics['total_generation']:,.0f} MWh")
    print(f"  Capacity Factor: {metrics['capacity_factor']:.2%}")
    print(f"  Capture Price: £{metrics['capture_price']:.2f}/MWh")
    print(f"  Avg Market Price: £{metrics['avg_market_price']:.2f}/MWh")
    print(f"  Revenue per MW: £{metrics['revenue_per_mw']:,.0f}/MW")
    
    # ==========================================
    # 5. Contract Structure Comparison
    # ==========================================
    print_section("5. REVENUE CONTRACT COMPARISON")
    
    scenario = RevenueScenario(
        asset_capacity_mw=asset_capacity_mw,
        asset_type=asset_type
    )
    
    scenarios = {
        'merchant': {'scenario_type': 'merchant'},
        'ppa_50': {'scenario_type': 'ppa', 'ppa_price': 50},
        'cfd_55': {'scenario_type': 'cfd', 'strike_price': 55},
        'blended': {'scenario_type': 'blended', 'ppa_price': 50, 'ppa_fraction': 0.5}
    }
    
    comparison = scenario.compare_scenarios(clean_data, scenarios)
    print(f"✓ Scenario comparison complete")
    print(f"\n{comparison[['total_revenue', 'capture_price', 'capacity_factor']].to_string()}")
    
    # ==========================================
    # 6. Generation Scenarios
    # ==========================================
    print_section("6. GENERATION SCENARIOS")
    
    gen_scenarios = create_generation_scenarios(
        clean_data,
        asset_type=asset_type,
        asset_capacity_mw=asset_capacity_mw
    )
    
    print(f"✓ Generation scenarios created: {len(gen_scenarios)}")
    for name, gen in gen_scenarios.items():
        cf = gen.sum() / (asset_capacity_mw * len(gen))
        print(f"  {name}: CF = {cf:.2%}, Total = {gen.sum():,.0f} MWh")
    
    # ==========================================
    # 7. Price Scenarios
    # ==========================================
    print_section("7. PRICE SCENARIOS")
    
    price_scenarios = create_price_scenarios(clean_data, base_scenario='historical')
    
    print(f"✓ Price scenarios created: {len(price_scenarios)}")
    for name, prices in price_scenarios.items():
        print(f"  {name}: Mean = £{prices.mean():.2f}/MWh, "
              f"Std = £{prices.std():.2f}/MWh")
    
    # ==========================================
    # 8. Monte Carlo Simulation
    # ==========================================
    print_section("8. MONTE CARLO REVENUE SIMULATION")
    
    mc_results = monte_carlo_revenue_analysis(
        clean_data,
        asset_type=asset_type,
        n_simulations=1000,
        price_volatility=0.2,
        generation_uncertainty=0.1
    )
    
    base_case = mc_results['base_case']
    risk_metrics = mc_results['risk_metrics']
    
    print(f"✓ Monte Carlo simulation complete (1000 simulations)")
    print(f"\n  Revenue Distribution:")
    print(f"    Mean:          £{base_case['mean']:>15,.0f}")
    print(f"    P10 (downside):£{base_case['p10']:>15,.0f}")
    print(f"    P50 (median):  £{base_case['p50']:>15,.0f}")
    print(f"    P90 (upside):  £{base_case['p90']:>15,.0f}")
    print(f"    Std Dev:       £{base_case['std']:>15,.0f}")
    
    print(f"\n  Risk Metrics:")
    print(f"    VaR 95%:       £{risk_metrics['var_95']:>15,.0f}")
    print(f"    CVaR 95%:      £{risk_metrics['cvar_95']:>15,.0f}")
    
    # ==========================================
    # 9. Sensitivity Analysis
    # ==========================================
    print_section("9. SENSITIVITY ANALYSIS")
    
    base_revenue = (clean_data['price'] * clean_data[asset_type]).sum()
    
    print(f"✓ Price sensitivity analysis:")
    for price_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
        adj_revenue = (clean_data['price'] * price_mult * clean_data[asset_type]).sum()
        change = (adj_revenue / base_revenue - 1) * 100
        print(f"  Price {(price_mult-1)*100:+.0f}%: Revenue {change:+.1f}% "
              f"(£{adj_revenue:,.0f})")
    
    # ==========================================
    # Summary
    # ==========================================
    print_section("WORKFLOW COMPLETE ✓")
    
    print(f"""
Summary:
  • Data processed: {len(clean_data)} hourly records
  • Features created: {len(df_features.columns)}
  • Base case revenue: £{base_revenue:,.0f}
  • Capacity factor: {metrics['capacity_factor']:.2%}
  • Monte Carlo P50: £{base_case['p50']:,.0f}
  • Risk (VaR 95%): £{risk_metrics['var_95']:,.0f}

Next Steps:
  1. Run Jupyter notebooks for detailed analysis
  2. Customize scenarios in configs/modelling_config.yaml
  3. Add your own data to data_raw/
  4. Train ML models for price forecasting
    """)
    
    print("\n" + "#"*70)
    print("  Thank you for using the Revenue Modelling Framework!")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
