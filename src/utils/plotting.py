"""
Plotting Utilities Module

Functions for creating visualizations of revenue, prices, and generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_time_series(
    df: pd.DataFrame,
    columns: List[str],
    title: str = "Time Series",
    ylabel: str = "Value",
    save_path: Optional[str] = None
):
    """
    Plot time series data.
    
    Args:
        df: DataFrame with time series
        columns: List of columns to plot
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for col in columns:
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, alpha=0.8)
    
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_price_vs_residual_demand(
    df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot price vs residual demand scatter.
    
    Args:
        df: DataFrame with price and residual_demand columns
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        df['residual_demand'],
        df['price'],
        c=df.index.hour,
        cmap='viridis',
        alpha=0.5,
        s=10
    )
    
    ax.set_xlabel('Residual Demand (MW)')
    ax.set_ylabel('Price (£/MWh)')
    ax.set_title('Price vs Residual Demand')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Hour of Day')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_res_cannibalisation(
    df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot RES cannibalisation effect (price vs RES share).
    
    Args:
        df: DataFrame with price and res_share columns
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bin RES share for clearer visualization
    df_plot = df[['res_share', 'price']].copy()
    df_plot['res_share_bin'] = pd.cut(df_plot['res_share'], bins=20)
    
    # Calculate mean price per bin
    binned = df_plot.groupby('res_share_bin')['price'].agg(['mean', 'std'])
    bin_centers = [interval.mid for interval in binned.index]
    
    ax.plot(bin_centers, binned['mean'], 'o-', linewidth=2, markersize=6)
    ax.fill_between(
        bin_centers,
        binned['mean'] - binned['std'],
        binned['mean'] + binned['std'],
        alpha=0.3
    )
    
    ax.set_xlabel('RES Share')
    ax.set_ylabel('Average Price (£/MWh)')
    ax.set_title('RES Cannibalisation Effect')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_revenue_distribution(
    revenues: np.ndarray,
    title: str = "Revenue Distribution",
    percentiles: List[float] = [10, 50, 90],
    save_path: Optional[str] = None
):
    """
    Plot revenue distribution from Monte Carlo simulation.
    
    Args:
        revenues: Array of revenue values
        title: Plot title
        percentiles: Percentiles to mark
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(revenues, bins=50, alpha=0.7, edgecolor='black')
    
    # Mark percentiles
    colors = ['red', 'green', 'red']
    for p, color in zip(percentiles, colors):
        percentile_val = np.percentile(revenues, p)
        ax.axvline(percentile_val, color=color, linestyle='--', linewidth=2,
                  label=f'P{p}: £{percentile_val:,.0f}')
    
    ax.set_xlabel('Revenue (£)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_scenario_comparison(
    scenario_results: pd.DataFrame,
    metric: str = 'total_revenue',
    save_path: Optional[str] = None
):
    """
    Plot comparison of revenue scenarios.
    
    Args:
        scenario_results: DataFrame with scenario metrics
        metric: Metric to compare
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = scenario_results.index
    values = scenario_results[metric]
    
    bars = ax.bar(range(len(scenarios)), values, alpha=0.7, edgecolor='black')
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(scenarios)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Scenario Comparison: {metric.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (scenario, value) in enumerate(zip(scenarios, values)):
        ax.text(i, value, f'£{value:,.0f}' if 'revenue' in metric else f'{value:.2f}',
               ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None
):
    """
    Plot feature importance from model.
    
    Args:
        feature_importance: DataFrame with feature and importance columns
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = feature_importance.head(top_n)
    
    ax.barh(range(len(top_features)), top_features['importance'], alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_monthly_revenue(
    revenue: pd.Series,
    title: str = "Monthly Revenue",
    save_path: Optional[str] = None
):
    """
    Plot monthly aggregated revenue.
    
    Args:
        revenue: Revenue time series
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Aggregate by month
    monthly = revenue.resample('M').sum()
    
    ax.bar(monthly.index, monthly.values, width=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue (£)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_price_duration_curve(
    prices: pd.Series,
    save_path: Optional[str] = None
):
    """
    Plot price duration curve.
    
    Args:
        prices: Price time series
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_prices = prices.sort_values(ascending=False).reset_index(drop=True)
    hours = np.arange(len(sorted_prices))
    
    ax.plot(hours, sorted_prices, linewidth=1.5)
    ax.set_xlabel('Hours')
    ax.set_ylabel('Price (£/MWh)')
    ax.set_title('Price Duration Curve')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def create_dashboard(
    df: pd.DataFrame,
    revenue: pd.Series,
    save_path: Optional[str] = None
):
    """
    Create a comprehensive dashboard with multiple plots.
    
    Args:
        df: DataFrame with all data
        revenue: Revenue time series
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Price time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['price'], linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('Price (£/MWh)')
    ax1.set_title('Electricity Price')
    ax1.grid(True, alpha=0.3)
    
    # 2. Generation
    ax2 = fig.add_subplot(gs[1, 0])
    for col in ['offshore_wind', 'onshore_wind', 'solar']:
        if col in df.columns:
            monthly = df[col].resample('M').mean()
            ax2.plot(monthly.index, monthly.values, label=col.replace('_', ' ').title())
    ax2.set_ylabel('Generation (MW)')
    ax2.set_title('Average Monthly Generation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Revenue
    ax3 = fig.add_subplot(gs[1, 1])
    monthly_revenue = revenue.resample('M').sum()
    ax3.bar(monthly_revenue.index, monthly_revenue.values, width=20, alpha=0.7)
    ax3.set_ylabel('Revenue (£)')
    ax3.set_title('Monthly Revenue')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Price vs Residual Demand
    ax4 = fig.add_subplot(gs[2, 0])
    if 'residual_demand' in df.columns:
        ax4.scatter(df['residual_demand'], df['price'], alpha=0.2, s=1)
        ax4.set_xlabel('Residual Demand (MW)')
        ax4.set_ylabel('Price (£/MWh)')
        ax4.set_title('Price vs Residual Demand')
        ax4.grid(True, alpha=0.3)
    
    # 5. RES Share
    ax5 = fig.add_subplot(gs[2, 1])
    if 'res_share' in df.columns:
        monthly_res = df['res_share'].resample('M').mean()
        ax5.plot(monthly_res.index, monthly_res.values * 100, linewidth=2)
        ax5.set_ylabel('RES Share (%)')
        ax5.set_title('Monthly Average RES Share')
        ax5.grid(True, alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Revenue Modelling Dashboard', fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dashboard to {save_path}")
    
    plt.show()
