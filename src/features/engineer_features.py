"""
Feature Engineering Module

Functions for creating features from raw data for price and revenue modelling.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for renewable revenue modelling."""
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        pass
    
    def create_residual_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create residual demand feature.
        
        Residual demand = Total demand - Renewable generation
        
        Args:
            df: DataFrame with demand and generation columns
            
        Returns:
            DataFrame with residual_demand column
        """
        df_feat = df.copy()
        
        # Calculate total renewable generation
        res_cols = []
        if 'offshore_wind' in df.columns:
            res_cols.append('offshore_wind')
        if 'onshore_wind' in df.columns:
            res_cols.append('onshore_wind')
        if 'solar' in df.columns:
            res_cols.append('solar')
        
        if len(res_cols) > 0 and 'demand' in df.columns:
            df_feat['total_res'] = df_feat[res_cols].sum(axis=1)
            df_feat['residual_demand'] = df_feat['demand'] - df_feat['total_res']
            logger.info("Created residual_demand feature")
        else:
            logger.warning("Could not create residual_demand: missing required columns")
        
        return df_feat
    
    def create_res_share(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RES share feature.
        
        RES share = Total RES generation / Total demand
        
        Args:
            df: DataFrame with demand and generation columns
            
        Returns:
            DataFrame with res_share column
        """
        df_feat = df.copy()
        
        # Calculate total renewable generation
        res_cols = []
        if 'offshore_wind' in df.columns:
            res_cols.append('offshore_wind')
        if 'onshore_wind' in df.columns:
            res_cols.append('onshore_wind')
        if 'solar' in df.columns:
            res_cols.append('solar')
        
        if len(res_cols) > 0 and 'demand' in df.columns:
            if 'total_res' not in df_feat.columns:
                df_feat['total_res'] = df_feat[res_cols].sum(axis=1)
            
            # Avoid division by zero
            df_feat['res_share'] = df_feat['total_res'] / df_feat['demand'].replace(0, np.nan)
            logger.info("Created res_share feature")
        else:
            logger.warning("Could not create res_share: missing required columns")
        
        return df_feat
    
    def create_generation_shares(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create individual generation technology shares.
        
        Args:
            df: DataFrame with demand and generation columns
            
        Returns:
            DataFrame with offshore_share, onshore_share, solar_share columns
        """
        df_feat = df.copy()
        
        if 'demand' not in df.columns:
            logger.warning("Could not create generation shares: demand column missing")
            return df_feat
        
        # Create shares for each technology
        for tech in ['offshore_wind', 'onshore_wind', 'solar']:
            if tech in df.columns:
                share_col = f"{tech.replace('_wind', '')}_share"
                df_feat[share_col] = df_feat[tech] / df_feat['demand'].replace(0, np.nan)
                logger.info(f"Created {share_col} feature")
        
        return df_feat
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create calendar and time-based features.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with time features
        """
        df_feat = df.copy()
        
        # Hour of day
        df_feat['hour'] = df_feat.index.hour
        
        # Day of week (0=Monday, 6=Sunday)
        df_feat['day_of_week'] = df_feat.index.dayofweek
        
        # Month
        df_feat['month'] = df_feat.index.month
        
        # Season (1=Winter, 2=Spring, 3=Summer, 4=Autumn)
        df_feat['season'] = df_feat['month'].apply(
            lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] 
            else 3 if x in [6, 7, 8] else 4
        )
        
        # Weekend flag
        df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
        
        # Year
        df_feat['year'] = df_feat.index.year
        
        # Day of year
        df_feat['day_of_year'] = df_feat.index.dayofyear
        
        # Cyclical encoding for hour (useful for ML models)
        df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
        df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
        
        # Cyclical encoding for day of week
        df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
        df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)
        
        # Cyclical encoding for month
        df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
        df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
        
        logger.info("Created time features")
        
        return df_feat
    
    def create_lag_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        lags: List[int] = [1, 24, 168]
    ) -> pd.DataFrame:
        """
        Create lag features for specified columns.
        
        Args:
            df: Input dataframe
            columns: Columns to create lags for
            lags: List of lag periods (in hours)
            
        Returns:
            DataFrame with lag features
        """
        df_feat = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping lags")
                continue
            
            for lag in lags:
                lag_col = f"{col}_lag_{lag}h"
                df_feat[lag_col] = df_feat[col].shift(lag)
                logger.info(f"Created {lag_col}")
        
        return df_feat
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [3, 24, 168],
        agg_funcs: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input dataframe
            columns: Columns to create rolling features for
            windows: Rolling window sizes (in hours)
            agg_funcs: Aggregation functions to apply
            
        Returns:
            DataFrame with rolling features
        """
        df_feat = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping rolling features")
                continue
            
            for window in windows:
                for func in agg_funcs:
                    feat_col = f"{col}_roll_{window}h_{func}"
                    
                    if func == 'mean':
                        df_feat[feat_col] = df_feat[col].rolling(window).mean()
                    elif func == 'std':
                        df_feat[feat_col] = df_feat[col].rolling(window).std()
                    elif func == 'min':
                        df_feat[feat_col] = df_feat[col].rolling(window).min()
                    elif func == 'max':
                        df_feat[feat_col] = df_feat[col].rolling(window).max()
                    
                    logger.info(f"Created {feat_col}")
        
        return df_feat
    
    def create_price_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price spread and volatility features.
        
        Args:
            df: DataFrame with price column
            
        Returns:
            DataFrame with price spread features
        """
        df_feat = df.copy()
        
        if 'price' not in df.columns:
            logger.warning("Price column not found, skipping price spread features")
            return df_feat
        
        # Price changes
        df_feat['price_change_1h'] = df_feat['price'].diff(1)
        df_feat['price_change_24h'] = df_feat['price'].diff(24)
        
        # Price volatility (rolling std)
        df_feat['price_volatility_24h'] = df_feat['price'].rolling(24).std()
        df_feat['price_volatility_168h'] = df_feat['price'].rolling(168).std()
        
        # Price range
        df_feat['price_range_24h'] = (
            df_feat['price'].rolling(24).max() - 
            df_feat['price'].rolling(24).min()
        )
        
        logger.info("Created price spread features")
        
        return df_feat
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        price_lags: List[int] = [1, 24, 168],
        demand_lags: List[int] = [24, 168],
        rolling_windows: List[int] = [3, 24, 168]
    ) -> pd.DataFrame:
        """
        Create all features in one pipeline.
        
        Args:
            df: Input dataframe
            price_lags: Lag periods for price
            demand_lags: Lag periods for demand
            rolling_windows: Rolling window sizes
            
        Returns:
            DataFrame with all features
        """
        logger.info("Creating all features")
        
        df_feat = df.copy()
        
        # Fundamental features
        df_feat = self.create_residual_demand(df_feat)
        df_feat = self.create_res_share(df_feat)
        df_feat = self.create_generation_shares(df_feat)
        
        # Time features
        df_feat = self.create_time_features(df_feat)
        
        # Lag features
        if 'price' in df.columns:
            df_feat = self.create_lag_features(df_feat, ['price'], price_lags)
        if 'demand' in df.columns:
            df_feat = self.create_lag_features(df_feat, ['demand'], demand_lags)
        if 'residual_demand' in df_feat.columns:
            df_feat = self.create_lag_features(df_feat, ['residual_demand'], demand_lags)
        
        # Rolling features
        rolling_cols = []
        if 'demand' in df.columns:
            rolling_cols.append('demand')
        if 'residual_demand' in df_feat.columns:
            rolling_cols.append('residual_demand')
        if 'total_res' in df_feat.columns:
            rolling_cols.append('total_res')
        
        if len(rolling_cols) > 0:
            df_feat = self.create_rolling_features(
                df_feat, 
                rolling_cols, 
                windows=rolling_windows,
                agg_funcs=['mean', 'std']
            )
        
        # Price spread features
        if 'price' in df.columns:
            df_feat = self.create_price_spread_features(df_feat)
        
        logger.info(f"Feature engineering complete. Total features: {len(df_feat.columns)}")
        
        return df_feat


def feature_engineering_pipeline(
    df: pd.DataFrame,
    price_lags: List[int] = [1, 24, 168],
    demand_lags: List[int] = [24, 168],
    rolling_windows: List[int] = [3, 24, 168]
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Clean input dataframe
        price_lags: Lag periods for price
        demand_lags: Lag periods for demand
        rolling_windows: Rolling window sizes
        
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(
        df,
        price_lags=price_lags,
        demand_lags=demand_lags,
        rolling_windows=rolling_windows
    )
    
    return df_features
