"""
Data Cleaning Module

Functions for cleaning, aligning, and validating raw data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning and alignment operations."""
    
    def __init__(self, freq: str = 'h'):
        """
        Initialize DataCleaner.
        
        Args:
            freq: Target frequency for alignment ('h' for hourly, '30min', etc.)
        """
        self.freq = freq
    
    def align_timestamps(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align all dataframes to common timestamp range.
        
        Args:
            dfs: Dictionary of dataframes with timestamp indices
            
        Returns:
            Dictionary of aligned dataframes
        """
        logger.info("Aligning timestamps across datasets")
        
        # Find common timestamp range
        start_dates = [df.index.min() for df in dfs.values()]
        end_dates = [df.index.max() for df in dfs.values()]
        
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        logger.info(f"Common range: {common_start} to {common_end}")
        
        # Create complete timestamp index
        full_index = pd.date_range(common_start, common_end, freq=self.freq)
        
        # Reindex all dataframes
        aligned_dfs = {}
        for name, df in dfs.items():
            aligned_dfs[name] = df.reindex(full_index)
            logger.info(f"Aligned {name}: {len(aligned_dfs[name])} records")
        
        return aligned_dfs
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame,
        method: str = 'interpolate',
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in a dataframe.
        
        Args:
            df: Input dataframe
            method: Method for handling missing values:
                    - 'interpolate': Linear interpolation
                    - 'forward': Forward fill
                    - 'backward': Backward fill
                    - 'drop': Drop rows with missing values
                    - 'zero': Fill with zeros
            limit: Maximum number of consecutive NaNs to fill
            
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Handling missing values with method: {method}")
        
        df_clean = df.copy()
        missing_before = df_clean.isna().sum().sum()
        
        if method == 'interpolate':
            df_clean = df_clean.interpolate(method='linear', limit=limit, limit_direction='both')
        elif method == 'forward':
            df_clean = df_clean.fillna(method='ffill', limit=limit)
        elif method == 'backward':
            df_clean = df_clean.fillna(method='bfill', limit=limit)
        elif method == 'drop':
            df_clean = df_clean.dropna()
        elif method == 'zero':
            df_clean = df_clean.fillna(0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        missing_after = df_clean.isna().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
        
        return df_clean
    
    def normalize_units(
        self, 
        df: pd.DataFrame, 
        column: str,
        source_unit: str = 'MW',
        target_unit: str = 'MW'
    ) -> pd.DataFrame:
        """
        Normalize generation units.
        
        Args:
            df: Input dataframe
            column: Column to normalize
            source_unit: Current unit (MW, MWh, GW, GWh, kW, kWh)
            target_unit: Target unit (MW, MWh, GW, GWh)
            
        Returns:
            Dataframe with normalized units
        """
        df_norm = df.copy()
        
        # Conversion factors to MW
        to_mw = {
            'kW': 0.001,
            'MW': 1,
            'GW': 1000,
            'kWh': 0.001,  # Assuming hourly data
            'MWh': 1,      # Assuming hourly data
            'GWh': 1000    # Assuming hourly data
        }
        
        # Conversion factors from MW
        from_mw = {
            'kW': 1000,
            'MW': 1,
            'GW': 0.001,
            'kWh': 1000,
            'MWh': 1,
            'GWh': 0.001
        }
        
        # Convert source to MW, then to target
        conversion_factor = to_mw[source_unit] * from_mw[target_unit]
        df_norm[column] = df_norm[column] * conversion_factor
        
        logger.info(f"Normalized {column} from {source_unit} to {target_unit}")
        
        return df_norm
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers from a column.
        
        Args:
            df: Input dataframe
            column: Column to clean
            method: Method for outlier detection:
                    - 'iqr': Interquartile range method
                    - 'zscore': Z-score method
                    - 'percentile': Percentile-based clipping
            threshold: Threshold parameter (IQR multiplier, z-score, or percentile)
            
        Returns:
            Dataframe with outliers removed
        """
        df_clean = df.copy()
        initial_count = len(df_clean)
        
        if method == 'iqr':
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[
                (df_clean[column] >= lower_bound) & 
                (df_clean[column] <= upper_bound)
            ]
        
        elif method == 'zscore':
            z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
            df_clean = df_clean[z_scores < threshold]
        
        elif method == 'percentile':
            lower_percentile = (100 - threshold) / 2
            upper_percentile = 100 - lower_percentile
            lower_bound = df_clean[column].quantile(lower_percentile / 100)
            upper_bound = df_clean[column].quantile(upper_percentile / 100)
            df_clean = df_clean[
                (df_clean[column] >= lower_bound) & 
                (df_clean[column] <= upper_bound)
            ]
        
        removed_count = initial_count - len(df_clean)
        logger.info(f"Removed {removed_count} outliers from {column} using {method}")
        
        return df_clean
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data quality.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check for missing values
        missing = df.isna().sum()
        if missing.any():
            issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            issues.append(f"Duplicate timestamps found: {df.index.duplicated().sum()}")
        
        # Check for negative values (where inappropriate)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df[col] < 0).any():
                issues.append(f"Negative values found in {col}")
        
        # Check for constant values
        for col in numeric_cols:
            if df[col].nunique() == 1:
                issues.append(f"Constant values in {col}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation found {len(issues)} issues")
        
        return is_valid, issues
    
    def merge_datasets(self, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all datasets into a single dataframe.
        
        Args:
            dfs: Dictionary of dataframes
            
        Returns:
            Merged dataframe
        """
        logger.info("Merging all datasets")
        
        # Start with first dataset
        merged = None
        
        for name, df in dfs.items():
            if merged is None:
                merged = df.copy()
            else:
                merged = merged.join(df, how='outer')
        
        logger.info(f"Merged dataset shape: {merged.shape}")
        
        return merged


def clean_pipeline(
    raw_data: Dict[str, pd.DataFrame],
    freq: str = 'h',
    missing_method: str = 'interpolate',
    missing_limit: int = 3
) -> pd.DataFrame:
    """
    Complete data cleaning pipeline.
    
    Args:
        raw_data: Dictionary of raw dataframes
        freq: Target frequency
        missing_method: Method for handling missing values
        missing_limit: Maximum consecutive NaNs to fill
        
    Returns:
        Clean, merged dataframe
    """
    cleaner = DataCleaner(freq=freq)
    
    # Align timestamps
    aligned_data = cleaner.align_timestamps(raw_data)
    
    # Handle missing values for each dataset
    cleaned_data = {}
    for name, df in aligned_data.items():
        cleaned_data[name] = cleaner.handle_missing_values(
            df, 
            method=missing_method,
            limit=missing_limit
        )
    
    # Merge all datasets
    merged = cleaner.merge_datasets(cleaned_data)
    
    # Final validation
    is_valid, issues = cleaner.validate_data(merged)
    if not is_valid:
        logger.warning(f"Final validation issues: {issues}")
    
    return merged
