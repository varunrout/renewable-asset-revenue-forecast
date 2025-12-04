"""
Data Loading Module

Functions for loading raw data files (prices, generation, demand).
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading of all raw data files."""
    
    def __init__(self, data_path: Union[str, Path] = "data_raw"):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to raw data directory
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist")
    
    def load_prices(self, filename: str = "prices.csv") -> pd.DataFrame:
        """
        Load wholesale electricity prices data.
        
        Expected columns: timestamp, price
        
        Args:
            filename: Name of prices file
            
        Returns:
            DataFrame with timestamp index and price column
        """
        filepath = self.data_path / filename
        logger.info(f"Loading prices from {filepath}")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        logger.info(f"Loaded {len(df)} price records")
        return df
    
    def load_offshore_wind(self, filename: str = "offshore_wind.csv") -> pd.DataFrame:
        """
        Load offshore wind generation data.
        
        Expected columns: timestamp, generation (MW or MWh)
        
        Args:
            filename: Name of offshore wind file
            
        Returns:
            DataFrame with timestamp index and generation column
        """
        filepath = self.data_path / filename
        logger.info(f"Loading offshore wind from {filepath}")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Rename generation column to offshore_wind
        if 'generation' in df.columns:
            df = df.rename(columns={'generation': 'offshore_wind'})
        
        logger.info(f"Loaded {len(df)} offshore wind records")
        return df
    
    def load_onshore_wind(self, filename: str = "onshore_wind.csv") -> pd.DataFrame:
        """
        Load onshore wind generation data.
        
        Expected columns: timestamp, generation (MW or MWh)
        
        Args:
            filename: Name of onshore wind file
            
        Returns:
            DataFrame with timestamp index and generation column
        """
        filepath = self.data_path / filename
        logger.info(f"Loading onshore wind from {filepath}")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Rename generation column to onshore_wind
        if 'generation' in df.columns:
            df = df.rename(columns={'generation': 'onshore_wind'})
        
        logger.info(f"Loaded {len(df)} onshore wind records")
        return df
    
    def load_solar(self, filename: str = "solar.csv") -> pd.DataFrame:
        """
        Load solar generation data.
        
        Expected columns: timestamp, generation (MW or MWh)
        
        Args:
            filename: Name of solar file
            
        Returns:
            DataFrame with timestamp index and generation column
        """
        filepath = self.data_path / filename
        logger.info(f"Loading solar from {filepath}")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Rename generation column to solar
        if 'generation' in df.columns:
            df = df.rename(columns={'generation': 'solar'})
        
        logger.info(f"Loaded {len(df)} solar records")
        return df
    
    def load_demand(self, filename: str = "demand.csv") -> pd.DataFrame:
        """
        Load system demand data.
        
        Expected columns: timestamp, demand (MW or MWh)
        
        Args:
            filename: Name of demand file
            
        Returns:
            DataFrame with timestamp index and demand column
        """
        filepath = self.data_path / filename
        logger.info(f"Loading demand from {filepath}")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        logger.info(f"Loaded {len(df)} demand records")
        return df
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available data files.
        
        Returns:
            Dictionary with keys: prices, offshore_wind, onshore_wind, solar, demand
        """
        data = {}
        
        # Try to load each dataset
        datasets = {
            'prices': 'prices.csv',
            'offshore_wind': 'offshore_wind.csv',
            'onshore_wind': 'onshore_wind.csv',
            'solar': 'solar.csv',
            'demand': 'demand.csv'
        }
        
        for name, filename in datasets.items():
            filepath = self.data_path / filename
            if filepath.exists():
                load_func = getattr(self, f'load_{name}')
                data[name] = load_func(filename)
            else:
                logger.warning(f"File not found: {filepath}")
        
        return data


def load_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Load sample/demo data for testing.
    
    Returns:
        Dictionary with sample dataframes
    """
    # Create sample timestamp range
    timestamps = pd.date_range('2023-01-01', '2023-12-31', freq='h')
    
    # Sample data
    data = {
        'prices': pd.DataFrame({
            'price': 50 + 20 * pd.Series(range(len(timestamps))).apply(
                lambda x: (x % 24) / 24
            )
        }, index=timestamps),
        'offshore_wind': pd.DataFrame({
            'offshore_wind': 1000 * pd.Series(range(len(timestamps))).apply(
                lambda x: 0.3 + 0.3 * ((x % 168) / 168)
            )
        }, index=timestamps),
        'onshore_wind': pd.DataFrame({
            'onshore_wind': 800 * pd.Series(range(len(timestamps))).apply(
                lambda x: 0.25 + 0.25 * ((x % 168) / 168)
            )
        }, index=timestamps),
        'solar': pd.DataFrame({
            'solar': 600 * pd.Series(range(len(timestamps))).apply(
                lambda x: max(0, (1 - abs((x % 24) - 12) / 12))
            )
        }, index=timestamps),
        'demand': pd.DataFrame({
            'demand': 30000 + 5000 * pd.Series(range(len(timestamps))).apply(
                lambda x: (x % 24) / 24
            )
        }, index=timestamps)
    }
    
    logger.info("Loaded sample data")
    return data
