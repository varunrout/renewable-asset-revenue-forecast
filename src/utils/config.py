"""
Configuration Utilities Module

Functions for loading and managing configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config file. If None, uses default config.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "modelling_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self._default_config()
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'data': {
                'raw_path': 'data_raw',
                'processed_path': 'data_processed',
                'freq': 'h'
            },
            'asset': {
                'capacity_mw': 100,
                'type': 'offshore_wind'
            },
            'features': {
                'price_lags': [1, 24, 168],
                'demand_lags': [24, 168],
                'rolling_windows': [3, 24, 168]
            },
            'models': {
                'test_size': 0.2,
                'val_size': 0.1,
                'model_types': ['linear', 'lightgbm'],
                'quantiles': [0.1, 0.5, 0.9]
            },
            'monte_carlo': {
                'n_simulations': 1000,
                'price_volatility': 0.2,
                'generation_uncertainty': 0.1
            },
            'scenarios': {
                'ppa_price': 50,
                'cfd_strike_price': 55,
                'ppa_fraction': 0.5
            },
            'outputs': {
                'reports_path': 'reports',
                'figures_path': 'reports/figures',
                'models_path': 'models'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (use dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (use dot notation for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.info(f"Set {key} = {value}")
    
    def save(self, filepath: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save config. If None, uses current config_path.
        """
        if filepath is None:
            filepath = self.config_path
        else:
            filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved configuration to {filepath}")
    
    def __repr__(self) -> str:
        """String representation of config."""
        return yaml.dump(self.config, default_flow_style=False, sort_keys=False)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config object
    """
    return Config(config_path)


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration dictionary to file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save config
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved configuration to {filepath}")


# Global config instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance.
    
    Returns:
        Config object
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_global_config(config: Config):
    """
    Set global configuration instance.
    
    Args:
        config: Config object
    """
    global _global_config
    _global_config = config
