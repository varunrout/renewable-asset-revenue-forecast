"""
Price Modelling Module

Machine learning models for electricity price forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceModel:
    """Price forecasting model wrapper."""
    
    def __init__(self, model_type: str = 'lightgbm'):
        """
        Initialize price model.
        
        Args:
            model_type: Type of model ('linear', 'lightgbm', 'xgboost', 'rf', 'gbm')
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        
    def _create_model(self, params: Optional[Dict] = None):
        """Create the underlying model."""
        if self.model_type == 'linear':
            self.model = LinearRegression()
        
        elif self.model_type == 'lightgbm':
            default_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            if params:
                default_params.update(params)
            self.model = lgb.LGBMRegressor(**default_params)
        
        elif self.model_type == 'rf':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
            if params:
                default_params.update(params)
            self.model = RandomForestRegressor(**default_params)
        
        elif self.model_type == 'gbm':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
            if params:
                default_params.update(params)
            self.model = GradientBoostingRegressor(**default_params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[Dict] = None
    ):
        """
        Train the price model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Model parameters
        """
        logger.info(f"Training {self.model_type} model")
        
        self._create_model(params)
        
        # Train model
        if self.model_type == 'lightgbm' and X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        logger.info("Training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        self.metrics = metrics
        
        logger.info(f"MAE: {metrics['mae']:.2f}")
        logger.info(f"RMSE: {metrics['rmse']:.2f}")
        logger.info(f"R²: {metrics['r2']:.4f}")
        logger.info(f"MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def save(self, filepath: str):
        """Save model to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'PriceModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model


class QuantilePriceModel:
    """Quantile regression model for price uncertainty estimation."""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        """
        Initialize quantile price model.
        
        Args:
            quantiles: List of quantiles to predict
        """
        self.quantiles = quantiles
        self.models = {}
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        Train quantile models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        logger.info(f"Training quantile models for {self.quantiles}")
        
        for q in self.quantiles:
            logger.info(f"Training quantile {q}")
            
            # LightGBM supports quantile regression
            params = {
                'objective': 'quantile',
                'alpha': q,
                'metric': 'quantile',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            else:
                model.fit(X_train, y_train)
            
            self.models[q] = model
        
        logger.info("Quantile training complete")
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict quantiles.
        
        Args:
            X: Features
            
        Returns:
            DataFrame with predictions for each quantile
        """
        predictions = {}
        
        for q, model in self.models.items():
            predictions[f'p{int(q*100)}'] = model.predict(X)
        
        return pd.DataFrame(predictions, index=X.index)
    
    def save(self, filepath: str):
        """Save models to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Quantile models saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'QuantilePriceModel':
        """Load models from file."""
        with open(filepath, 'rb') as f:
            models = pickle.load(f)
        logger.info(f"Quantile models loaded from {filepath}")
        return models


def train_price_models(
    df: pd.DataFrame,
    target_col: str = 'price',
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    model_types: List[str] = ['linear', 'lightgbm'],
    quantiles: List[float] = [0.1, 0.5, 0.9]
) -> Dict[str, Any]:
    """
    Train multiple price models.
    
    Args:
        df: DataFrame with features and target
        target_col: Target column name
        feature_cols: List of feature columns (if None, use all except target)
        test_size: Test set size
        val_size: Validation set size
        model_types: List of model types to train
        quantiles: Quantiles for quantile regression
        
    Returns:
        Dictionary with trained models and metrics
    """
    logger.info("Starting price model training pipeline")
    
    # Prepare data
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    
    # Remove any rows with NaN in features or target
    df_clean = df[feature_cols + [target_col]].dropna()
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Split data: train, validation, test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), shuffle=False
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train models
    results = {
        'models': {},
        'metrics': {},
        'feature_importance': {},
        'data_splits': {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    }
    
    # Train point forecast models
    for model_type in model_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type} model")
        logger.info('='*50)
        
        model = PriceModel(model_type=model_type)
        model.train(X_train, y_train, X_val, y_val)
        
        metrics = model.evaluate(X_test, y_test)
        
        results['models'][model_type] = model
        results['metrics'][model_type] = metrics
        
        if model.feature_importance is not None:
            results['feature_importance'][model_type] = model.feature_importance
    
    # Train quantile model
    logger.info(f"\n{'='*50}")
    logger.info("Training quantile regression model")
    logger.info('='*50)
    
    quantile_model = QuantilePriceModel(quantiles=quantiles)
    quantile_model.train(X_train, y_train, X_val, y_val)
    
    results['models']['quantile'] = quantile_model
    
    logger.info("\nModel training pipeline complete")
    
    return results
