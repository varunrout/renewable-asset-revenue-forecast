# Wind & Solar Asset Revenue Forecasting

A comprehensive framework for modelling renewable energy asset revenue using wholesale electricity prices, generation data, and system demand. This project provides end-to-end tools for:

- **Data engineering** - Load, clean, and align market data
- **Feature engineering** - Create fundamental and ML features
- **Price modelling** - Train ML models to forecast electricity prices
- **Revenue calculation** - Backcast and forecast asset revenue
- **Scenario analysis** - Model different market and operational scenarios
- **Risk analysis** - Monte Carlo simulation for revenue uncertainty

---

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Requirements](#data-requirements)
- [Notebooks](#notebooks)
- [Modules](#modules)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)

---

## 📁 Project Structure

```
renewable-asset-revenue-forecast/
│
├── data_raw/                   # Raw input data (prices, generation, demand)
├── data_processed/             # Cleaned and feature-engineered datasets
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_revenue_backcast.ipynb
│   ├── 04_price_model.ipynb
│   └── 05_revenue_scenario_modelling.ipynb
│
├── src/                        # Source code modules
│   ├── data/                   # Data loading and cleaning
│   │   ├── load_data.py
│   │   └── clean_data.py
│   ├── features/               # Feature engineering
│   │   └── engineer_features.py
│   ├── models/                 # ML models and revenue calculators
│   │   ├── price_model.py
│   │   ├── revenue_model.py
│   │   └── monte_carlo.py
│   ├── scenarios/              # Scenario generation
│   │   ├── generation_scenarios.py
│   │   └── price_scenarios.py
│   └── utils/                  # Utilities
│       ├── plotting.py
│       └── config.py
│
├── configs/                    # Configuration files
│   └── modelling_config.yaml
│
├── reports/                    # Output reports and figures
│   ├── eda/
│   ├── figures/
│   └── final_report/
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/varunrout/renewable-asset-revenue-forecast.git
cd renewable-asset-revenue-forecast
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```python
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
print("✓ All dependencies installed successfully")
```

---

## ⚡ Quick Start

### Using Sample Data

```python
from src.data import load_sample_data
from src.features import feature_engineering_pipeline
from src.models import train_price_models, backcast_historical_revenue

# Load sample data
data = load_sample_data()

# Engineer features
df_features = feature_engineering_pipeline(data['prices'].join([
    data['offshore_wind'], data['onshore_wind'], 
    data['solar'], data['demand']
]))

# Backcast revenue
result = backcast_historical_revenue(
    df_features, 
    asset_capacity_mw=100, 
    asset_type='offshore_wind'
)

print(f"Annual Revenue: £{result['revenue'].sum():,.0f}")
```

### Using Your Own Data

1. Place your CSV files in `data_raw/`:
   - `prices.csv` - timestamp, price
   - `offshore_wind.csv` - timestamp, generation
   - `onshore_wind.csv` - timestamp, generation
   - `solar.csv` - timestamp, generation
   - `demand.csv` - timestamp, demand

2. Run the notebooks in sequence:
   - 01_data_cleaning.ipynb
   - 02_feature_engineering.ipynb
   - 03_baseline_revenue_backcast.ipynb
   - 04_price_model.ipynb
   - 05_revenue_scenario_modelling.ipynb

---

## 📊 Data Requirements

### Input Data Format

All datasets should have:
- **timestamp** column (datetime format)
- **value** column (numeric)

Example `prices.csv`:
```csv
timestamp,price
2023-01-01 00:00:00,45.20
2023-01-01 01:00:00,42.50
```

### Expected Columns

| Dataset | Columns | Units | Frequency |
|---------|---------|-------|-----------|
| Prices | timestamp, price | £/MWh | Hourly |
| Offshore Wind | timestamp, generation | MWh or MW | Hourly |
| Onshore Wind | timestamp, generation | MWh or MW | Hourly |
| Solar | timestamp, generation | MWh or MW | Hourly |
| Demand | timestamp, demand | MWh or MW | Hourly |

---

## 📓 Notebooks

### 01 - Data Cleaning
- Load raw data files
- Align timestamps across datasets
- Handle missing values
- Quality checks and validation

**Outputs**: `data_processed/clean_data.csv`

### 02 - Feature Engineering
- Create residual demand (demand - RES)
- Calculate RES shares
- Add time features (hour, day, season)
- Create lag and rolling features
- Price volatility features

**Outputs**: `data_processed/features.csv`

### 03 - Baseline Revenue Backcast
- Calculate historical revenue
- Analyze capacity factor and capture price
- Compare contract structures (merchant, PPA, CfD)
- Seasonal revenue patterns

**Outputs**: Revenue backcast results

### 04 - Price Model Training
- Train multiple ML models (Linear, LightGBM, etc.)
- Evaluate performance (MAE, RMSE, R²)
- Feature importance analysis
- Quantile regression for uncertainty

**Outputs**: Trained models, evaluation metrics

### 05 - Revenue Scenario Modelling
- Create price and generation scenarios
- Monte Carlo simulation
- Revenue distributions (P10/P50/P90)
- Risk metrics (VaR, CVaR)
- Sensitivity analysis

**Outputs**: Scenario results, risk analysis

---

## 🔧 Modules

### Data Module (`src/data/`)

**DataLoader**: Load raw data files
```python
from src.data import DataLoader

loader = DataLoader('data_raw')
data = loader.load_all_data()
```

**DataCleaner**: Clean and align data
```python
from src.data import DataCleaner, clean_pipeline

clean_data = clean_pipeline(raw_data, freq='h', missing_method='interpolate')
```

### Features Module (`src/features/`)

**FeatureEngineer**: Create features for modelling
```python
from src.features import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_all_features(clean_data)
```

### Models Module (`src/models/`)

**PriceModel**: Train price forecasting models
```python
from src.models import PriceModel

model = PriceModel(model_type='lightgbm')
model.train(X_train, y_train, X_val, y_val)
metrics = model.evaluate(X_test, y_test)
```

**RevenueCalculator**: Calculate revenue
```python
from src.models import RevenueCalculator

calculator = RevenueCalculator(asset_capacity_mw=100)
revenue = calculator.calculate_merchant_revenue(prices, generation)
```

**MonteCarloSimulator**: Simulate revenue uncertainty
```python
from src.models import MonteCarloSimulator

simulator = MonteCarloSimulator(n_simulations=1000)
results = simulator.simulate_revenue_distribution(prices, generation)
```

### Scenarios Module (`src/scenarios/`)

**GenerationScenarioBuilder**: Create generation scenarios
```python
from src.scenarios import create_generation_scenarios

scenarios = create_generation_scenarios(df, asset_type='offshore_wind')
```

**PriceScenarioBuilder**: Create price scenarios
```python
from src.scenarios import create_price_scenarios

scenarios = create_price_scenarios(df, base_scenario='historical')
```

### Utils Module (`src/utils/`)

Plotting functions and configuration management.

---

## ⚙️ Configuration

Edit `configs/modelling_config.yaml` to customize:

```yaml
# Asset configuration
asset:
  capacity_mw: 100
  type: offshore_wind  # offshore_wind, onshore_wind, solar

# Feature engineering
features:
  price_lags: [1, 24, 168]
  demand_lags: [24, 168]
  rolling_windows: [3, 24, 168]

# Model training
models:
  test_size: 0.2
  model_types: [linear, lightgbm]
  quantiles: [0.1, 0.5, 0.9]

# Monte Carlo simulation
monte_carlo:
  n_simulations: 1000
  price_volatility: 0.2
  generation_uncertainty: 0.1

# Scenarios
scenarios:
  ppa_price: 50
  cfd_strike_price: 55
```

---

## 💡 Usage Examples

### Example 1: Calculate Historical Revenue

```python
from src.data import DataLoader, clean_pipeline
from src.models import backcast_historical_revenue

# Load and clean data
loader = DataLoader('data_raw')
raw_data = loader.load_all_data()
clean_data = clean_pipeline(raw_data)

# Calculate revenue
result = backcast_historical_revenue(
    clean_data,
    asset_capacity_mw=100,
    asset_type='offshore_wind'
)

print(f"Total Revenue: £{result['revenue'].sum():,.0f}")
print(f"Capacity Factor: {result.attrs['metrics']['capacity_factor']:.2%}")
print(f"Capture Price: £{result.attrs['metrics']['capture_price']:.2f}/MWh")
```

### Example 2: Train Price Model

```python
from src.features import feature_engineering_pipeline
from src.models import train_price_models

# Engineer features
df_features = feature_engineering_pipeline(clean_data)

# Train models
results = train_price_models(
    df_features,
    target_col='price',
    model_types=['linear', 'lightgbm']
)

# Best model
best_model = results['models']['lightgbm']
print(f"RMSE: {results['metrics']['lightgbm']['rmse']:.2f}")
```

### Example 3: Monte Carlo Revenue Analysis

```python
from src.models import monte_carlo_revenue_analysis

# Run Monte Carlo simulation
mc_results = monte_carlo_revenue_analysis(
    df_features,
    asset_type='offshore_wind',
    n_simulations=1000,
    price_volatility=0.2,
    generation_uncertainty=0.1
)

print(f"P10 Revenue: £{mc_results['base_case']['p10']:,.0f}")
print(f"P50 Revenue: £{mc_results['base_case']['p50']:,.0f}")
print(f"P90 Revenue: £{mc_results['base_case']['p90']:,.0f}")
```

### Example 4: Compare Scenarios

```python
from src.models import RevenueScenario

scenario = RevenueScenario(asset_capacity_mw=100, asset_type='offshore_wind')

scenarios = {
    'merchant': {'scenario_type': 'merchant'},
    'ppa_50': {'scenario_type': 'ppa', 'ppa_price': 50},
    'cfd_55': {'scenario_type': 'cfd', 'strike_price': 55}
}

comparison = scenario.compare_scenarios(df_features, scenarios)
print(comparison)
```

---

## 📈 Key Features

### Fundamental Features
- **Residual Demand**: Demand minus renewable generation
- **RES Share**: Renewable penetration level
- **Technology Shares**: Individual generation technology shares

### Price Modelling
- **Multiple Algorithms**: Linear regression, LightGBM, Random Forest
- **Quantile Regression**: P10/P50/P90 price forecasts
- **Feature Importance**: Understand price drivers

### Revenue Modelling
- **Contract Types**: Merchant, PPA, CfD, Blended
- **Metrics**: Capacity factor, capture price, revenue per MW
- **Temporal Analysis**: Hourly, daily, monthly, seasonal

### Scenario Analysis
- **Price Scenarios**: High/low gas, demand changes, RES build-out
- **Generation Scenarios**: High/low wind, capacity factor variations
- **Combined Scenarios**: Optimistic, pessimistic, base case

### Risk Analysis
- **Monte Carlo Simulation**: Revenue distribution under uncertainty
- **Value at Risk (VaR)**: Downside risk quantification
- **Conditional VaR (CVaR)**: Expected shortfall
- **Sensitivity Analysis**: Impact of parameter changes

---

## 📚 Documentation

### Data Engineering Specification

**Timestamp Alignment**:
- All datasets aligned to common hourly frequency
- Missing timestamps filled with interpolation
- Maximum 3-hour gaps filled

**Missing Value Handling**:
- Linear interpolation (default)
- Forward/backward fill options
- Outlier detection and removal

**Unit Normalization**:
- All generation in MWh or MW
- Prices in £/MWh
- Automatic unit conversion

### Feature Engineering Specification

**Core Features**:
- `residual_demand = demand - (offshore + onshore + solar)`
- `res_share = total_RES / demand`
- `offshore_share`, `onshore_share`, `solar_share`

**Time Features**:
- Hour, day of week, month, season
- Weekend flag
- Cyclical encodings (sin/cos)

**Lag Features**:
- Price lags: 1h, 24h, 168h (1 week)
- Demand lags: 24h, 168h

**Rolling Features**:
- 3h, 24h, 168h windows
- Mean and standard deviation

### Model Card - Price Forecasting

**Objective**: Forecast wholesale electricity prices

**Models**:
- Linear Regression (baseline)
- LightGBM (primary)
- Random Forest (optional)
- Quantile Regression (uncertainty)

**Evaluation Metrics**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

**Train/Test Split**:
- Train: 70%
- Validation: 10%
- Test: 20%
- Temporal order preserved

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

## 🙏 Acknowledgments

Built for energy market analysts and renewable asset developers to model revenue under various market conditions and operational scenarios.