# 🚀 Financial ML Pipeline

A comprehensive, modular machine learning pipeline for financial trading strategies, featuring advanced data processing, sophisticated labeling methods, and state-of-the-art ML techniques.

## ✨ Features

### 📊 Data Processing
- **Dollar Bar Sampling**: Creates price bars based on dollar volume instead of time
- **Fractional Differentiation**: Preserves maximum information while achieving stationarity
- **Technical Indicators**: 100+ technical analysis features using the `ta` library
- **Wavelet Transforms**: Decomposes price series into frequency components

### 🏷️ Advanced Labeling  
- **Triple-Barrier Method**: Meta-labeling approach with profit-taking, stop-loss, and time barriers
- **Volatility Breakout Events**: Intelligent event sampling based on volatility regimes
- **Sample Uniqueness Weighting**: Addresses sample overlap and redundancy

### 🤖 Machine Learning
- **LightGBM**: Gradient boosting with automatic hyperparameter optimization
- **Walk-Forward Cross-Validation**: Time-series aware validation with embargo
- **Meta-Labeling**: Secondary model to filter primary model signals for improved precision

### 📈 Evaluation & Analysis
- **Out-of-Sample Testing**: Rigorous backtesting methodology
- **Confusion Matrices**: Detailed performance visualization  
- **Precision Analysis**: Performance across different confidence thresholds

## 🏗️ Architecture

```
financial-ml-pipeline/
├── data/                   # Data loading and preprocessing
│   ├── __init__.py
│   └── loaders.py         # Dollar bars, fractional diff, mock data
├── features/              # Feature engineering
│   ├── __init__.py
│   └── engineering.py     # Technical indicators, wavelets, lags
├── labeling/              # Event sampling and labeling
│   ├── __init__.py
│   └── triple_barrier.py  # Triple-barrier, sample weights
├── modeling/              # ML training and optimization
│   ├── __init__.py
│   ├── optimizer.py       # Optuna hyperparameter optimization
│   └── trainer.py         # Model training, evaluation, meta-labeling
├── utils/                 # Utilities
│   ├── __init__.py
│   ├── logging.py         # Colored logging functions
│   └── config.py          # YAML configuration management
├── configs/               # Configuration files
│   └── default.yaml       # Default pipeline configuration
├── main.py               # CLI entry point with typer
├── pipeline.py           # Main pipeline orchestrator class
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/nguyenduchuyiu/quant-trading
cd quant-trading

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Run the full pipeline with default settings
python main.py run

# Run with custom configuration
python main.py run --config configs/my_config.yaml

# Run only specific stages
python main.py run --stage features   # Run up to feature engineering
python main.py run --stage labeling   # Run up to labeling
python main.py run --stage optimize   # Run up to optimization
python main.py run --stage evaluate   # Run up to model evaluation
python main.py run --stage meta       # Run up to meta-labeling with backtesting
python main.py run --stage backtest   # Run full dataset backtesting
```

### 3. Configuration

Generate a configuration template:

```bash
python main.py config-template --output my_config.yaml
```

Example configuration:

```yaml
# Data settings
data: "data/your_data.csv"
seed: 42
threshold: null  # Auto-calculate dollar bar threshold

# Labeling parameters  
pt: 2.0    # Profit taking multiplier
sl: 2.0    # Stop loss multiplier
vertical: 12  # Vertical barrier (number of bars)

# Model optimization
n_trials: 30   # Optuna optimization trials
n_splits: 4    # Cross-validation splits
train_size: 0.8  # Training data fraction

# Backtesting parameters
initial_capital: 100000.0    # Starting capital ($)
transaction_cost_pct: 0.001  # Transaction cost (0.1%)

# Hyperparameters (null = use Optuna)
hyperparameters:
  n_estimators: 500
  learning_rate: 0.1
  # ... other LightGBM parameters
```

## 📋 Data Format

Your CSV data should have these columns:
- `date`: Timestamp (will be parsed automatically)
- `open`: Opening price
- `high`: Highest price  
- `close`: Closing price
- `low`: Lowest price
- `volume`: Trading volume

## 🎯 Pipeline Stages

### Stage 1: Data Processing & Feature Engineering
1. Load raw OHLCV data
2. Create dollar bars from tick data
3. Generate 100+ technical indicators
4. Apply fractional differentiation
5. Create wavelet decompositions
6. Add lagged features

### Stage 2: Event Sampling & Labeling
1. Identify volatility breakout events
2. Apply triple-barrier labeling method
3. Calculate sample uniqueness weights
4. Create final labeled dataset

### Stage 3: Hyperparameter Optimization
1. Walk-forward cross-validation setup
2. Optuna-based parameter search
3. Model performance evaluation
4. Best parameter selection

### Stage 4: Out-of-Sample Testing
1. Train final model on training set
2. Evaluate on test set
3. Generate confusion matrices
4. Performance reporting

### Stage 5: Meta-Labeling
1. Generate out-of-sample primary predictions
2. Train meta-model for signal filtering
3. Evaluate combined strategy
4. Threshold analysis for precision optimization

### Stage 6: Backtesting & Performance Analysis
1. Generate trading signals from trained models
2. Apply meta-model filtering with confidence thresholds
3. Calculate strategy returns with transaction costs
4. Compute performance metrics (Sharpe ratio, drawdown, win rate)
5. Create equity curves and drawdown charts
6. Compare against buy-and-hold benchmark

## 🛠️ Advanced Usage

### Using as a Python Library

```python
from pipeline import FinancialMLPipeline
from utils.config import load_config

# Load configuration
config = load_config("configs/my_config.yaml")

# Initialize pipeline
pipeline = FinancialMLPipeline(config)

# Run individual stages
pipeline.load_data()
pipeline.run_feature_engineering()
pipeline.run_labeling()
pipeline.optimize_model()
pipeline.evaluate_model()
pipeline.run_meta_labeling()

# Get results
results = pipeline.get_results()
print(results)
```

### Custom Feature Engineering

```python
from features.engineering import add_technical_indicators, apply_wavelet_transform

# Add your custom features
def add_custom_features(df):
    # Your custom feature engineering logic
    df['custom_feature'] = df['close'].rolling(20).mean()
    return df

# Integrate with pipeline
features_df = add_technical_indicators(dollar_bars)
features_df = add_custom_features(features_df)
```

## 📊 Example Results

```
Pipeline Results Summary
  - data_shape: (50000, 5)
  - features_shape: (1247, 156)  
  - final_dataset_shape: (623, 159)
  - best_params: {'n_estimators': 742, 'learning_rate': 0.087, ...}
  - primary_model_trained: True
  - meta_labeling_completed: True

Out-of-Sample Performance:
              precision    recall  f1-score   support
   Sell (-1)       0.52      0.48      0.50        85
   Hold (0)        0.71      0.76      0.73       186  
   Buy (1)         0.55      0.51      0.53        92
```

## 🧪 Testing

Run basic functionality tests:

```bash
python -m pytest tests/ -v
```

## 🔧 Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data` | Path to CSV data file | `"data/sample_data.csv"` |
| `seed` | Random seed for reproducibility | `42` |
| `threshold` | Dollar bar threshold | `null` (auto-calculate) |
| `pt` | Profit taking multiplier | `2.0` |
| `sl` | Stop loss multiplier | `2.0` |
| `vertical` | Vertical barrier (number of bars) | `12` |
| `n_trials` | Optuna optimization trials | `30` |
| `n_splits` | Cross-validation splits | `4` |
| `train_size` | Training data fraction | `0.8` |
| `initial_capital` | Starting capital ($) | `100000.0` |
| `transaction_cost_pct` | Transaction cost (%) | `0.001` |
| `hyperparameters` | LightGBM hyperparameters (null = use Optuna) | `{'n_estimators': 500, 'learning_rate': 0.1}` |

## 📝 Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) before submitting a pull request.

## 📜 License

This project is licensed under the [MIT License](LICENSE).