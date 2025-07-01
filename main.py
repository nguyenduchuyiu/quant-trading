#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for the Financial ML Pipeline with CLI support.
"""
import typer
from pathlib import Path
from pipeline import FinancialMLPipeline
from utils.config import load_config
from utils.logging import log_info, log_error, log_section, log_subsection

app = typer.Typer(
    name="financial-ml-pipeline",
    help="A comprehensive financial machine learning pipeline for trading strategies.",
    add_completion=False
)


@app.command()
def run(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to the YAML configuration file"
    ),
    stage: str = typer.Option(
        "full",
        "--stage",
        "-s",
        help="Pipeline stage to run: 'full', 'data', 'features', 'labeling', 'optimize', 'evaluate', 'meta', 'backtest'"
    )
):
    """
    Run the financial ML pipeline with specified configuration.
    
    Stages:
    - full: Run the complete pipeline including backtesting
    - data: Load data only
    - features: Run up to feature engineering
    - labeling: Run up to labeling
    - optimize: Run up to hyperparameter optimization
    - evaluate: Run up to model evaluation
    - meta: Run up to meta-labeling with backtesting
    - backtest: Run backtesting on full dataset (requires completed meta stage)
    """
    try:
        # Load configuration
        config_obj = load_config(config)
        log_info(f"Loaded configuration from: {config}")
        
        # Initialize pipeline
        pipeline = FinancialMLPipeline(config_obj)
        
        # Execute based on stage
        if stage == "full":
            pipeline.run_full_pipeline()
        elif stage == "data":
            pipeline.load_data()
        elif stage == "features":
            pipeline.load_data()
            if pipeline.raw_df is not None:
                pipeline.run_feature_engineering()
        elif stage == "labeling":
            pipeline.load_data()
            if pipeline.raw_df is not None:
                pipeline.run_feature_engineering()
            if pipeline.features is not None and not pipeline.features.empty:
                pipeline.run_labeling()
        elif stage == "optimize":
            pipeline.load_data()
            if pipeline.raw_df is not None:
                pipeline.run_feature_engineering()
            if pipeline.features is not None and not pipeline.features.empty:
                pipeline.run_labeling()
            if pipeline.final_dataset is not None and not pipeline.final_dataset.empty:
                pipeline.optimize_model()
        elif stage == "evaluate":
            pipeline.load_data()
            if pipeline.raw_df is not None:
                pipeline.run_feature_engineering()
            if pipeline.features is not None and not pipeline.features.empty:
                pipeline.run_labeling()
            if pipeline.final_dataset is not None and not pipeline.final_dataset.empty:
                pipeline.optimize_model()
            if pipeline.best_params is not None:
                pipeline.evaluate_model()
        elif stage == "meta":
            pipeline.load_data()
            if pipeline.raw_df is not None:
                pipeline.run_feature_engineering()
            if pipeline.features is not None and not pipeline.features.empty:
                pipeline.run_labeling()
            if pipeline.final_dataset is not None and not pipeline.final_dataset.empty:
                pipeline.optimize_model()
            if pipeline.best_params is not None:
                pipeline.evaluate_model()
            if pipeline.primary_model is not None:
                pipeline.run_meta_labeling()
        elif stage == "backtest":
            # Run all stages up to meta-labeling, then run full dataset backtest
            pipeline.load_data()
            if pipeline.raw_df is not None:
                pipeline.run_feature_engineering()
            if pipeline.features is not None and not pipeline.features.empty:
                pipeline.run_labeling()
            if pipeline.final_dataset is not None and not pipeline.final_dataset.empty:
                pipeline.optimize_model()
            if pipeline.best_params is not None:
                pipeline.evaluate_model()
            if pipeline.primary_model is not None:
                pipeline.run_meta_labeling()
            if pipeline.meta_results is not None:
                log_section("Stage 6: Vectorized Backtest & Performance Analysis")
                log_subsection("Backtest on full dataset")
                pipeline.run_backtest(dataset_range='full')
        else:
            log_error(f"Unknown stage: {stage}")
            raise typer.Exit(1)
        
        # Display results summary
        results = pipeline.get_results()
        log_section("Pipeline Results Summary")
        for key, value in results.items():
            log_info(f"  - {key}: {value}")
            
    except Exception as e:
        log_error(f"Pipeline execution failed: {str(e)}")
        raise typer.Exit(1)


@app.command()
def config_template(
    output: str = typer.Option(
        "configs/my_config.yaml",
        "--output",
        "-o",
        help="Output path for the configuration template"
    )
):
    """
    Generate a configuration template file.
    """
    template = """# Financial ML Pipeline Configuration Template

# Data settings
data: "path/to/your/data.csv"  # CSV file with columns: date, open, high, low, close, volume
seed: 42
threshold: null  # Dollar bar threshold (auto-calculate if null)

# Labeling parameters
pt: 2.0    # Profit taking multiplier
sl: 2.0    # Stop loss multiplier
vertical: 12  # Number of bars for vertical barrier

# Model optimization
n_trials: 30   # Number of Optuna trials
n_splits: 4    # Number of CV splits
train_size: 0.8  # Fraction of data for training

# Backtesting parameters
initial_capital: 100000.0  # Starting capital for backtesting
transaction_cost_pct: 0.001  # Transaction cost as percentage (0.1%)

# Hyperparameters (set to null to use Optuna optimization)
hyperparameters:
  n_estimators: null
  learning_rate: null
  num_leaves: null
  max_depth: null
  lambda_l1: null
  lambda_l2: null
  feature_fraction: null
  bagging_fraction: null
  bagging_freq: null
  min_child_samples: null

# Advanced settings
# vol_short_window: 12  # Short-term volatility window
# vol_long_window: 48   # Long-term volatility window
# ffd_diff: 0.4         # Fractional differentiation parameter
"""
    
    # Create output directory if it doesn't exist
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write template
    with open(output_path, 'w') as f:
        f.write(template)
    
    log_info(f"Configuration template created at: {output}")


@app.command()
def info():
    """
    Display information about the pipeline.
    """
    info_text = """
🚀 Financial ML Pipeline

A comprehensive machine learning pipeline for financial trading strategies, featuring:

📊 Data Processing:
  - Dollar bar sampling
  - Fractional differentiation
  - Technical indicators (100+ features)
  - Wavelet transforms

🏷️  Advanced Labeling:
  - Triple-barrier method
  - Volatility breakout event sampling
  - Sample uniqueness weighting

🤖 Machine Learning:
  - LightGBM with hyperparameter optimization
  - Walk-forward cross-validation
  - Meta-labeling for precision improvement

📈 Evaluation & Backtesting:
  - Out-of-sample testing
  - Confusion matrices
  - Precision analysis across confidence thresholds
  - Vectorized backtesting with performance metrics
  - Full dataset backtesting

Usage Examples:
  # Run full pipeline with default config
  python main.py run

  # Run with custom config
  python main.py run --config configs/my_config.yaml

  # Run only up to feature engineering
  python main.py run --stage features

  # Run up to meta-labeling with backtesting
  python main.py run --stage meta

  # Run full dataset backtest
  python main.py run --stage backtest

  # Generate config template
  python main.py config-template --output my_config.yaml
"""
    
    print(info_text)


if __name__ == "__main__":
    app() 