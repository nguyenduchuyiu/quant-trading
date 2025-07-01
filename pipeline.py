#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Financial ML Pipeline Class - Main orchestrator for the entire pipeline.
"""
import pandas as pd
import numpy as np
from types import SimpleNamespace

# Import pipeline components
from data.loaders import load_raw_data
from features.engineering import run_feature_engineering_pipeline
from labeling.triple_barrier import run_labeling_pipeline
from modeling.optimizer import run_hyperparameter_optimization
from modeling.trainer import run_out_of_sample_test, run_meta_labeling_pipeline, generate_signals_from_models
from modeling.backtest import run_event_driven_backtest, run_signal_analysis
from utils.logging import log_info, log_subsection, log_warning, log_error, log_section
from utils.config import load_config, get_hyperparameter_config


class FinancialMLPipeline:
    """
    Main pipeline class for financial machine learning workflow.
    """
    
    def __init__(self, config: SimpleNamespace):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration object with pipeline parameters
        """
        self.config = config
        self.raw_df = None
        self.features = None
        self.dollar_bars = None
        self.final_dataset = None
        self.best_params = None
        self.primary_model = None
        self.meta_model = None
        self.meta_results = None
        self.backtest_results = None
        self.training_features = None  # Store training feature names
        
        # Set random seed
        np.random.seed(config.seed)
        
    def load_data(self) -> None:
        """Load raw data from file or create mock data."""
        log_section("Loading Data")
        self.raw_df = load_raw_data(self.config.data)
        log_info(f"Data loaded with shape: {self.raw_df.shape}")
        
    def run_feature_engineering(self) -> None:
        """Execute feature engineering pipeline."""
        if self.raw_df is None:
            log_error("No data loaded. Please run load_data() first.")
            return
            
        self.features, self.dollar_bars = run_feature_engineering_pipeline(
            self.raw_df, config=self.config
        )
        
        if self.features.empty:
            log_error("Feature engineering failed.")
            return
            
        log_info(f"Feature engineering completed. Features shape: {self.features.shape}")
        
    def run_labeling(self) -> None:
        """Execute labeling pipeline."""
        if self.features is None or self.dollar_bars is None:
            log_error("Features not available. Please run feature engineering first.")
            return
            
        pt_sl = [self.config.pt, self.config.sl]
        self.final_dataset = run_labeling_pipeline(
            self.features, self.dollar_bars, pt_sl=pt_sl, num_bars=self.config.vertical
        )
        
        if self.final_dataset.empty:
            log_error("Labeling failed.")
            return
            
        log_info(f"Labeling completed. Final dataset shape: {self.final_dataset.shape}")
        
    def optimize_model(self) -> None:
        """Run hyperparameter optimization."""
        if self.final_dataset is None:
            log_error("Final dataset not available. Please run labeling first.")
            return
            
        # Prepare data for modeling
        datetime_cols = self.final_dataset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
        feature_to_drop = ['label', 'weight', 'open', 'high', 'low', 'close', 'volume']
        feature_to_drop += datetime_cols
        feature_cols = [col for col in self.final_dataset.columns if col not in feature_to_drop]
        self.training_features = feature_cols
        
        X = self.final_dataset[self.training_features]
        y = self.final_dataset['label']
        weights = self.final_dataset['weight']
        t1 = self.final_dataset['t1']
        
        # Clean weights
        weights.replace([np.inf, -np.inf], np.nan, inplace=True)
        if not weights.dropna().empty:
            weights.fillna(weights.dropna().max(), inplace=True)
        else:
            weights.fillna(1.0, inplace=True)
        
        # Get hyperparameter config
        hyper_config = get_hyperparameter_config(self.config)
        
        # Run optimization
        self.best_params = run_hyperparameter_optimization(
            X, y, weights, t1, 
            n_trials=self.config.n_trials, 
            hyper_config=hyper_config, 
            n_splits=getattr(self.config, 'n_splits', 4)
        )
        
        log_info("Hyperparameter optimization completed.")
        
    def evaluate_model(self) -> None:
        """Run out-of-sample testing."""
        if self.best_params is None:
            log_error("No optimized parameters available. Please run optimization first.")
            return
            
        # Prepare data       
        X = self.final_dataset[self.training_features]
        y = self.final_dataset['label']
        weights = self.final_dataset['weight']
        
        # Clean weights
        weights.replace([np.inf, -np.inf], np.nan, inplace=True)
        if not weights.dropna().empty:
            weights.fillna(weights.dropna().max(), inplace=True)
        else:
            weights.fillna(1.0, inplace=True)
        
        # Train and evaluate primary model
        self.primary_model = run_out_of_sample_test(
            X, y, weights, self.best_params, 
            train_size=getattr(self.config, 'train_size', 0.8)
        )
        
        log_info("Model evaluation completed.")
        
    def run_meta_labeling(self) -> None:
        """Run meta-labeling pipeline without automatic backtesting."""
        if self.primary_model is None:
            log_error("Primary model not available. Please run model evaluation first.")
            return
        
        # Get backtest config from pipeline config
        initial_capital = getattr(self.config, 'initial_capital', 100000.0)
        transaction_cost = getattr(self.config, 'transaction_cost_pct', 0.001)
        
        # Store training features for later use            
        self.meta_results = run_meta_labeling_pipeline(
            self.final_dataset, 
            self.training_features,
            self.best_params, 
            self.primary_model, 
            train_size=getattr(self.config, 'train_size', 0.8),
        )
        
        log_info("Meta-labeling completed.")
        
    def run_backtest(self, dataset_range='test', signals=None) -> None:
        """
        Run backtesting on signals.
        
        Args:
            dataset_range: 'test' for test set only, 'full' for entire dataset
            signals: Optional pre-computed signals. If None, will generate from models
        """
        if self.primary_model is None or self.final_dataset is None or self.final_dataset.empty:
            log_error("Primary model and dataset required for backtesting.")
            return
        
        # Get backtest config
        initial_capital = getattr(self.config, 'initial_capital', 100000.0)
        transaction_cost = getattr(self.config, 'transaction_cost_pct', 0.001)
        
        if signals is None:
            # Generate signals based on dataset range
            if dataset_range == 'test':
                # Use signals from meta-labeling results
                if self.meta_results is None or 'final_signals' not in self.meta_results:
                    log_error("Meta-labeling results not available for test set backtest.")
                    return
                
                signals = self.meta_results['final_signals']
                log_returns = self.final_dataset['log_returns'].reindex(signals.index)
                
            elif dataset_range == 'full':
                # Generate signals for full dataset with proper feature alignment
                meta_model = None
                best_threshold = 0.5
                
                if self.meta_results is not None and self.meta_results.get('meta_model_trained'):
                    meta_model = self.meta_results.get('meta_model')
                    best_threshold = self.meta_results.get('best_threshold', 0.5)
                
                signals = generate_signals_from_models(
                    self.final_dataset, 
                    self.primary_model, 
                    meta_model, 
                    best_threshold,
                    training_features=self.training_features
                )
                log_returns = self.final_dataset['log_returns']
            else:
                log_error(f"Unknown dataset_range: {dataset_range}")
                return
        else:
            # Use provided signals - assume full dataset
            log_returns = self.final_dataset['log_returns'].reindex(signals.index)
        
        # Run signal analysis
        signal_analysis = run_signal_analysis(signals)
        
        # Get full returns for buy-and-hold comparison
        full_returns = None
        if self.dollar_bars is not None and 'log_returns' in self.dollar_bars.columns:
            full_returns = self.dollar_bars['log_returns']
        
        initial_capital = getattr(self.config, 'initial_capital')
        transaction_cost = getattr(self.config, 'transaction_cost_pct')
        pt_sl_multipliers = [getattr(self.config, 'pt'), getattr(self.config, 'sl')]
        risk_fraction = getattr(self.config, 'risk_fraction')
        long_only = getattr(self.config, 'long_only')
    
        # Run backtest
        backtest_results = run_event_driven_backtest(
            signals=signals,
            full_df=self.final_dataset,
            pt_sl_multipliers=pt_sl_multipliers,
            initial_capital=initial_capital,
            risk_fraction=risk_fraction,
            transaction_cost_pct=transaction_cost,
            long_only=long_only,
            full_returns=full_returns  # Pass full returns for proper buy-and-hold
        )
        
        # Store results
        self.backtest_results = {
            'dataset_range': dataset_range,
            'signals': signals,
            'signal_analysis': signal_analysis,
            'backtest_results': backtest_results
        }
        
        log_info("Backtesting completed.")
        
    def run_full_pipeline(self) -> None:
        """Execute the complete pipeline including backtesting."""
        log_section("Starting Full Financial ML Pipeline")
        
        # Execute each stage
        self.load_data()
        if self.raw_df is not None:
            self.run_feature_engineering()
            
        if self.features is not None and not self.features.empty:
            self.run_labeling()
            
        if self.final_dataset is not None and not self.final_dataset.empty:
            log_info("\nPIPELINE STAGES 1 & 2 COMPLETED SUCCESSFULLY.")
            self.optimize_model()
            
        if self.best_params is not None:
            self.evaluate_model()
            
        if self.primary_model is not None:
            self.run_meta_labeling()
            
        # Run backtesting separately to avoid duplication
        if self.meta_results is not None:
            log_section("Stage 6: Vectorized Backtest & Performance Analysis")
            log_subsection("Backtest on test set")
            self.run_backtest(dataset_range='test')
            log_subsection("Backtest on full dataset")
            self.run_backtest(dataset_range='full')
            
        log_section("Pipeline Execution Complete")
        
    def get_results(self) -> dict:
        """
        Get pipeline results summary.
        
        Returns:
            Dictionary with pipeline results
        """
        results = {
            'data_shape': self.raw_df.shape if self.raw_df is not None else None,
            'features_shape': self.features.shape if self.features is not None else None,
            'final_dataset_shape': self.final_dataset.shape if self.final_dataset is not None else None,
            'best_params': self.best_params,
            'primary_model_trained': self.primary_model is not None,
            'meta_labeling_completed': self.meta_results is not None,
        }
        
        # Add backtest results if available
        if self.backtest_results is not None and 'backtest_results' in self.backtest_results:
            try:
                backtest_summary = {
                    'dataset_range': self.backtest_results['dataset_range'],
                    'total_return': self.backtest_results['backtest_results']['total_return'],
                    'sharpe_ratio': self.backtest_results['backtest_results']['sharpe_ratio'],
                    'max_drawdown': self.backtest_results['backtest_results']['max_drawdown'],
                    'num_trades': self.backtest_results['backtest_results']['num_trades']
                }
                results['backtest_summary'] = backtest_summary
            except (KeyError, TypeError) as e:
                log_warning(f"Could not extract backtest summary: {e}")
        
        return results 