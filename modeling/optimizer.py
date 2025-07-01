#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hyperparameter optimization utilities for the financial ML pipeline.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
import optuna
from utils.logging import log_info, log_section, log_subsection


def run_hyperparameter_optimization(X, y, weights, t1, n_trials=30, hyper_config=None, n_splits=4) -> dict:
    """
    Finds the best hyperparameters for the primary model using Optuna and Walk-Forward CV.
    
    Args:
        X: Feature matrix
        y: Target labels
        weights: Sample weights
        t1: End times for embargo
        n_trials: Number of Optuna trials
        hyper_config: Pre-configured hyperparameters
        n_splits: Number of CV splits
        
    Returns:
        Dictionary with best hyperparameters
    """
    log_section("Stage 3: Hyperparameter Optimization")

    hyper_keys = [
        'n_estimators', 'learning_rate', 'num_leaves', 'max_depth', 'lambda_l1', 'lambda_l2',
        'feature_fraction', 'bagging_fraction', 'bagging_freq', 'min_child_samples'
    ]
    
    # Check if all hyperparameters are pre-configured
    if hyper_config and all(hyper_config.get(k) is not None for k in hyper_keys):
        log_info("All hyperparameters are configured from YAML. Skipping Optuna optimization.")
        best_params = {k: hyper_config[k] for k in hyper_keys}
        return best_params

    def get_param_from_config_or_trial(param_name, trial, suggest_func, *args, **kwargs):
        """Get parameter from config or trial suggestion."""
        if hyper_config and param_name in hyper_config and hyper_config[param_name] is not None:
            return hyper_config[param_name]
        return getattr(trial, suggest_func)(param_name, *args, **kwargs)

    def objective(trial):
        """Optuna objective function."""
        params = {
            'objective': 'multiclass', 
            'metric': 'multi_logloss', 
            'num_class': 3,
            'verbosity': -1, 
            'boosting_type': 'gbdt',
            'n_estimators': get_param_from_config_or_trial('n_estimators', trial, 'suggest_int', 100, 1000),
            'learning_rate': get_param_from_config_or_trial('learning_rate', trial, 'suggest_float', 0.01, 0.3),
            'num_leaves': get_param_from_config_or_trial('num_leaves', trial, 'suggest_int', 20, 300),
            'max_depth': get_param_from_config_or_trial('max_depth', trial, 'suggest_int', 3, 12),
            'lambda_l1': get_param_from_config_or_trial('lambda_l1', trial, 'suggest_float', 1e-8, 10.0, log=True),
            'lambda_l2': get_param_from_config_or_trial('lambda_l2', trial, 'suggest_float', 1e-8, 10.0, log=True),
            'feature_fraction': get_param_from_config_or_trial('feature_fraction', trial, 'suggest_float', 0.4, 1.0),
            'bagging_fraction': get_param_from_config_or_trial('bagging_fraction', trial, 'suggest_float', 0.4, 1.0),
            'bagging_freq': get_param_from_config_or_trial('bagging_freq', trial, 'suggest_int', 1, 7),
            'min_child_samples': get_param_from_config_or_trial('min_child_samples', trial, 'suggest_int', 5, 100),
        }
        
        # Walk-forward cross-validation
        f1_scores = []
        splits = np.array_split(np.arange(len(X)), n_splits + 1)
        
        for i in range(n_splits):
            train_indices = np.concatenate(splits[:i+1])
            test_indices = splits[i+1]
            
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            weights_train = weights.iloc[train_indices]
            
            # Apply embargo
            last_train_t1 = t1.iloc[train_indices].max()
            embargo_mask = t1.index[test_indices] > last_train_t1
            X_test, y_test = X_test.loc[embargo_mask], y_test.loc[embargo_mask]
            
            if X_train.empty or X_test.empty:
                continue
            
            # Train and evaluate model
            model = lgb.LGBMClassifier(**params, class_weight='balanced')
            model.fit(X_train, y_train, sample_weight=weights_train)
            y_pred = model.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        return np.mean(f1_scores) if f1_scores else 0.0

    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    
    log_info("-> Optimization complete.\n")
    log_subsection("Best Hyperparameters Found")
    best_params = study.best_trial.params
    log_info(f"  - Best Weighted F1-score: {study.best_trial.value:.5f}")
    for key, value in best_params.items():
        log_info(f"    - {key}: {value}")
    
    return best_params 