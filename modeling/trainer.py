#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model training and evaluation utilities for the financial ML pipeline.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils.logging import log_info, log_warning, log_section, log_subsection


def run_out_of_sample_test(X, y, weights, best_params, train_size=0.8) -> lgb.LGBMClassifier:
    """
    Trains a model on the first train_size of data, evaluates on the rest, and returns the trained model.
    
    Args:
        X: Feature matrix
        y: Target labels  
        weights: Sample weights
        best_params: Best hyperparameters from optimization
        train_size: Fraction of data for training
        
    Returns:
        Trained LGBMClassifier model
    """
    log_section("Stage 4: Out-of-Sample Testing")
    
    train_size_count = int(len(X) * train_size)
    X_train, X_test = X.iloc[:train_size_count], X.iloc[train_size_count:]
    y_train, y_test = y.iloc[:train_size_count], y.iloc[train_size_count:]
    weights_train = weights.iloc[:train_size_count]
    
    log_info(f"-> Training Set Size: {len(X_train)}, Test Set Size: {len(X_test)}\n")
    
    # Prepare model parameters
    model_params = best_params.copy()
    model_params.update({
        'objective': 'multiclass', 
        'num_class': 3, 
        'class_weight': 'balanced', 
        'verbosity': -1
    })
    
    # Train primary model
    primary_model = lgb.LGBMClassifier(**model_params)
    primary_model.fit(X_train, y_train, sample_weight=weights_train)
    log_info("-> Final primary model trained on train set.\n")
    
    # Evaluate on test set
    y_pred_test = primary_model.predict(X_test)
    log_subsection("Out-of-Sample Performance (Primary Model)")
    print(classification_report(
        y_test, y_pred_test, 
        labels=[-1, 0, 1], 
        target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)'], 
        zero_division=0
    ))
    print("="*60)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_test, labels=[-1, 0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Pred Sell', 'Pred Hold', 'Pred Buy'],
        yticklabels=['True Sell', 'True Hold', 'True Buy']
    )
    plt.title('Out-of-Sample Confusion Matrix (Primary Model)')
    plt.show()
    
    return primary_model


def run_meta_labeling_pipeline(final_dataset, best_params, primary_model_oos, train_size=0.8, 
                                initial_capital=100000.0, 
                              transaction_cost_pct=0.001) -> dict:
    """
    Builds and evaluates a meta-model to filter primary model signals, with optional backtesting.
    
    Args:
        final_dataset: Complete dataset with features and labels
        best_params: Best hyperparameters
        primary_model_oos: Pre-trained primary model
        train_size: Fraction of data for training
        run_backtest: Whether to run backtesting on final signals
        initial_capital: Initial capital for backtesting
        transaction_cost_pct: Transaction cost percentage
        
    Returns:
        Dictionary with meta-labeling results and backtest results (if run)
    """
    log_section("Stage 5: Meta-Labeling for Precision Improvement")
    
    # Prepare data
    feature_cols = [c for c in final_dataset.columns if c not in ['label', 'weight', 't1']]
    X = final_dataset[feature_cols]
    y = final_dataset['label']
    weights = final_dataset['weight']
    t1 = final_dataset['t1']
    
    train_size_count = int(len(X) * train_size)
    X_train_full = X.iloc[:train_size_count]
    X_test = X.iloc[train_size_count:]
    y_train_full = y.iloc[:train_size_count]
    y_test = y.iloc[train_size_count:]
    weights_train_full = weights.iloc[:train_size_count]
    
    # Generate out-of-sample predictions for meta-labeling
    log_info("-> Generating Out-of-Sample predictions for meta-labeling...")
    model_params = best_params.copy()
    model_params.update({
        'objective': 'multiclass', 
        'num_class': 3, 
        'class_weight': 'balanced', 
        'verbosity': -1
    })
    
    n_splits = 4
    splits = np.array_split(np.arange(len(X_train_full)), n_splits + 1)
    primary_preds_oos = []
    
    for i in range(n_splits):
        train_indices = np.concatenate(splits[:i+1])
        val_indices = splits[i+1]
        
        X_train_cv = X_train_full.iloc[train_indices]
        X_val_cv = X_train_full.iloc[val_indices]
        y_train_cv = y_train_full.iloc[train_indices]
        weights_train_cv = weights_train_full.iloc[train_indices]
        
        fold_model = lgb.LGBMClassifier(**model_params)
        fold_model.fit(X_train_cv, y_train_cv, sample_weight=weights_train_cv)
        primary_preds_oos.append(pd.Series(fold_model.predict(X_val_cv), index=X_val_cv.index))
    
    # Prepare meta-labeling data
    primary_pred_train = pd.concat(primary_preds_oos)
    actionable_mask_train = primary_pred_train != 0
    y_train_filtered = y_train_full.loc[primary_pred_train[actionable_mask_train].index]
    meta_y_train = (primary_pred_train[actionable_mask_train] * y_train_filtered > 0).astype(int)
    meta_X_train = X_train_full.loc[meta_y_train.index]
    
    log_info(f"-> Created {len(meta_y_train)} labels for the Meta Model.")
    log_info(f"  - Meta-label distribution:\n{meta_y_train.value_counts()}\n")

    # Train meta-model
    log_info("-> Training Meta Model...")
    meta_model = lgb.LGBMClassifier(
        n_estimators=200, 
        class_weight='balanced', 
        n_jobs=-1, 
        verbosity=-1
    )
    
    if not meta_X_train.empty:
        meta_model.fit(meta_X_train, meta_y_train)
        log_info("-> Meta Model trained.\n")
    else:
        log_warning("No data to train Meta Model. Skipping.")
        return {'meta_model_trained': False}

    # Evaluate combined strategy
    log_info("-> Evaluating combined strategy on the test set...")
    primary_pred_test = pd.Series(primary_model_oos.predict(X_test), index=X_test.index)
    actionable_mask_test = primary_pred_test != 0
    X_test_actionable = X_test.loc[actionable_mask_test]
    
    # Initialize results dictionary
    results = {
        'meta_model_trained': True,
        'meta_model': meta_model,  # Store the trained meta model
        'primary_predictions': primary_pred_test,
        'final_signals': pd.Series(0, index=X_test.index),
        'X_test': X_test,  # Store test features for feature alignment
        'y_test': y_test   # Store test labels
    }
    
    if not X_test_actionable.empty:
        meta_pred_prob_test = pd.Series(
            meta_model.predict_proba(X_test_actionable)[:, 1], 
            index=X_test_actionable.index
        )
        
        # Threshold analysis
        log_subsection("Meta Model Confidence Threshold Analysis")
        threshold_results = []
        
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
            confident_indices = meta_pred_prob_test[meta_pred_prob_test > threshold].index
            final_predictions = pd.Series(0, index=X_test.index)
            
            if not confident_indices.empty:
                final_predictions.loc[confident_indices] = primary_pred_test.loc[confident_indices]
            
            precision_buy = precision_score(
                y_test, final_predictions, 
                labels=[1], average='micro', zero_division=0
            )
            precision_sell = precision_score(
                y_test, final_predictions, 
                labels=[-1], average='micro', zero_division=0
            )
            
            threshold_results.append({
                'threshold': threshold,
                'num_trades': (final_predictions != 0).sum(),
                'precision_buy': precision_buy,
                'precision_sell': precision_sell
            })
        
        threshold_results_df = pd.DataFrame(threshold_results)
        print(threshold_results_df)

        # Best threshold analysis
        best_threshold = threshold_results_df.loc[
            (threshold_results_df['precision_buy'] + threshold_results_df['precision_sell']).idxmax()
        ]['threshold']
        
        log_subsection(f"Detailed Report for Best Threshold ({best_threshold})")
        confident_indices_best = meta_pred_prob_test[meta_pred_prob_test > best_threshold].index
        final_signals = pd.Series(0, index=X_test.index)
        
        if not confident_indices_best.empty:
            final_signals.loc[confident_indices_best] = primary_pred_test.loc[confident_indices_best]
        
        print(classification_report(
            y_test, final_signals, 
            labels=[-1, 0, 1], 
            target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)'], 
            zero_division=0
        ))
        
        # Update results with final signals
        results['final_signals'] = final_signals
        results['best_threshold'] = best_threshold
        results['threshold_analysis'] = threshold_results_df
    
    return results


def generate_signals_from_models(dataset, primary_model, meta_model=None, best_threshold=0.5, 
                                training_features=None) -> pd.Series:
    """
    Generate trading signals from trained models for the entire dataset.
    
    Args:
        dataset: Dataset with features
        primary_model: Trained primary model
        meta_model: Trained meta model (optional)
        best_threshold: Confidence threshold for meta model
        training_features: List of feature columns used during training (for alignment)
        
    Returns:
        Series of trading signals for the entire dataset
    """
    # Prepare features for prediction
    if training_features is not None:
        # Use the exact same features as training
        feature_cols = training_features
    else:
        # Fallback to auto-detection (may cause issues)
        feature_cols = [c for c in dataset.columns if c not in ['label', 'weight', 't1', 'log_returns']]
    
    # Ensure all required features are present
    available_features = [f for f in feature_cols if f in dataset.columns]
    if len(available_features) != len(feature_cols):
        missing_features = set(feature_cols) - set(available_features)
        log_warning(f"Missing features: {missing_features}")
        log_info(f"Using {len(available_features)} out of {len(feature_cols)} features")
    
    X_full = dataset[available_features]
    
    # Generate primary predictions
    primary_predictions = pd.Series(primary_model.predict(X_full), index=X_full.index)
    
    # Apply meta-model filtering if available
    if meta_model is not None:
        actionable_mask = primary_predictions != 0
        X_actionable = X_full.loc[actionable_mask]
        
        final_signals = pd.Series(0, index=X_full.index)
        
        if not X_actionable.empty:
            meta_probs = pd.Series(
                meta_model.predict_proba(X_actionable)[:, 1], 
                index=X_actionable.index
            )
            
            confident_indices = meta_probs[meta_probs > best_threshold].index
            if not confident_indices.empty:
                final_signals.loc[confident_indices] = primary_predictions.loc[confident_indices]
    else:
        # If no meta-model, use primary predictions directly
        final_signals = primary_predictions
    
    return final_signals 