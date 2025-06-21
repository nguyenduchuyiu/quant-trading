#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A complete, refactored pipeline for financial machine learning, from raw data
processing to hyperparameter optimization, out-of-sample testing, and
meta-labeling. This script is designed for clarity, modularity, and reusability.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from colorama import Fore, Style, init as colorama_init

# Machine Learning and Analysis Libraries
import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

# Feature Engineering Libraries
from ta import add_all_ta_features
import pywt

# Suppress future warnings for cleaner output

warnings.filterwarnings("ignore", category=FutureWarning)

colorama_init(autoreset=True)

def log_section(title):
    print(Fore.CYAN + "="*60)
    print(Fore.CYAN + title.center(60))
    print(Fore.CYAN + "="*60 + Style.RESET_ALL)

def log_subsection(title):
    print(Fore.GREEN + "-"*60)
    print(Fore.GREEN + title.center(60))
    print(Fore.GREEN + "-"*60 + Style.RESET_ALL)

def log_warning(msg):
    print(Fore.YELLOW + "[WARNING] " + msg + Style.RESET_ALL)

def log_error(msg):
    print(Fore.RED + "[ERROR] " + msg + Style.RESET_ALL)

def log_info(msg):
    print(Fore.WHITE + msg + Style.RESET_ALL)

# ======================================================================================
# SECTION 1: CORE IMPLEMENTATION FUNCTIONS
# This section contains the low-level functions for data structuring,
# feature creation, and labeling.
# ======================================================================================

def get_dollar_bars(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Computes dollar bars from a DataFrame of ticks."""
    df_iterator = df.itertuples(index=False)
    bars = []
    current_bar = {
        'date_time': None, 'open': 0, 'high': -np.inf, 'low': np.inf,
        'close': 0, 'volume': 0, 'dollar_volume': 0
    }
    try:
        row = next(df_iterator)
        current_bar['date_time'], current_bar['open'] = row.date_time, row.close
    except StopIteration:
        return pd.DataFrame(bars)
    for row in df_iterator:
        current_bar['high'] = max(current_bar['high'], row.close)
        current_bar['low'] = min(current_bar['low'], row.close)
        current_bar['volume'] += row.volume
        dollar_value = row.close * row.volume
        current_bar['dollar_volume'] += dollar_value
        if current_bar['dollar_volume'] >= threshold:
            current_bar['close'] = row.close
            bars.append(current_bar.copy())
            current_bar = {'date_time': row.date_time, 'open': row.close, 'high': row.close,
                           'low': row.close, 'close': row.close, 'volume': 0, 'dollar_volume': 0}
    if current_bar['dollar_volume'] > 0:
        bars.append(current_bar)
    return pd.DataFrame(bars)

def get_frac_diff_weights(d: float, thres: float = 1e-4) -> np.ndarray:
    """Computes weights for fractional differentiation."""
    w, k = [1.], 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres: break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])

def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    """Computes Fractional Differentiation with a fixed-width window."""
    weights = get_frac_diff_weights(d, thres)
    width = len(weights) - 1
    df_ = pd.Series(0., index=series.index)
    series_values = series.values.flatten()
    for i in range(width, len(series)):
        window = series_values[i - width : i + 1]
        df_.iloc[i] = np.dot(weights, window)
    return df_

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a comprehensive set of technical indicators using the 'ta' library."""
    df_copy = df.copy()
    # The 'ta' library requires capitalized column names.
    df_copy.columns = [c.capitalize() for c in df_copy.columns]
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df_copy.columns:
            raise ValueError(f"Required column '{col}' not in DataFrame for TA calculation.")
    log_info(f"  - Adding TA features to DataFrame with shape: {df_copy.shape}")
    df_with_ta = add_all_ta_features(df_copy, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    df_with_ta.columns = [c.lower() for c in df_with_ta.columns]
    df_with_ta.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_with_ta.ffill(inplace=True)
    df_with_ta.bfill(inplace=True)
    log_info(f"    -> New shape after adding TA features: {df_with_ta.shape}")
    return df_with_ta

def apply_wavelet_transform(df: pd.DataFrame, columns: list, wavelet: str = 'db4', level: int = 1) -> pd.DataFrame:
    """Applies Discrete Wavelet Transform (DWT) to specified columns."""
    new_df = df.copy()
    log_info(f"  - Applying Wavelet Transform (wavelet: {wavelet}) to columns: {columns}")
    for col in columns:
        if col not in new_df.columns:
            log_warning(f"    -> Warning: Column '{col}' not found for wavelet transform, skipping.")
            continue
        signal = new_df[col].values
        if len(signal) < pywt.Wavelet(wavelet).dec_len + level - 1:
            log_warning(f"    -> Data too short for column '{col}' at level={level}, skipping.")
            continue
        try:
            coeffs = pywt.dwt(signal, wavelet, mode='symmetric')
            cA, cD = coeffs
            cA_padded, cD_padded = np.pad(cA, (0, len(signal) - len(cA)), 'edge'), np.pad(cD, (0, len(signal) - len(cD)), 'edge')
            new_df[f'{col}_cA'], new_df[f'{col}_cD'] = cA_padded, cD_padded
        except Exception as e:
            log_error(f"    -> Error applying wavelet to column '{col}': {e}")
    return new_df

def get_volatility_breakout_events(log_returns: pd.Series, short_window: int, long_window: int) -> pd.Index:
    """Generates event timestamps based on volatility breakouts."""
    print("-> Sampling events using Volatility Breakout method...")
    short_term_vol = log_returns.rolling(window=short_window).std()
    long_term_vol = log_returns.rolling(window=long_window).std()
    print("--- VOLATILITY FILTER DIAGNOSTICS ---")
    print(f"  - Short Window: {short_window}, Long Window: {long_window}")
    if not short_term_vol.empty: print(f"  - Short-Term Vol Stats: Mean={short_term_vol.mean():.8f}, Max={short_term_vol.max():.8f}")
    if not long_term_vol.empty: print(f"  - Long-Term Vol Stats: Mean={long_term_vol.mean():.8f}, Max={long_term_vol.max():.8f}")
    print("---------------------------------")
    breakout_signal = (short_term_vol > long_term_vol)
    events_signal = breakout_signal & ~breakout_signal.shift(1).fillna(False)
    return log_returns.index[events_signal]

def get_triple_barrier_labels(close: pd.Series, t_events: pd.DatetimeIndex, pt_sl_multipliers: list, num_bars: int, target_volatility: pd.Series) -> pd.DataFrame:
    """Applies the Triple-Barrier Method to generate labels."""
    events = []
    for t0 in tqdm(t_events, desc="Processing Triple-Barrier"):
        vol = target_volatility.get(t0)
        if pd.isna(vol) or vol == 0: continue
        price0 = close.get(t0)
        if pd.isna(price0): continue
        pt_barrier, sl_barrier = price0 * (1 + pt_sl_multipliers[0] * vol), price0 * (1 - pt_sl_multipliers[1] * vol)
        end_idx_loc = close.index.get_loc(t0) + num_bars
        if end_idx_loc >= len(close.index): end_idx_loc = len(close.index) - 1
        vb_timestamp = close.index[end_idx_loc]
        path = close.loc[t0:vb_timestamp]
        hit_time, label = vb_timestamp, 0
        for t, price in path.iloc[1:].items():
            if price >= pt_barrier: hit_time, label = t, 1; break
            if price <= sl_barrier: hit_time, label = t, -1; break
        events.append({'t0': t0, 't1': hit_time, 'target_vol': vol, 'label': label})
    if not events: return pd.DataFrame(columns=['t1', 'target_vol', 'label']).rename_axis('t0')
    return pd.DataFrame(events).set_index('t0')

def get_sample_uniqueness(events_df: pd.DataFrame) -> pd.Series:
    """Computes sample uniqueness based on the concurrency of labels."""
    if events_df.empty: return pd.Series(dtype=float)
    all_times_index = events_df.index.union(events_df['t1']).unique().sort_values()
    concurrency_events = pd.concat([pd.Series(1, index=events_df.index), pd.Series(-1, index=events_df['t1'])])
    concurrency = concurrency_events.groupby(level=0).sum().cumsum().reindex(all_times_index).ffill()
    uniqueness_weights = pd.Series(index=events_df.index, dtype=float)
    for idx, row in tqdm(events_df.iterrows(), desc="Calculating Uniqueness"):
        label_concurrency = concurrency.loc[idx : row['t1']]
        avg_uniqueness = (1. / label_concurrency[label_concurrency > 0]).mean()
        uniqueness_weights[idx] = avg_uniqueness
    return uniqueness_weights.fillna(uniqueness_weights.mean())

# ======================================================================================
# SECTION 2: PIPELINE EXECUTION WORKFLOW
# ======================================================================================

def run_feature_engineering_pipeline(df: pd.DataFrame, threshold=None) -> tuple:
    """Executes the full feature engineering and preprocessing pipeline."""
    log_section("Stage 1: Preprocessing & Feature Engineering")
    df['dollar_volume'] = df['close'] * df['volume']
    avg_dollar_volume = df['dollar_volume'].mean()
    dollar_bar_threshold = threshold if threshold is not None else avg_dollar_volume * 1.5
    dollar_bars = get_dollar_bars(df.reset_index(), threshold=dollar_bar_threshold)
    if dollar_bars.empty:
        log_error("CRITICAL WARNING: No dollar bars were created. Check the input data and threshold.")
        return pd.DataFrame(), pd.DataFrame()
    dollar_bars['date_time'] = pd.to_datetime(dollar_bars['date_time'])
    dollar_bars.set_index('date_time', inplace=True)
    log_info(f"-> Created {len(dollar_bars)} dollar bars.\n")
    
    log_info("-> Building comprehensive feature set...")
    features_df = add_technical_indicators(dollar_bars)
    features_df['log_returns'] = np.log(features_df['close']).diff()
    features_df['close_ffd'] = frac_diff_ffd(features_df[['close']], d=0.4)
    wavelet_cols = ['close', 'volume', 'momentum_rsi', 'trend_macd_diff']
    features_df = apply_wavelet_transform(features_df, columns=wavelet_cols, level=2)
    log_info("  - Adding lagged features...")
    lag_cols = ['log_returns', 'close_ffd', 'momentum_rsi', 'volatility_bbw']
    for col in lag_cols:
        if col in features_df.columns:
            for lag in [1, 2, 3]:
                features_df[f'{col}_lag_{lag}'] = features_df[col].shift(lag)
    log_info("-> Feature engineering complete.\n")
    
    log_info("-> Cleaning final feature set...")
    final_features = features_df.drop(columns=['open', 'high', 'low', 'close', 'volume'])
    final_features.dropna(inplace=True)
    log_info(f"-> Cleaning complete. Final feature set has {len(final_features)} samples.\n")
    return final_features, dollar_bars

def run_labeling_pipeline(features_df: pd.DataFrame, dollar_bars_df: pd.DataFrame, pt_sl=None, num_bars=None) -> pd.DataFrame:
    """Executes the full event sampling and labeling pipeline."""
    log_section("Stage 2: Labeling & Data Preparation")
    vol_short_window, vol_long_window = 12, 48
    t_events = get_volatility_breakout_events(features_df['log_returns'], vol_short_window, vol_long_window)
    t_events = t_events[t_events.isin(features_df.index)]
    log_info(f"-> Identified {len(t_events)} trading events.\n")
    if len(t_events) == 0:
        log_error("No events were sampled. The pipeline cannot continue.")
        return pd.DataFrame()
        
    span = 30
    target_volatility = features_df['log_returns'].ewm(span=span).std()
    log_info(f"-> Estimated target volatility with EWMA (span={span}).\n")
    
    if pt_sl is None:
        pt_sl = [2.0, 2.0]
    if num_bars is None:
        num_bars = 12
    log_subsection(f"TRIPLE-BARRIER DIAGNOSTICS\n  - PT/SL Multipliers: {pt_sl}\n  - Vertical Barrier: {num_bars} bars\n--------------------------------")
    labels_df = get_triple_barrier_labels(close=dollar_bars_df['close'].reindex(features_df.index), t_events=t_events, pt_sl_multipliers=pt_sl, num_bars=num_bars, target_volatility=target_volatility)
    log_info("-> Triple-barrier labels generated.\n")
    
    sample_weights = get_sample_uniqueness(labels_df)
    log_info("-> Sample uniqueness weights calculated.\n")
    
    log_info("-> Assembling final dataset...")
    final_df = features_df.copy()
    final_df = final_df.join(labels_df[['label', 't1']], how='left')
    final_df['weight'] = sample_weights
    final_df = final_df.loc[labels_df.index].copy()
    final_df.dropna(subset=['label'], inplace=True)
    final_df['label'] = final_df['label'].astype(int)
    
    log_section("FINAL RESULT")
    print(final_df.head(10))
    log_info("Label distribution:")
    print(final_df['label'].value_counts())
    print("-"*60)
    return final_df

def run_hyperparameter_optimization(X, y, weights, t1, n_trials=30, hyper_config=None, n_splits=4) -> dict:
    """Finds the best hyperparameters for the primary model using Optuna and Walk-Forward CV."""
    log_section("Stage 3: Hyperparameter Optimization")

    hyper_keys = [
        'n_estimators', 'learning_rate', 'num_leaves', 'max_depth', 'lambda_l1', 'lambda_l2',
        'feature_fraction', 'bagging_fraction', 'bagging_freq', 'min_child_samples'
    ]
    if hyper_config and all(hyper_config.get(k) is not None for k in hyper_keys):
        log_info("Tất cả hyperparameter đã được cấu hình từ YAML. Bỏ qua Optuna, dùng trực tiếp các giá trị này.")
        best_params = {k: hyper_config[k] for k in hyper_keys}
        return best_params

    def get_param_from_config_or_trial(param_name, trial, suggest_func, *args, **kwargs):
        if hyper_config and param_name in hyper_config and hyper_config[param_name] is not None:
            return hyper_config[param_name]
        return getattr(trial, suggest_func)(param_name, *args, **kwargs)

    def objective(trial):
        params = {
            'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 3,
            'verbosity': -1, 'boosting_type': 'gbdt',
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
        f1_scores = []
        splits = np.array_split(np.arange(len(X)), n_splits + 1)
        for i in range(n_splits):
            train_indices, test_indices = np.concatenate(splits[:i+1]), splits[i+1]
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            weights_train = weights.iloc[train_indices]
            last_train_t1 = t1.iloc[train_indices].max()
            embargo_mask = t1.index[test_indices] > last_train_t1
            X_test, y_test = X_test.loc[embargo_mask], y_test.loc[embargo_mask]
            if X_train.empty or X_test.empty: continue
            model = lgb.LGBMClassifier(**params, class_weight='balanced')
            model.fit(X_train, y_train, sample_weight=weights_train)
            y_pred = model.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        return np.mean(f1_scores) if f1_scores else 0.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    log_info("-> Optimization complete.\n")
    log_subsection("Best Hyperparameters Found")
    best_params = study.best_trial.params
    log_info(f"  - Best Weighted F1-score: {study.best_trial.value:.5f}")
    for key, value in best_params.items():
        log_info(f"    - {key}: {value}")
    return best_params

def run_out_of_sample_test(X, y, weights, best_params, train_size=0.8) -> lgb.LGBMClassifier:
    """Trains a model on the first train_size of data, evaluates on the rest, and returns the trained model."""
    log_section("Stage 4: Out-of-Sample Testing")
    train_size_count = int(len(X) * train_size)
    X_train, X_test = X.iloc[:train_size_count], X.iloc[train_size_count:]
    y_train, y_test = y.iloc[:train_size_count], y.iloc[train_size_count:]
    weights_train = weights.iloc[:train_size_count]
    log_info(f"-> Training Set Size: {len(X_train)}, Test Set Size: {len(X_test)}\n")
    
    model_params = best_params.copy()
    model_params.update({'objective': 'multiclass', 'num_class': 3, 'class_weight': 'balanced', 'verbosity': -1})
    
    primary_model = lgb.LGBMClassifier(**model_params)
    primary_model.fit(X_train, y_train, sample_weight=weights_train)
    log_info("-> Final primary model trained on train set.\n")
    
    y_pred_test = primary_model.predict(X_test)
    log_subsection("Out-of-Sample Performance (Primary Model)")
    print(classification_report(y_test, y_pred_test, labels=[-1, 0, 1], target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)'], zero_division=0))
    print("="*60)
    
    cm = confusion_matrix(y_test, y_pred_test, labels=[-1, 0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Sell', 'Pred Hold', 'Pred Buy'], yticklabels=['True Sell', 'True Hold', 'True Buy'])
    plt.title('Out-of-Sample Confusion Matrix (Primary Model)')
    plt.show()
    return primary_model

def run_meta_labeling_pipeline(final_dataset, best_params, primary_model_oos, train_size=0.8):
    """Builds and evaluates a meta-model to filter primary model signals."""
    log_section("Stage 5: Meta-Labeling for Precision Improvement")
    feature_cols = [c for c in final_dataset.columns if c not in ['label', 'weight', 't1']]
    X, y, weights, t1 = final_dataset[feature_cols], final_dataset['label'], final_dataset['weight'], final_dataset['t1']
    
    train_size_count = int(len(X) * train_size)
    X_train_full, X_test = X.iloc[:train_size_count], X.iloc[train_size_count:]
    y_train_full, y_test = y.iloc[:train_size_count], y.iloc[train_size_count:]
    weights_train_full = weights.iloc[:train_size_count]
    
    log_info("-> Generating Out-of-Sample predictions for meta-labeling...")
    model_params = best_params.copy()
    model_params.update({'objective': 'multiclass', 'num_class': 3, 'class_weight': 'balanced', 'verbosity': -1})
    n_splits = 4
    splits = np.array_split(np.arange(len(X_train_full)), n_splits + 1)
    primary_preds_oos = []
    for i in range(n_splits):
        train_indices, val_indices = np.concatenate(splits[:i+1]), splits[i+1]
        X_train_cv, X_val_cv = X_train_full.iloc[train_indices], X_train_full.iloc[val_indices]
        y_train_cv, _ = y_train_full.iloc[train_indices], y_train_full.iloc[val_indices]
        weights_train_cv = weights_train_full.iloc[train_indices]
        fold_model = lgb.LGBMClassifier(**model_params)
        fold_model.fit(X_train_cv, y_train_cv, sample_weight=weights_train_cv)
        primary_preds_oos.append(pd.Series(fold_model.predict(X_val_cv), index=X_val_cv.index))
    
    primary_pred_train = pd.concat(primary_preds_oos)
    actionable_mask_train = primary_pred_train != 0
    y_train_filtered = y_train_full.loc[primary_pred_train[actionable_mask_train].index]
    meta_y_train = (primary_pred_train[actionable_mask_train] * y_train_filtered > 0).astype(int)
    meta_X_train = X_train_full.loc[meta_y_train.index]
    log_info(f"-> Created {len(meta_y_train)} labels for the Meta Model.")
    log_info(f"  - Meta-label distribution:\n{meta_y_train.value_counts()}\n")

    log_info("-> Training Meta Model...")
    meta_model = lgb.LGBMClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, verbosity=-1)
    if not meta_X_train.empty:
        meta_model.fit(meta_X_train, meta_y_train)
        log_info("-> Meta Model trained.\n")
    else:
        log_warning("No data to train Meta Model. Skipping.")
        return

    log_info("-> Evaluating combined strategy on the test set...")
    primary_pred_test = pd.Series(primary_model_oos.predict(X_test), index=X_test.index)
    actionable_mask_test = primary_pred_test != 0
    X_test_actionable = X_test.loc[actionable_mask_test]
    
    if not X_test_actionable.empty:
        meta_pred_prob_test = pd.Series(meta_model.predict_proba(X_test_actionable)[:, 1], index=X_test_actionable.index)
        log_subsection("Meta Model Confidence Threshold Analysis")
        results = []
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
            confident_indices = meta_pred_prob_test[meta_pred_prob_test > threshold].index
            final_predictions = pd.Series(0, index=X_test.index)
            if not confident_indices.empty:
                final_predictions.loc[confident_indices] = primary_pred_test.loc[confident_indices]
            precision_buy = precision_score(y_test, final_predictions, labels=[1], average='micro', zero_division=0)
            precision_sell = precision_score(y_test, final_predictions, labels=[-1], average='micro', zero_division=0)
            results.append({'threshold': threshold, 'num_trades': (final_predictions != 0).sum(), 'precision_buy': precision_buy, 'precision_sell': precision_sell})
        
        results_df = pd.DataFrame(results)
        print(results_df)

        best_threshold = results_df.loc[(results_df['precision_buy'] + results_df['precision_sell']).idxmax()]['threshold']
        log_subsection(f"Detailed Report for Best Threshold ({best_threshold})")
        confident_indices_best = meta_pred_prob_test[meta_pred_prob_test > best_threshold].index
        final_predictions_best = pd.Series(0, index=X_test.index)
        if not confident_indices_best.empty:
            final_predictions_best.loc[confident_indices_best] = primary_pred_test.loc[confident_indices_best]
        print(classification_report(y_test, final_predictions_best, labels=[-1, 0, 1], target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)'], zero_division=0))

# ======================================================================================
# SECTION 3: MAIN EXECUTION BLOCK
# ======================================================================================

if __name__ == '__main__':
    import argparse
    import yaml
    import os

    # Parse YAML config path from terminal
    parser = argparse.ArgumentParser(description="Run the full pipeline with YAML config.")
    parser.add_argument('--config', type=str, default="configs/default.yaml", help="Path to the YAML config file.")
    args_terminal = parser.parse_args()

    CONFIG_YAML_PATH = args_terminal.config

    # Load YAML config if file exists, else use default
    if os.path.exists(CONFIG_YAML_PATH):
        with open(CONFIG_YAML_PATH, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}
    else:
        yaml_config = {}

    # Merge configs: YAML > default
    config = yaml_config

    # For backward compatibility, assign to args-like object
    from types import SimpleNamespace
    args = SimpleNamespace(**config)

    # Extract hyperparameter config if present
    hyper_config = getattr(args, 'hyperparameters', None)

    np.random.seed(args.seed)
    # --- Load Data ---
    try:
        raw_df = pd.read_csv(args.data, parse_dates=['date'])
        raw_df.set_index('date', inplace=True)
    except FileNotFoundError:
        log_warning(f"'{args.data}' not found. Creating mock 5-minute data for demonstration.")
        num_periods = 12 * 24 * 10
        date_rng = pd.to_datetime(pd.date_range(start='2024-01-01', periods=num_periods, freq='5T'))
        mock_data = {
            'date_time': date_rng,
            'open': (np.random.uniform(-0.001, 0.001, num_periods)).cumsum() + 200,
            'volume': np.random.randint(100, 1000, num_periods)
        }
        raw_df = pd.DataFrame(mock_data)
        raw_df['high'] = raw_df['open'] + np.random.uniform(0, 0.05, num_periods)
        raw_df['low'] = raw_df['open'] - np.random.uniform(0, 0.05, num_periods)
        raw_df['close'] = raw_df['open'] + np.random.uniform(-0.04, 0.04, num_periods)
        raw_df.set_index('date_time', inplace=True)

    raw_df.index.name = 'date_time'

    # --- Run Pipeline ---
    features, dollar_bars = run_feature_engineering_pipeline(raw_df, threshold=getattr(args, 'threshold', None))
    
    if not features.empty:
        pt_sl = [args.pt, args.sl]
        final_dataset = run_labeling_pipeline(features, dollar_bars, pt_sl=pt_sl, num_bars=args.vertical)
        
        if not final_dataset.empty:
            log_info("\nPIPELINE STAGES 1 & 2 COMPLETED SUCCESSFULLY.")
            
            # Prepare data for modeling stages
            feature_cols = [c for c in final_dataset.columns if c not in ['label', 'weight', 't1']]
            X = final_dataset[feature_cols]
            y = final_dataset['label']
            weights = final_dataset['weight']
            t1 = final_dataset['t1']

            weights.replace([np.inf, -np.inf], np.nan, inplace=True)
            if not weights.dropna().empty:
                weights.fillna(weights.dropna().max(), inplace=True)
            else:
                weights.fillna(1.0, inplace=True)

            # Run Modeling Stages
            best_hyperparams = run_hyperparameter_optimization(
                X, y, weights, t1, args.n_trials, hyper_config, n_splits=getattr(args, 'n_splits', 4)
            )
            # The primary model is trained once and returned for reuse.
            primary_model = run_out_of_sample_test(
                X, y, weights, best_hyperparams, train_size=getattr(args, 'train_size', 0.8)
            )
            # Pass the trained model to the meta-labeling stage to avoid re-fitting.
            run_meta_labeling_pipeline(
                final_dataset, best_hyperparams, primary_model, train_size=getattr(args, 'train_size', 0.8)
            )

