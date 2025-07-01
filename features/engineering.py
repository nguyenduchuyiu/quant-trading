#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature engineering utilities for the financial ML pipeline.
"""
import pandas as pd
import numpy as np
from ta import add_all_ta_features
import pywt
import warnings
from utils.logging import log_info, log_warning, log_error
from data.loaders import get_dollar_bars, frac_diff_ffd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a comprehensive set of technical indicators using the 'ta' library.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with technical indicators added
    """
    df_copy = df.copy()
    # The 'ta' library requires capitalized column names.
    df_copy.columns = [c.capitalize() for c in df_copy.columns]
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for col in required_cols:
        if col not in df_copy.columns:
            raise ValueError(f"Required column '{col}' not in DataFrame for TA calculation.")
    
    log_info(f"  - Adding TA features to DataFrame with shape: {df_copy.shape}")
    
    # Suppress FutureWarning from ta library for pandas 2.x compatibility
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="ta")
        df_with_ta = add_all_ta_features(
            df_copy, 
            open="Open", 
            high="High", 
            low="Low", 
            close="Close", 
            volume="Volume", 
            fillna=True
        )
    
    df_with_ta.columns = [c.lower() for c in df_with_ta.columns]
    df_with_ta.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_with_ta.ffill(inplace=True)
    df_with_ta.bfill(inplace=True)
    
    log_info(f"    -> New shape after adding TA features: {df_with_ta.shape}")
    return df_with_ta


def apply_wavelet_transform(df: pd.DataFrame, columns: list, wavelet: str = 'db4', level: int = 1) -> pd.DataFrame:
    """
    Applies Discrete Wavelet Transform (DWT) to specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to apply wavelet transform
        wavelet: Wavelet type
        level: Decomposition level
        
    Returns:
        DataFrame with wavelet features added
    """
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
            cA_padded = np.pad(cA, (0, len(signal) - len(cA)), 'edge')
            cD_padded = np.pad(cD, (0, len(signal) - len(cD)), 'edge')
            new_df[f'{col}_cA'] = cA_padded
            new_df[f'{col}_cD'] = cD_padded
        except Exception as e:
            log_error(f"    -> Error applying wavelet to column '{col}': {e}")
    
    return new_df


def add_lagged_features(df: pd.DataFrame, lag_cols: list, lags: list = [1, 2, 3]) -> pd.DataFrame:
    """
    Add lagged features to the DataFrame.
    
    Args:
        df: Input DataFrame
        lag_cols: Columns to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with lagged features added
    """
    new_df = df.copy()
    log_info("  - Adding lagged features...")
    
    for col in lag_cols:
        if col in new_df.columns:
            for lag in lags:
                new_df[f'{col}_lag_{lag}'] = new_df[col].shift(lag)
    
    return new_df


def run_feature_engineering_pipeline(df: pd.DataFrame, config) -> tuple:
    """
    Executes the full feature engineering and preprocessing pipeline.
    
    Args:
        df: Raw DataFrame with OHLCV data
        threshold: Dollar bar threshold (optional)
        
    Returns:
        Tuple of (features_df, dollar_bars_df)
    """
    from utils.logging import log_section

    log_section("Stage 1: Preprocessing & Feature Engineering")

    # Calculate dollar volume and threshold
    df['dollar_volume'] = df['close'] * df['volume']
    desired_num_bars = getattr(config, 'desired_num_bars')
    total_dollar_volume = df['dollar_volume'].sum()
    dollar_bar_threshold = total_dollar_volume / desired_num_bars
    log_info(f"-> Dollar bar threshold: {dollar_bar_threshold}")
    
    # Create dollar bars
    dollar_bars = get_dollar_bars(df.reset_index(), threshold=dollar_bar_threshold)
    if dollar_bars.empty:
        log_error("CRITICAL WARNING: No dollar bars were created. Check the input data and threshold.")
        return pd.DataFrame(), pd.DataFrame()

    dollar_bars['date_time'] = pd.to_datetime(dollar_bars['date_time'])
    dollar_bars.set_index('date_time', inplace=True)
    log_info(f"-> Created {len(dollar_bars)} dollar bars.\n")

    # Build comprehensive feature set
    log_info("-> Building comprehensive feature set...")
    features_df = add_technical_indicators(dollar_bars)

    # Add returns and fractional differentiation
    features_df['log_returns'] = np.log(features_df['close']).diff()
    features_df['close_ffd'] = frac_diff_ffd(features_df['close'], d=0.4)

    # Apply wavelet transform
    wavelet_cols = ['close', 'volume', 'momentum_rsi', 'trend_macd_diff']
    features_df = apply_wavelet_transform(features_df, columns=wavelet_cols, level=2)

    # Add lagged features
    lag_cols = ['log_returns', 'close_ffd', 'momentum_rsi', 'volatility_bbw']
    features_df = add_lagged_features(features_df, lag_cols)

    log_info("-> Feature engineering complete.\n")

    # Clean final feature set
    log_info("-> Cleaning final feature set...")
    # Keep log_returns in the final features since it's needed for backtesting
    columns_to_drop = ['open', 'high', 'low', 'close', 'volume']
    # Don't drop log_returns even though it's a price-based feature
    final_features = features_df.drop(columns=[col for col in columns_to_drop if col in features_df.columns])
    final_features.dropna(inplace=True)
    log_info(f"-> Cleaning complete. Final feature set has {len(final_features)} samples.\n")

    return final_features, dollar_bars