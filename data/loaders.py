#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data loading and processing utilities for the financial ML pipeline.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from utils.logging import log_info, log_warning, log_error


def get_dollar_bars(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Computes dollar bars from a DataFrame of ticks.
    
    Args:
        df: DataFrame with columns ['date_time', 'close', 'volume']
        threshold: Dollar volume threshold for creating bars
        
    Returns:
        DataFrame with dollar bars
    """
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
            current_bar = {
                'date_time': row.date_time, 'open': row.close, 'high': row.close,
                'low': row.close, 'close': row.close, 'volume': 0, 'dollar_volume': 0
            }
    
    if current_bar['dollar_volume'] > 0:
        bars.append(current_bar)
    
    return pd.DataFrame(bars)


def get_frac_diff_weights(d: float, thres: float = 1e-4) -> np.ndarray:
    """
    Computes weights for fractional differentiation.
    
    Args:
        d: Differentiation parameter
        thres: Threshold for weight cutoff
        
    Returns:
        Array of weights
    """
    w, k = [1.], 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])


def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    """
    Computes Fractional Differentiation with a fixed-width window.
    
    Args:
        series: Time series to differentiate
        d: Differentiation parameter
        thres: Threshold for weight cutoff
        
    Returns:
        Fractionally differentiated series
    """
    weights = get_frac_diff_weights(d, thres)
    width = len(weights) - 1
    df_ = pd.Series(0., index=series.index)
    series_values = series.values.flatten()
    
    for i in range(width, len(series)):
        window = series_values[i - width : i + 1]
        df_.iloc[i] = np.dot(weights, window)
    
    return df_


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load raw data from CSV file or create mock data if file doesn't exist.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        DataFrame with raw data
    """
    try:
        raw_df = pd.read_csv(data_path, parse_dates=['date'])
        raw_df.set_index('date', inplace=True)
        log_info(f"Successfully loaded data from {data_path}")
    except FileNotFoundError:
        log_warning(f"'{data_path}' not found. Creating mock 5-minute data for demonstration.")
        raw_df = create_mock_data()
    
    raw_df.index.name = 'date_time'
    return raw_df


def create_mock_data(num_periods: int = 12 * 24 * 10) -> pd.DataFrame:
    """
    Create mock financial data for testing purposes.
    
    Args:
        num_periods: Number of periods to generate
        
    Returns:
        DataFrame with mock OHLCV data
    """
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
    
    return raw_df 