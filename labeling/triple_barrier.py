#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Triple-barrier labeling and event sampling utilities for the financial ML pipeline.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.logging import log_info, log_warning, log_error, log_section, log_subsection


def get_volatility_breakout_events(log_returns: pd.Series, short_window: int, long_window: int) -> pd.Index:
    """
    Generates event timestamps based on volatility breakouts.
    
    Args:
        log_returns: Series of log returns
        short_window: Short-term volatility window
        long_window: Long-term volatility window
        
    Returns:
        Index of event timestamps
    """
    print("-> Sampling events using Volatility Breakout method...")
    short_term_vol = log_returns.rolling(window=short_window).std()
    long_term_vol = log_returns.rolling(window=long_window).std()
    
    print("--- VOLATILITY FILTER DIAGNOSTICS ---")
    print(f"  - Short Window: {short_window}, Long Window: {long_window}")
    if not short_term_vol.empty:
        print(f"  - Short-Term Vol Stats: Mean={short_term_vol.mean():.8f}, Max={short_term_vol.max():.8f}")
    if not long_term_vol.empty:
        print(f"  - Long-Term Vol Stats: Mean={long_term_vol.mean():.8f}, Max={long_term_vol.max():.8f}")
    print("---------------------------------")
    
    breakout_signal = (short_term_vol > long_term_vol)
    events_signal = breakout_signal & ~breakout_signal.shift(1).fillna(False)
    
    return log_returns.index[events_signal]


def get_triple_barrier_labels(close: pd.Series, t_events: pd.DatetimeIndex, 
                             pt_sl_multipliers: list, num_bars: int, 
                             target_volatility: pd.Series) -> pd.DataFrame:
    """
    Applies the Triple-Barrier Method to generate labels.
    
    Args:
        close: Price series
        t_events: Event timestamps
        pt_sl_multipliers: [profit_taking_multiplier, stop_loss_multiplier]
        num_bars: Number of bars for vertical barrier
        target_volatility: Target volatility series
        
    Returns:
        DataFrame with labels and metadata
    """
    events = []
    
    for t0 in tqdm(t_events, desc="Processing Triple-Barrier"):
        vol = target_volatility.get(t0)
        if pd.isna(vol) or vol == 0:
            continue
            
        price0 = close.get(t0)
        if pd.isna(price0):
            continue
            
        # Calculate barriers
        pt_barrier = price0 * (1 + pt_sl_multipliers[0] * vol)
        sl_barrier = price0 * (1 - pt_sl_multipliers[1] * vol)
        
        # Vertical barrier
        end_idx_loc = close.index.get_loc(t0) + num_bars
        if end_idx_loc >= len(close.index):
            end_idx_loc = len(close.index) - 1
        vb_timestamp = close.index[end_idx_loc]
        
        # Check path for barrier hits
        path = close.loc[t0:vb_timestamp]
        hit_time, label = vb_timestamp, 0
        
        for t, price in path.iloc[1:].items():
            if price >= pt_barrier:
                hit_time, label = t, 1
                break
            if price <= sl_barrier:
                hit_time, label = t, -1
                break
        
        events.append({
            't0': t0,
            't1': hit_time,
            'target_vol': vol,
            'label': label
        })
    
    if not events:
        return pd.DataFrame(columns=['t1', 'target_vol', 'label']).rename_axis('t0')
    
    return pd.DataFrame(events).set_index('t0')


def get_sample_uniqueness(events_df: pd.DataFrame) -> pd.Series:
    """
    Computes sample uniqueness based on the concurrency of labels.
    
    Args:
        events_df: DataFrame with event information
        
    Returns:
        Series with uniqueness weights
    """
    if events_df.empty:
        return pd.Series(dtype=float)
    
    # Create timeline of all events
    all_times_index = events_df.index.union(events_df['t1']).unique().sort_values()
    
    # Calculate concurrency
    concurrency_events = pd.concat([
        pd.Series(1, index=events_df.index),
        pd.Series(-1, index=events_df['t1'])
    ])
    concurrency = concurrency_events.groupby(level=0).sum().cumsum().reindex(all_times_index).ffill()
    
    # Calculate uniqueness weights
    uniqueness_weights = pd.Series(index=events_df.index, dtype=float)
    
    for idx, row in tqdm(events_df.iterrows(), desc="Calculating Uniqueness"):
        label_concurrency = concurrency.loc[idx : row['t1']]
        avg_uniqueness = (1. / label_concurrency[label_concurrency > 0]).mean()
        uniqueness_weights[idx] = avg_uniqueness
    
    return uniqueness_weights.fillna(uniqueness_weights.mean())


def run_labeling_pipeline(features_df: pd.DataFrame, dollar_bars_df: pd.DataFrame, 
                         pt_sl=None, num_bars=None) -> pd.DataFrame:
    """
    Executes the full event sampling and labeling pipeline.
    
    Args:
        features_df: DataFrame with features
        dollar_bars_df: DataFrame with dollar bars
        pt_sl: [profit_taking, stop_loss] multipliers
        num_bars: Number of bars for vertical barrier
        
    Returns:
        Final dataset with labels and weights
    """
    log_section("Stage 2: Labeling & Data Preparation")
    
    # Event sampling
    vol_short_window, vol_long_window = 12, 48
    t_events = get_volatility_breakout_events(
        features_df['log_returns'], vol_short_window, vol_long_window
    )
    # t_events = features_df.index 
    t_events = t_events[t_events.isin(features_df.index)]
    log_info(f"-> Identified {len(t_events)} trading events.\n")
    
    if len(t_events) == 0:
        log_error("No events were sampled. The pipeline cannot continue.")
        return pd.DataFrame()
    
    # Target volatility estimation
    span = 30
    target_volatility = features_df['log_returns'].ewm(span=span).std()
    log_info(f"-> Estimated target volatility with EWMA (span={span}).\n")
    
    # Set defaults
    if pt_sl is None:
        pt_sl = [2.0, 2.0]
    if num_bars is None:
        num_bars = 12
    
    log_subsection(f"TRIPLE-BARRIER DIAGNOSTICS\n  - PT/SL Multipliers: {pt_sl}\n  - Vertical Barrier: {num_bars} bars\n--------------------------------")
    
    # Generate labels
    labels_df = get_triple_barrier_labels(
        close=dollar_bars_df['close'].reindex(features_df.index),
        t_events=t_events,
        pt_sl_multipliers=pt_sl,
        num_bars=num_bars,
        target_volatility=target_volatility
    )
    log_info("-> Triple-barrier labels generated.\n")
    
    # Calculate sample weights
    sample_weights = get_sample_uniqueness(labels_df)
    log_info("-> Sample uniqueness weights calculated.\n")
    
    # Assemble final dataset
    log_info("-> Assembling final dataset...")
    final_df = features_df.copy()
    final_df = final_df.join(labels_df[['label', 't1']], how='left')
    final_df['weight'] = sample_weights
    final_df = final_df.loc[labels_df.index].copy()
    final_df.dropna(subset=['label'], inplace=True)
    final_df['label'] = final_df['label'].astype(int)
    final_df = final_df.reset_index().merge(
        dollar_bars_df.reset_index(),
        left_on='t0', right_on='date_time', how='left'
    ).set_index('t0')    
    final_df['target_vol'] = target_volatility.loc[final_df.index]
    # Display results
    log_section("FINAL RESULT")
    print(final_df.head(10))
    log_info("Label distribution:")
    print(final_df['label'].value_counts())
    print("-"*60)
    
    return final_df 