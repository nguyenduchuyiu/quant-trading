#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backtesting utilities for the financial ML pipeline.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.logging import log_info, log_section, log_subsection


def run_vectorized_backtest(signals: pd.Series, 
                           asset_log_returns: pd.Series, 
                           initial_capital: float = 100000.0,
                           transaction_cost_pct: float = 0.001,
                           full_returns: pd.Series = None) -> dict:
    """
    Performs a vectorized backtest on a set of trading signals.

    Args:
        signals (pd.Series): Series of trading signals (1: Buy, -1: Sell, 0: Hold) 
                             with a DatetimeIndex.
        asset_log_returns (pd.Series): Series of asset log returns with a matching 
                                       DatetimeIndex.
        initial_capital (float): The starting capital for the simulation.
        transaction_cost_pct (float): The percentage cost per transaction (e.g., 0.001 for 0.1%).
        full_returns (pd.Series): Full asset returns for buy-and-hold comparison (optional)
        
    Returns:
        dict: Dictionary containing backtest results and performance metrics
    """

    # --- 1. Align data and create positions ---
    # Đảm bảo signals và returns có cùng index
    aligned_returns, aligned_signals = asset_log_returns.align(signals, join='right', fill_value=0.0)
    
    # Giữ vị thế cho đến khi có tín hiệu mới
    positions = aligned_signals.ffill().fillna(0)

    # --- 2. Calculate strategy returns & costs ---
    # Lợi nhuận gộp: vị thế ngày hôm trước * lợi nhuận tài sản ngày hôm nay
    gross_strategy_log_returns = positions.shift(1) * aligned_returns
    
    # Chi phí: xảy ra khi vị thế thay đổi
    trades = positions.diff().abs()
    costs = trades * transaction_cost_pct
    
    # Lợi nhuận ròng
    net_strategy_log_returns = gross_strategy_log_returns - costs
    
    # --- 3. Create equity curve ---
    cumulative_net_returns = net_strategy_log_returns.cumsum().apply(np.exp)
    equity_curve = initial_capital * cumulative_net_returns

    # --- 4. Calculate Key Performance Indicators (KPIs) ---
    total_return = (equity_curve.iloc[-1] / initial_capital) - 1
    num_days = len(equity_curve)
    trading_days_per_year = 252  # Giả định có 252 ngày giao dịch trong năm
    annualized_return = (1 + total_return) ** (trading_days_per_year / num_days) - 1
    
    annualized_volatility = net_strategy_log_returns.std() * np.sqrt(trading_days_per_year)
    
    # Giả sử tỷ lệ phi rủi ro = 0
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    
    # Tính Max Drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Tính thêm một số metrics khác
    num_trades = int(trades.sum())
    win_rate = calculate_win_rate(net_strategy_log_returns, positions)
    
    # --- 5. Print performance summary ---
    log_subsection("Backtest Performance Summary")
    log_info(f"Initial Capital:       ${initial_capital:,.2f}")
    log_info(f"Final Capital:         ${equity_curve.iloc[-1]:,.2f}")
    log_info(f"Total Return:          {total_return:.2%}")
    log_info(f"Annualized Return:     {annualized_return:.2%}")
    log_info(f"Annualized Volatility: {annualized_volatility:.2%}")
    log_info(f"Sharpe Ratio:          {sharpe_ratio:.2f}")
    log_info(f"Maximum Drawdown:      {max_drawdown:.2%}")
    log_info(f"Number of Trades:      {num_trades}")
    log_info(f"Win Rate:              {win_rate:.2f}%")

    # --- 6. Calculate Buy and Hold comparison ---
    # Sử dụng full_returns nếu có, nếu không thì dùng aligned_returns
    if full_returns is not None:
        # Tính buy and hold từ start date của strategy đến end date
        strategy_start = equity_curve.index[0]
        strategy_end = equity_curve.index[-1]
        
        # Lấy full returns trong khoảng thời gian strategy
        bnh_returns = full_returns.loc[strategy_start:strategy_end]
        
        if len(bnh_returns) > 0:
            buy_and_hold_returns = initial_capital * bnh_returns.cumsum().apply(np.exp)
            # Align với strategy timeline
            buy_and_hold_returns = buy_and_hold_returns.reindex(equity_curve.index, method='ffill')
        else:
            # Fallback nếu không có data
            buy_and_hold_returns = initial_capital * aligned_returns.cumsum().apply(np.exp)
    else:
        # Fallback to original method
        buy_and_hold_returns = initial_capital * aligned_returns.cumsum().apply(np.exp)

    # Calculate buy and hold metrics
    bnh_total_return = (buy_and_hold_returns.iloc[-1] / initial_capital) - 1 if len(buy_and_hold_returns) > 0 else 0
    
    # --- 7. Plot results ---
    plt.figure(figsize=(14, 7))
    equity_curve.plot(label='Strategy Equity Curve', lw=2, color='blue')
    buy_and_hold_returns.plot(label=f'Buy and Hold ({bnh_total_return:.1%})', lw=2, linestyle='--', color='orange')
    plt.title(f'Strategy Performance vs. Buy and Hold (Strategy: {total_return:.1%}, Sharpe: {sharpe_ratio:.2f})')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot drawdown
    plt.figure(figsize=(14, 5))
    drawdown.plot(kind='area', color='red', alpha=0.3)
    plt.title('Strategy Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- 8. Return results dictionary ---
    results = {
        'initial_capital': initial_capital,
        'final_capital': equity_curve.iloc[-1],
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'equity_curve': equity_curve,
        'drawdown': drawdown,
        'positions': positions,
        'net_returns': net_strategy_log_returns,
        'buy_and_hold_return': bnh_total_return,
        'buy_and_hold_curve': buy_and_hold_returns
    }
    
    return results


def calculate_win_rate(returns: pd.Series, positions: pd.Series) -> float:
    """
    Calculate the win rate of trading strategy.
    
    Args:
        returns: Strategy returns
        positions: Trading positions
        
    Returns:
        Win rate as a percentage
    """
    # Chỉ tính win rate cho các ngày có vị thế khác 0
    active_positions = positions.shift(1) != 0
    active_returns = returns[active_positions]
    
    if len(active_returns) == 0:
        return 0.0
    
    # Tính win rate dựa trên số ngày có lợi nhuận dương
    winning_days = (active_returns > 0).sum()
    total_active_days = len(active_returns)
    
    if total_active_days == 0:
        return 0.0
    
    return (winning_days / total_active_days) * 100


def run_signal_analysis(signals: pd.Series) -> dict:
    """
    Analyze trading signals distribution and patterns.
    
    Args:
        signals: Trading signals series
        
    Returns:
        Dictionary with signal analysis results
    """
    log_subsection("Signal Analysis")
    
    signal_counts = signals.value_counts().sort_index()
    total_signals = len(signals)
    
    log_info("Signal Distribution:")
    for signal, count in signal_counts.items():
        signal_name = {-1: "Sell", 0: "Hold", 1: "Buy"}.get(signal, f"Signal_{signal}")
        percentage = (count / total_signals) * 100
        log_info(f"  {signal_name}: {count:,} ({percentage:.1f}%)")
    
    # Calculate signal transitions
    transitions = signals.diff().value_counts().sort_index()
    log_info("\nSignal Transitions:")
    for transition, count in transitions.items():
        if not pd.isna(transition) and transition != 0:
            log_info(f"  {transition:+.0f}: {count:,} times")
    
    return {
        'signal_counts': signal_counts,
        'signal_transitions': transitions,
        'total_signals': total_signals
    } 