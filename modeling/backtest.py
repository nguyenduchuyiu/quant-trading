#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backtesting utilities for the financial ML pipeline.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.logging import log_info, log_section, log_subsection



def run_event_driven_backtest(
    signals: pd.Series,
    full_df: pd.DataFrame,
    pt_sl_multipliers: list,
    initial_capital: float,
    risk_fraction: float,
    transaction_cost_pct: float,
    long_only: bool = False,
    full_returns: pd.Series = None
):
    """
    Performs a proper event-driven backtest and returns a comprehensive set of metrics.

    Args:
        signals (pd.Series): The trading signals series.
        full_df (pd.DataFrame): The DataFrame from the labeling pipeline.
                                  Must contain 'label', 't1', and 'target_vol'.
        close_prices (pd.Series): The series of close prices, indexed by datetime.
        pt_sl_multipliers (list): The [pt, sl] multipliers used to generate the labels.
        initial_capital (float): The starting capital.
        risk_fraction (float): The fraction of equity to risk per trade (position sizing).
        transaction_cost_pct (float): The transaction cost as percentage (0.1%).
        long_only (bool): If True, treats SELL signals (-1) as EXIT signals (0).
        full_returns (pd.Series): The full, unfiltered log returns of the asset for
                                  a correct Buy and Hold comparison.

    Returns:
        dict: A comprehensive dictionary of performance metrics and data series.
    """
    print("--- Running Correct Event-Driven Backtest for TBM Labels ---")
    if long_only:
        print("Mode: Long-Only (SELL signals will be treated as EXIT)")

    if long_only:
        signals[signals == -1] = 0

    # Portfolio tracking
    close_prices = full_df['close']
    equity = initial_capital
    start_date = full_df.index[0] if not full_df.empty else close_prices.index[0]
    equity_history = [{'date': start_date, 'equity': initial_capital}]
    positions_history = [{'date': start_date, 'position': 0}] # For tracking positions over time
    full_returns = full_returns.loc[start_date:]
    
    num_trades, wins = 0, 0
    pct_returns = []

    for t0, signal_to_act_on in tqdm(signals.items(), desc="Simulating TBM Trades"):
        if signal_to_act_on == 0:
            continue
        
        num_trades += 1
        positions_history.append({'date': t0, 'position': signal_to_act_on}) # Record entry
        
        label_info = full_df.loc[t0]
        trade_outcome = label_info['label']
        pct_return = 0.0

        if signal_to_act_on == 1:
            if trade_outcome == 1:
                pct_return = pt_sl_multipliers[0] * label_info['target_vol']
                wins += 1
            elif trade_outcome == -1:
                pct_return = -pt_sl_multipliers[1] * label_info['target_vol']
            else:
                entry_price = close_prices.get(t0)
                exit_price = close_prices.get(label_info['t1'])
                if entry_price and exit_price and entry_price != 0:
                    pct_return = (exit_price / entry_price) - 1
                    if pct_return > 0: wins += 1
        elif signal_to_act_on == -1:
            if trade_outcome == 1:
                pct_return = -pt_sl_multipliers[0] * label_info['target_vol']
            elif trade_outcome == -1:
                pct_return = pt_sl_multipliers[1] * label_info['target_vol']
                wins += 1
            else:
                entry_price = close_prices.get(t0)
                exit_price = close_prices.get(label_info['t1'])
                if entry_price and exit_price and exit_price != 0:
                    pct_return = (entry_price / exit_price) - 1
                    if pct_return > 0: wins += 1
        
        capital_to_invest = equity * risk_fraction
        dollar_pnl = capital_to_invest * pct_return
        dollar_pnl -= (capital_to_invest * 2 * transaction_cost_pct)
        equity += dollar_pnl
        
        if equity <= 0:
            print(f"BANKRUPTCY at {label_info['t1']}! Simulation stopped.")
            equity = 0
            equity_history.append({'date': label_info['t1'], 'equity': equity})
            positions_history.append({'date': label_info['t1'], 'position': 0})
            break
        
        equity_history.append({'date': label_info['t1'], 'equity': equity})
        positions_history.append({'date': label_info['t1'], 'position': 0}) # Record exit
        pct_returns.append({'date': t0, 'pct_return': pct_return})

    # --- Finalize and Analyze Results ---
    equity_curve = pd.DataFrame(equity_history).set_index('date')['equity'].sort_index()
    equity_curve = equity_curve.resample('D').last().ffill()

    if len(equity_curve) < 2:
        print("Error: Not enough data to calculate performance.")
        return {}

    # --- Calculate KPIs ---
    total_return = (equity_curve.iloc[-1] / initial_capital) - 1
    net_daily_log_returns = np.log(equity_curve / equity_curve.shift(1)).fillna(0)
    
    if net_daily_log_returns.std() == 0:
        sharpe_ratio, annualized_return, annualized_volatility = 0, 0, 0
    else:
        annualized_volatility = net_daily_log_returns.std() * np.sqrt(252)
        annualized_return = np.expm1(net_daily_log_returns.mean() * 252)
        sharpe_ratio = annualized_return / annualized_volatility

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    win_rate = (wins / num_trades) * 100 if num_trades > 0 else 0

    # --- Construct Positions Series ---
    positions_df = pd.DataFrame(positions_history).set_index('date').sort_index()
    positions = positions_df['position'].resample('D').last().ffill().fillna(0)

    # --- Calculate Buy and Hold Comparison ---
    bnh_total_return = 0.0
    buy_and_hold_curve = pd.Series(initial_capital, index=equity_curve.index)
    if full_returns is not None:
        strategy_start = equity_curve.index[0]
        strategy_end = equity_curve.index[-1]
        bnh_returns_slice = full_returns.loc[strategy_start:strategy_end]
        if not bnh_returns_slice.empty:
            buy_and_hold_curve = initial_capital * np.exp(bnh_returns_slice.cumsum())
            buy_and_hold_curve = buy_and_hold_curve.reindex(equity_curve.index, method='ffill').fillna(initial_capital)
            bnh_total_return = (buy_and_hold_curve.iloc[-1] / initial_capital) - 1

    # --- Calculate Buy & Hold Drawdown ---
    bnh_running_max = buy_and_hold_curve.cummax()
    bnh_drawdown = (buy_and_hold_curve - bnh_running_max) / bnh_running_max
    bnh_max_drawdown = bnh_drawdown.min()

    # --- Print Summary & Plot ---
    print("\n--- Event-Driven Backtest Performance ---")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Max Drawdown (Buy & Hold): {bnh_max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total Trades: {num_trades}")
    print(f"Total Wins: {wins}")
    print(f"Total Losses: {num_trades - wins}")
    print(f"Total Returns: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")

    # --- Plotting ---

    # 1. Biểu đồ tăng trưởng vốn (Equity Curve)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 6))
    plt.plot(equity_curve, label='Strategy Equity Curve', lw=2, color='darkcyan')
    plt.plot(buy_and_hold_curve, label=f'Buy and Hold ({bnh_total_return:.1%})', lw=2, color='orange')
    plt.title(f'Equity Curve (Sharpe: {sharpe_ratio:.2f})', fontsize=16)
    plt.ylabel('Equity')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Biểu đồ drawdown (area fill, màu sắc như yêu cầu)
    plt.figure(figsize=(15, 4))
    # Drawdown strategy: xanh dương
    plt.fill_between(drawdown.index, drawdown, 0, color='dodgerblue', alpha=0.3, label='Strategy Drawdown')
    plt.plot(drawdown, color='dodgerblue', lw=1.5)
    # Drawdown buy & hold: cam
    plt.fill_between(bnh_drawdown.index, bnh_drawdown, 0, color='orange', alpha=0.3, label='Buy & Hold Drawdown')
    plt.plot(bnh_drawdown, color='orange', lw=1.5)
    plt.title('Drawdown')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Biểu đồ đường pct_return từng trade và transaction_cost_pct (đường ngang)
    pct_returns_df = pd.DataFrame(pct_returns).set_index('date')
    # Tính net return per trade (sau phí)
    net_returns = pct_returns_df['pct_return'] - 2 * transaction_cost_pct

    # Vẽ biểu đồ Net Return per Trade theo thời gian
    plt.figure(figsize=(15, 5))
    plt.plot(net_returns.index, net_returns, color='green', alpha=0.7, label='Net Return per Trade (after fee)')
    plt.axhline(0, color='red', label='Break-even')
    plt.title("Net Return per Trade vs. Time")
    plt.xlabel("Trade Date")
    plt.ylabel("Net Return")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()
    
    # --- Return Comprehensive Dictionary ---
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
        'net_returns': net_daily_log_returns,
        'buy_and_hold_return': bnh_total_return,
        'buy_and_hold_curve': buy_and_hold_curve,
        'pct_returns': pct_returns_df,
        'net_returns_per_trade': net_returns,
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