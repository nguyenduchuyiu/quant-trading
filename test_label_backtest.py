#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Label Quality Test via Backtesting

Mục đích:
1. Chạy pipeline đến Stage 2 để có labels
2. Dùng labels trực tiếp làm signals 
3. Backtest để xem chất lượng labeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pipeline import FinancialMLPipeline
from utils.config import load_config
from utils.logging import log_info, log_section, log_subsection
from modeling.backtest import run_vectorized_backtest, run_signal_analysis


def test_label_quality_via_backtest(config_path="configs/default.yaml"):
    """
    Test chất lượng labels bằng cách backtest trực tiếp
    
    Args:
        config_path: Đường dẫn config file
    """
    log_section("Testing Label Quality via Direct Backtesting")
    
    # 1. Load config và khởi tạo pipeline
    config = load_config(config_path)
    pipeline = FinancialMLPipeline(config)
    
    # 2. Chạy pipeline đến Stage 2 (có labels)
    
    pipeline.load_data()
    if pipeline.raw_df is None:
        log_info("❌ Failed to load data")
        return None
        
    pipeline.run_feature_engineering()
    if pipeline.features is None or pipeline.features.empty:
        log_info("❌ Failed to engineer features")
        return None
        
    pipeline.run_labeling()
    if pipeline.final_dataset is None or pipeline.final_dataset.empty:
        log_info("❌ Failed to generate labels")
        return None
    
    log_info(f"✅ Pipeline completed. Dataset shape: {pipeline.final_dataset.shape}")
    
    # 3. Extract labels và log_returns
    labels = pipeline.final_dataset['label']
    log_returns = pipeline.final_dataset['log_returns']
    
    # Extract full returns từ dollar bars cho buy-and-hold comparison
    full_returns = None
    
    if pipeline.dollar_bars is not None:
        if 'log_returns' in pipeline.dollar_bars.columns:
            full_returns = pipeline.dollar_bars['log_returns']
            log_info(f"✅ Using full dataset log returns: {len(full_returns)} data points")
        elif 'close' in pipeline.dollar_bars.columns:
            # Calculate log returns from close prices
            close_prices = pipeline.dollar_bars['close']
            full_returns = np.log(close_prices / close_prices.shift(1)).dropna()
            log_info(f"✅ Calculated full dataset returns from close prices: {len(full_returns)} data points")
        else:
            log_info("❌ Cannot calculate returns from dollar_bars")
    elif pipeline.features is not None and 'log_returns' in pipeline.features.columns:
        full_returns = pipeline.features['log_returns']
        log_info(f"✅ Using features dataset returns: {len(full_returns)} data points")
    else:
        log_info("❌ Full returns not available, using event returns for buy-and-hold")
    
    log_subsection("Label Statistics")
    label_counts = labels.value_counts().sort_index()
    total_labels = len(labels)
    
    for label, count in label_counts.items():
        pct = (count / total_labels) * 100
        label_name = {-1: "Sell", 0: "Hold", 1: "Buy"}.get(label, f"Label_{label}")
        log_info(f"  {label_name}: {count:,} ({pct:.1f}%)")
    
    # 4. Dùng labels trực tiếp làm signals
    signals = labels.copy()
    log_info(f"Using {len(signals)} labels directly as trading signals")
    
    # 5. Run signal analysis
    log_subsection("Signal Analysis")
    signal_analysis = run_signal_analysis(signals)
    
    # 6. Run backtest với labels làm signals
    log_subsection("Backtesting Label Quality")
    
    initial_capital = getattr(config, 'initial_capital', 100000.0)
    transaction_cost = getattr(config, 'transaction_cost_pct', 0.001)
    
    backtest_results = run_vectorized_backtest(
        signals=signals,
        asset_log_returns=log_returns,
        initial_capital=initial_capital,
        transaction_cost_pct=transaction_cost,
        full_returns=full_returns  # Pass full returns for proper buy-and-hold
    )
    
    # 7. Phân tích kết quả
    log_section("Label Quality Assessment")
    
    total_return = backtest_results['total_return']
    sharpe_ratio = backtest_results['sharpe_ratio']
    max_drawdown = backtest_results['max_drawdown']
    win_rate = backtest_results['win_rate']
    num_trades = backtest_results['num_trades']
    
    log_info(f"📊 Backtest Results Summary:")
    log_info(f"  Total Return: {total_return:.2%}")
    log_info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    log_info(f"  Max Drawdown: {max_drawdown:.2%}")
    log_info(f"  Win Rate: {win_rate:.1f}%")
    log_info(f"  Number of Trades: {num_trades}")
    
    # 8. Đánh giá chất lượng labeling
    log_subsection("Labeling Quality Assessment")
    
    if sharpe_ratio > 1.0 and total_return > 0:
        log_info("✅ EXCELLENT: Labels có chất lượng tốt - Sharpe > 1.0 và positive returns")
    elif sharpe_ratio > 0.5 and total_return > -0.1:
        log_info("✅ GOOD: Labels acceptable - Sharpe > 0.5")
    elif sharpe_ratio > 0 and total_return > -0.3:
        log_info("⚠️ MODERATE: Labels có potential nhưng cần improve")
    elif sharpe_ratio > -1.0:
        log_info("❌ POOR: Labels quality thấp - cần review labeling method")
    else:
        log_info("❌ VERY POOR: Labels rất kém - có thể có bug trong labeling")
    
    # 9. Forward returns validation (quick check)
    log_subsection("Forward Returns Validation")
    
    # Tính forward returns cho validation
    forward_1_period = log_returns.shift(-1)
    aligned_data = pd.DataFrame({
        'label': labels,
        'forward_return': forward_1_period
    }).dropna()
    
    # Group by label
    stats_by_label = aligned_data.groupby('label')['forward_return'].agg([
        'mean', 'count',
        lambda x: (x > 0).sum() / len(x) * 100  # win rate %
    ]).round(4)
    stats_by_label.columns = ['Mean_Forward_Return', 'Count', 'Forward_Win_Rate']
    
    print("\nForward Returns by Label:")
    print(stats_by_label)
    
    # Expected pattern check
    sell_mean = stats_by_label.loc[-1, 'Mean_Forward_Return'] if -1 in stats_by_label.index else 0
    buy_mean = stats_by_label.loc[1, 'Mean_Forward_Return'] if 1 in stats_by_label.index else 0
    
    if sell_mean < 0 and buy_mean > 0:
        log_info("✅ Forward Returns Pattern: CORRECT - Sell negative, Buy positive")
    elif sell_mean < buy_mean:
        log_info("⚠️ Forward Returns Pattern: MODERATE - Directional trend correct")
    else:
        log_info("❌ Forward Returns Pattern: INCORRECT - Labels may be wrong")
    
    # 10. Summary và recommendations
    log_section("Summary & Recommendations")
    
    results_summary = {
        'total_events': len(labels),
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'sell_forward_return': sell_mean,
        'buy_forward_return': buy_mean,
        'label_distribution': label_counts.to_dict()
    }
    
    log_info("🎯 Quick Recommendations:")
    
    if total_return < -0.5:
        log_info("  - Consider adjusting PT/SL ratios in labeling")
        log_info("  - Review volatility breakout thresholds")
    
    if abs(label_counts.get(-1, 0) - label_counts.get(1, 0)) > len(labels) * 0.3:
        log_info("  - Label distribution very imbalanced - check labeling logic")
    
    if win_rate < 45:
        log_info("  - Low win rate - consider different barrier methods")
    
    if sharpe_ratio < 0:
        log_info("  - Negative Sharpe - fundamental issues with labeling approach")
    
    log_info("✅ Label quality test completed!")
    
    return results_summary


def main():
    """
    Main function để chạy test
    """
    try:
        results = test_label_quality_via_backtest()
        if results:
            print(f"\n🎯 Final Score: Sharpe {results['sharpe_ratio']:.2f}, Return {results['total_return']:.2%}")
        else:
            print("❌ Test failed")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main() 