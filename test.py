import pandas as pd
import numpy as np
from modeling.backtest import run_event_driven_backtest

def test_run_event_driven_backtest_verbose():
    # Tạo dữ liệu giả lập với giá biến động ngẫu nhiên (không tuyến tính)
    rng = np.random.default_rng(42)
    n = 100  # Số lượng ngày lớn hơn
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    # Sinh giá đóng cửa ngẫu nhiên, bắt đầu từ 105, biến động mỗi ngày trong khoảng [-2, 2]
    price_changes = rng.uniform(-2, 2, size=n-1)
    close_prices = [105]
    for change in price_changes:
        close_prices.append(close_prices[-1] + change)
    close_prices = np.array(close_prices)
    target_vol = np.full(n, 0.01)
    t1 = dates[1:].tolist() + [dates[-1]]  # t1 là ngày tiếp theo, ngày cuối giữ nguyên

    # Nhãn: lặp lại chuỗi [1, -1, 0, 1, -1, 0, 1, -1, 0, 1] cho đủ n phần tử
    base_labels = [1, -1, 0, 1, -1, 0, 1, -1, 0, 1]
    labels = (base_labels * (n // len(base_labels) + 1))[:n]

    full_df = pd.DataFrame({
        'label': labels,
        't1': t1,
        'target_vol': target_vol,
        'close': close_prices
    }, index=dates)

    # Tín hiệu giống nhãn
    signals = full_df['label'].copy()

    # Log returns cho buy & hold (giá biến động ngẫu nhiên)
    log_returns = np.log(close_prices / np.roll(close_prices, 1))
    log_returns[0] = 0  # Giá trị đầu tiên không có return
    full_returns = pd.Series(log_returns, index=dates)

    # Tham số backtest
    pt_sl_multipliers = [2, 1]
    initial_capital = 1000
    risk_fraction = 1
    transaction_cost_pct = 0.001
    long_only = False

    # Chạy backtest
    results = run_event_driven_backtest(
        signals, full_df, pt_sl_multipliers, initial_capital,
        risk_fraction, transaction_cost_pct, long_only, full_returns
    )

# Gọi hàm test
test_run_event_driven_backtest_verbose()