#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic tests for the financial ML pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace

from pipeline import FinancialMLPipeline
from data.loaders import create_mock_data, get_dollar_bars
from utils.config import load_config


def test_mock_data_creation():
    """Test that mock data is created correctly."""
    mock_df = create_mock_data(num_periods=100)
    
    assert len(mock_df) == 100
    assert all(col in mock_df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert mock_df.index.name == 'date_time'
    assert not mock_df.isnull().all().any()  # No completely null columns


def test_dollar_bars():
    """Test dollar bars creation."""
    mock_df = create_mock_data(num_periods=100)
    mock_df_reset = mock_df.reset_index()
    
    # Add required 'date_time' column
    mock_df_reset.rename(columns={'index': 'date_time'}, inplace=True)
    
    dollar_bars = get_dollar_bars(mock_df_reset, threshold=1000)
    
    assert isinstance(dollar_bars, pd.DataFrame)
    assert len(dollar_bars) > 0
    assert all(col in dollar_bars.columns for col in ['date_time', 'open', 'high', 'low', 'close', 'volume'])


def test_config_loading():
    """Test configuration loading."""
    config = load_config("configs/default.yaml")
    
    assert hasattr(config, 'seed')
    assert hasattr(config, 'pt')
    assert hasattr(config, 'sl')
    assert hasattr(config, 'n_trials')
    assert config.seed == 42


def test_pipeline_initialization():
    """Test pipeline initialization."""
    config = SimpleNamespace(
        data="mock",
        seed=42,
        threshold=None,
        pt=2.0,
        sl=2.0,
        vertical=12,
        n_trials=5,  # Small number for testing
        n_splits=2,
        train_size=0.8
    )
    
    pipeline = FinancialMLPipeline(config)
    
    assert pipeline.config.seed == 42
    assert pipeline.raw_df is None
    assert pipeline.features is None


def test_pipeline_data_loading():
    """Test pipeline data loading stage."""
    config = SimpleNamespace(
        data="nonexistent_file.csv",  # This will trigger mock data creation
        seed=42,
        threshold=None,
        pt=2.0,
        sl=2.0,
        vertical=12,
        n_trials=5,
        n_splits=2,
        train_size=0.8
    )
    
    pipeline = FinancialMLPipeline(config)
    pipeline.load_data()
    
    assert pipeline.raw_df is not None
    assert len(pipeline.raw_df) > 0


def test_pipeline_feature_engineering():
    """Test pipeline feature engineering stage."""
    config = SimpleNamespace(
        data="nonexistent_file.csv",
        seed=42,
        threshold=1000,  # Set small threshold for testing
        pt=2.0,
        sl=2.0,
        vertical=12,
        n_trials=5,
        n_splits=2,
        train_size=0.8
    )
    
    pipeline = FinancialMLPipeline(config)
    pipeline.load_data()
    
    if pipeline.raw_df is not None:
        pipeline.run_feature_engineering()
        
        # Check if features were created (allowing for possibility of empty features)
        assert pipeline.features is not None
        assert pipeline.dollar_bars is not None


if __name__ == "__main__":
    # Run tests manually if called directly
    test_mock_data_creation()
    print("✓ Mock data creation test passed")
    
    test_config_loading()
    print("✓ Config loading test passed")
    
    test_pipeline_initialization()
    print("✓ Pipeline initialization test passed")
    
    test_pipeline_data_loading()
    print("✓ Pipeline data loading test passed")
    
    print("\n✅ All tests passed!") 