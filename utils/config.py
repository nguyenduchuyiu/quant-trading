#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration utilities for the financial ML pipeline.
"""
import yaml
import os
from types import SimpleNamespace


def load_config(config_path: str = "configs/default.yaml") -> SimpleNamespace:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        SimpleNamespace object with config parameters
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}
    else:
        yaml_config = {}
    
    # Set defaults
    defaults = {
        'data': 'data/sample_data.csv',
        'seed': 42,
        'threshold': None,
        'pt': 2.0,
        'sl': 2.0,
        'vertical': 12,
        'n_trials': 30,
        'n_splits': 4,
        'train_size': 0.8
    }
    
    # Merge configs: YAML overrides defaults
    config = {**defaults, **yaml_config}
    
    return SimpleNamespace(**config)


def get_hyperparameter_config(config: SimpleNamespace) -> dict:
    """
    Extract hyperparameter configuration from config object.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with hyperparameter settings or None
    """
    return getattr(config, 'hyperparameters', None) 