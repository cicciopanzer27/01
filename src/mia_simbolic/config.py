"""
Configuration module for M.I.A.-simbolic.

This module provides configuration management for the M.I.A.-simbolic optimizer,
including default settings, configuration loading, and validation.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

class Config(BaseModel):
    """Configuration class for M.I.A.-simbolic.
    
    This class defines the configuration parameters for the optimizer,
    including convergence criteria, agent parameters, and system settings.
    """
    
    # Convergence parameters
    convergence_tolerance: float = Field(1e-6, description="Convergence tolerance for optimization")
    max_iterations: int = Field(1000, description="Maximum number of iterations")
    early_stopping: bool = Field(True, description="Whether to use early stopping")
    patience: int = Field(10, description="Number of iterations with no improvement before early stopping")
    
    # Agent parameters
    generator_learning_rate: float = Field(0.01, description="Learning rate for the generator agent")
    orchestrator_update_frequency: int = Field(5, description="Update frequency for the orchestrator agent")
    validation_threshold: float = Field(0.95, description="Validation threshold for the validation agent")
    
    # Auto-tuning parameters
    auto_tune: bool = Field(True, description="Whether to use Bayesian auto-tuning")
    auto_tune_trials: int = Field(20, description="Number of trials for auto-tuning")
    auto_tune_init_points: int = Field(5, description="Number of initial points for auto-tuning")
    
    # Multi-objective parameters
    alpha: float = Field(0.6, description="Weight for validation loss component")
    beta: float = Field(0.3, description="Weight for computational cost component")
    gamma: float = Field(0.1, description="Weight for regularization component")
    
    # System parameters
    num_threads: int = Field(4, description="Number of threads to use")
    device: str = Field("cpu", description="Device to use (cpu or cuda)")
    log_level: str = Field("INFO", description="Logging level")
    
    # Validation parameters
    validation_frequency: int = Field(10, description="Frequency of validation checks")
    validation_split: float = Field(0.2, description="Fraction of data to use for validation")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate that log_level is a valid logging level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @validator('device')
    def validate_device(cls, v):
        """Validate that device is either 'cpu' or 'cuda'."""
        if v not in ['cpu', 'cuda']:
            raise ValueError("Device must be either 'cpu' or 'cuda'")
        return v
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, uses default config.
        
    Returns:
        Config object with loaded configuration.
    """
    if config_path is None:
        logger.info("No config path provided, using default configuration")
        return Config()
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using default configuration")
        return Config()
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return Config(**config_dict)
    
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        logger.warning("Using default configuration")
        return Config()


def setup_default_config() -> None:
    """Set up default configuration.
    
    This function sets up the default configuration for the optimizer,
    including environment variables and logging configuration.
    """
    # Set up logging
    log_level = os.environ.get('MIA_LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set environment variables for performance
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = str(Config().num_threads)
    
    # Set random seed for reproducibility
    import numpy as np
    import random
    seed = int(os.environ.get('MIA_RANDOM_SEED', '42'))
    np.random.seed(seed)
    random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass
    
    logger.debug("Default configuration set up")