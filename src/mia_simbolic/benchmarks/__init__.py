"""
Benchmarks module for M.I.A.-simbolic.

This module contains benchmark problems and baseline optimizers for
comparing the performance of the M.I.A.-simbolic optimizer.
"""

from .problems import (
    SphereFunction, RosenbrockFunction, RastriginFunction, 
    AckleyFunction, NeuralNetworkLoss, PortfolioOptimization
)
from .baselines import (
    AdamOptimizer, SGDOptimizer, LBFGSOptimizer, 
    RMSpropOptimizer, AdagradOptimizer
)

__all__ = [
    # Benchmark problems
    "SphereFunction",
    "RosenbrockFunction",
    "RastriginFunction",
    "AckleyFunction",
    "NeuralNetworkLoss",
    "PortfolioOptimization",
    
    # Baseline optimizers
    "AdamOptimizer",
    "SGDOptimizer",
    "LBFGSOptimizer",
    "RMSpropOptimizer",
    "AdagradOptimizer"
]