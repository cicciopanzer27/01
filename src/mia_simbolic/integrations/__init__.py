"""
Integrations module for M.I.A.-simbolic.

This module contains integrations with popular machine learning frameworks,
including PyTorch, TensorFlow, and scikit-learn.
"""

from .pytorch import MIAPyTorchOptimizer
from .tensorflow import MIATensorFlowOptimizer
from .sklearn import MIASklearnOptimizer, MIALinearRegression

__all__ = [
    "MIAPyTorchOptimizer",
    "MIATensorFlowOptimizer",
    "MIASklearnOptimizer",
    "MIALinearRegression"
]