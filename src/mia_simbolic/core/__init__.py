"""
Core module for M.I.A.-simbolic.

This module contains the core components of the M.I.A.-simbolic optimizer,
including the optimizer, multi-objective problem, agents, and auto-tuner.
"""

from .optimizer import MIAOptimizer, OptimizationResult
from .multi_objective import MultiObjectiveProblem, QuadraticProblem, NeuralNetworkProblem
from .agents import SymbolicGenerator, Orchestrator, ValidationAgent
from .auto_tuner import BayesianAutoTuner

__all__ = [
    "MIAOptimizer",
    "OptimizationResult",
    "MultiObjectiveProblem",
    "QuadraticProblem",
    "NeuralNetworkProblem",
    "SymbolicGenerator",
    "Orchestrator",
    "ValidationAgent",
    "BayesianAutoTuner"
]