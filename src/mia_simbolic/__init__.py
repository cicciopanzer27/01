"""
M.I.A.-simbolic: Multi-Agent Threshold Optimization

The first multi-agent optimization system with guaranteed convergence
for the validation loss vs computational cost threshold problem.
"""

__version__ = "1.0.0"
__author__ = "M.I.A.-simbolic Team"
__email__ = "contact@mia-simbolic.org"
__license__ = "MIT"

# Core imports
from .core.optimizer import MIAOptimizer
from .core.multi_objective import MultiObjectiveProblem
from .core.agents import SymbolicGenerator, Orchestrator, ValidationAgent
from .core.auto_tuner import BayesianAutoTuner

# Integration imports
from .integrations.pytorch import MIAPyTorchOptimizer
from .integrations.tensorflow import MIATensorFlowOptimizer
from .integrations.sklearn import MIASklearnOptimizer

# Utility imports
from .utils.monitoring import OptimizationMonitor
from .utils.benchmarks import BenchmarkSuite
from .utils.validation import ValidationProtocol

# Configuration
from .config import Config, load_config

__all__ = [
    # Core classes
    "MIAOptimizer",
    "MultiObjectiveProblem",
    "SymbolicGenerator",
    "Orchestrator", 
    "ValidationAgent",
    "BayesianAutoTuner",
    
    # Framework integrations
    "MIAPyTorchOptimizer",
    "MIATensorFlowOptimizer", 
    "MIASklearnOptimizer",
    
    # Utilities
    "OptimizationMonitor",
    "BenchmarkSuite",
    "ValidationProtocol",
    
    # Configuration
    "Config",
    "load_config",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Version compatibility check
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("M.I.A.-simbolic requires Python 3.8 or later")

# Optional dependency warnings
try:
    import torch
except ImportError:
    import warnings
    warnings.warn(
        "PyTorch not found. PyTorch integration will not be available. "
        "Install with: pip install torch",
        ImportWarning
    )

try:
    import tensorflow as tf
except ImportError:
    import warnings
    warnings.warn(
        "TensorFlow not found. TensorFlow integration will not be available. "
        "Install with: pip install tensorflow",
        ImportWarning
    )

# Configure logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Set up default configuration
from .config import setup_default_config
setup_default_config()

# Performance optimization hints
import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"  # Avoid oversubscription

# Banner for CLI usage
def print_banner():
    """Print the M.I.A.-simbolic banner."""
    banner = """
    ███╗   ███╗██╗   █████╗       ███████╗██╗███╗   ███╗██████╗  ██████╗ ██╗     ██╗ ██████╗
    ████╗ ████║██║  ██╔══██╗      ██╔════╝██║████╗ ████║██╔══██╗██╔═══██╗██║     ██║██╔════╝
    ██╔████╔██║██║  ███████║█████╗███████╗██║██╔████╔██║██████╔╝██║   ██║██║     ██║██║     
    ██║╚██╔╝██║██║  ██╔══██║╚════╝╚════██║██║██║╚██╔╝██║██╔══██╗██║   ██║██║     ██║██║     
    ██║ ╚═╝ ██║██║  ██║  ██║      ███████║██║██║ ╚═╝ ██║██████╔╝╚██████╔╝███████╗██║╚██████╗
    ╚═╝     ╚═╝╚═╝  ╚═╝  ╚═╝      ╚══════╝╚═╝╚═╝     ╚═╝╚═════╝  ╚═════╝ ╚══════╝╚═╝ ╚═════╝
    
    Multi-Agent Threshold Optimization with Guaranteed Convergence
    Version: {version} | License: {license} | https://github.com/mia-simbolic/optimization
    """.format(version=__version__, license=__license__)
    print(banner)

# Expose key functions at package level for convenience
def optimize(objective_function, initial_point, **kwargs):
    """
    Convenience function for quick optimization.
    
    Args:
        objective_function: Function to optimize
        initial_point: Starting point for optimization
        **kwargs: Additional arguments passed to MIAOptimizer
        
    Returns:
        OptimizationResult: Result of optimization
    """
    optimizer = MIAOptimizer(**kwargs)
    return optimizer.optimize(objective_function, initial_point)

def benchmark(problem_class="all", **kwargs):
    """
    Convenience function for running benchmarks.
    
    Args:
        problem_class: Class of problems to benchmark
        **kwargs: Additional arguments passed to BenchmarkSuite
        
    Returns:
        BenchmarkResults: Results of benchmark
    """
    suite = BenchmarkSuite(**kwargs)
    return suite.run(problem_class)

# Add convenience functions to __all__
__all__.extend(["optimize", "benchmark", "print_banner"])

