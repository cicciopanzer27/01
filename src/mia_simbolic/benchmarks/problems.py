"""
Benchmark problems for M.I.A.-simbolic.

This module implements standard benchmark problems for optimization,
including classical test functions and practical machine learning problems.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

logger = logging.getLogger(__name__)

class BenchmarkProblem:
    """Base class for benchmark problems.
    
    Attributes:
        name: Name of the problem
        dimension: Dimension of the problem
        bounds: Bounds for variables
        noise_std: Standard deviation of noise
    """
    
    def __init__(self, 
                name: str, 
                dimension: int, 
                bounds: Optional[List[Tuple[float, float]]] = None,
                noise_std: float = 0.0):
        """Initialize the benchmark problem.
        
        Args:
            name: Name of the problem
            dimension: Dimension of the problem
            bounds: Bounds for variables
            noise_std: Standard deviation of noise
        """
        self.name = name
        self.dimension = dimension
        self.bounds = bounds or [(-10.0, 10.0)] * dimension
        self.noise_std = noise_std
        
        logger.debug(f"Initialized {self.name} problem with dimension {self.dimension}")
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the objective function.
        
        Args:
            x: Point at which to evaluate the gradient
            
        Returns:
            Gradient vector
        """
        # Default implementation uses finite differences
        eps = 1e-6
        n = len(x)
        grad = np.zeros(n)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            f_plus = self.objective(x_plus)
            f_minus = self.objective(x_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    def initial_point(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate an initial point for optimization.
        
        Args:
            seed: Random seed
            
        Returns:
            Initial point
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random point within bounds
        x = np.zeros(self.dimension)
        for i in range(self.dimension):
            x[i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
        
        return x
    
    def add_noise(self, value: float) -> float:
        """Add noise to a function value.
        
        Args:
            value: Function value
            
        Returns:
            Noisy function value
        """
        if self.noise_std > 0:
            return value + np.random.normal(0, self.noise_std)
        return value


class SphereFunction(BenchmarkProblem):
    """Sphere function benchmark problem.
    
    f(x) = sum(x_i^2)
    """
    
    def __init__(self, dimension: int = 10, noise_std: float = 0.0):
        """Initialize the sphere function problem.
        
        Args:
            dimension: Dimension of the problem
            noise_std: Standard deviation of noise
        """
        super().__init__("Sphere", dimension, noise_std=noise_std)
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        value = np.sum(x**2)
        return self.add_noise(value)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the objective function.
        
        Args:
            x: Point at which to evaluate the gradient
            
        Returns:
            Gradient vector
        """
        return 2 * x


class RosenbrockFunction(BenchmarkProblem):
    """Rosenbrock function benchmark problem.
    
    f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    """
    
    def __init__(self, dimension: int = 10, noise_std: float = 0.0):
        """Initialize the Rosenbrock function problem.
        
        Args:
            dimension: Dimension of the problem
            noise_std: Standard deviation of noise
        """
        super().__init__("Rosenbrock", dimension, noise_std=noise_std)
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        value = 0.0
        for i in range(self.dimension - 1):
            value += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
        
        return self.add_noise(value)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the objective function.
        
        Args:
            x: Point at which to evaluate the gradient
            
        Returns:
            Gradient vector
        """
        grad = np.zeros_like(x)
        
        for i in range(self.dimension - 1):
            if i > 0:
                grad[i] += -200 * (x[i] - x[i-1]**2) * x[i-1] * 2
            grad[i] += 2 * (x[i] - 1)
            grad[i] += 200 * (x[i+1] - x[i]**2) * (-2 * x[i])
        
        # Last component
        grad[-1] = 200 * (x[-1] - x[-2]**2)
        
        return grad


class RastriginFunction(BenchmarkProblem):
    """Rastrigin function benchmark problem.
    
    f(x) = 10 * n + sum(x_i^2 - 10 * cos(2 * pi * x_i))
    """
    
    def __init__(self, dimension: int = 10, noise_std: float = 0.0):
        """Initialize the Rastrigin function problem.
        
        Args:
            dimension: Dimension of the problem
            noise_std: Standard deviation of noise
        """
        super().__init__("Rastrigin", dimension, noise_std=noise_std)
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        value = 10 * self.dimension + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        return self.add_noise(value)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the objective function.
        
        Args:
            x: Point at which to evaluate the gradient
            
        Returns:
            Gradient vector
        """
        return 2 * x + 10 * 2 * np.pi * np.sin(2 * np.pi * x)


class AckleyFunction(BenchmarkProblem):
    """Ackley function benchmark problem.
    
    f(x) = -20 * exp(-0.2 * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(2 * pi * x_i))) + 20 + e
    """
    
    def __init__(self, dimension: int = 10, noise_std: float = 0.0):
        """Initialize the Ackley function problem.
        
        Args:
            dimension: Dimension of the problem
            noise_std: Standard deviation of noise
        """
        super().__init__("Ackley", dimension, noise_std=noise_std)
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        n = float(self.dimension)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        
        value = term1 + term2 + 20 + np.e
        
        return self.add_noise(value)


class NeuralNetworkLoss(BenchmarkProblem):
    """Neural network loss function benchmark problem.
    
    This problem simulates the loss landscape of a neural network.
    """
    
    def __init__(self, dimension: int = 100, noise_std: float = 0.0):
        """Initialize the neural network loss problem.
        
        Args:
            dimension: Dimension of the problem
            noise_std: Standard deviation of noise
        """
        super().__init__("Neural Network Loss", dimension, noise_std=noise_std)
        
        # Generate random problem data
        np.random.seed(42)  # For reproducibility
        self.A = np.random.randn(dimension, dimension)
        self.A = self.A.T @ self.A / dimension  # Make positive definite
        self.b = np.random.randn(dimension)
        
        # Add some non-convexity
        self.C = np.random.randn(dimension, dimension)
        self.C = self.C.T @ self.C / dimension
        self.d = np.random.randn(dimension)
        
        # Eigenvalue decomposition for curvature
        self.eigvals, self.eigvecs = np.linalg.eigh(self.A)
        
        # Scale eigenvalues to create ill-conditioning
        self.eigvals = np.logspace(-3, 3, dimension)
        
        logger.debug(f"Initialized NeuralNetworkLoss with condition number {self.eigvals[-1]/self.eigvals[0]:.2e}")
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Objective function value
        """
        # Quadratic part (convex)
        value = 0.5 * x @ self.A @ x + self.b @ x
        
        # Non-convex part (to simulate neural network loss landscape)
        for i in range(self.dimension // 10):
            idx = i * 10
            value += 0.1 * np.sin(x[idx]) * np.cos(x[(idx + 5) % self.dimension])
        
        # Add L1 regularization
        value += 0.01 * np.sum(np.abs(x))
        
        return self.add_noise(value)


class PortfolioOptimization(BenchmarkProblem):
    """Portfolio optimization benchmark problem.
    
    This problem simulates portfolio optimization with risk constraints.
    """
    
    def __init__(self, dimension: int = 50, noise_std: float = 0.0):
        """Initialize the portfolio optimization problem.
        
        Args:
            dimension: Dimension of the problem (number of assets)
            noise_std: Standard deviation of noise
        """
        super().__init__("Portfolio Optimization", dimension, noise_std=noise_std)
        
        # Generate random problem data
        np.random.seed(42)  # For reproducibility
        
        # Expected returns
        self.returns = np.random.normal(0.05, 0.02, dimension)
        
        # Covariance matrix (positive definite)
        self.cov = np.random.randn(dimension, dimension)
        self.cov = self.cov.T @ self.cov / dimension
        self.cov = self.cov + np.diag(np.random.uniform(0.01, 0.05, dimension))
        
        # Risk aversion parameter
        self.risk_aversion = 2.0
        
        # Bounds for weights (between 0 and 1)
        self.bounds = [(0.0, 1.0)] * dimension
        
        logger.debug(f"Initialized PortfolioOptimization with {dimension} assets")
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the objective function value.
        
        Args:
            x: Point at which to evaluate the objective (portfolio weights)
            
        Returns:
            Objective function value (negative of risk-adjusted return)
        """
        # Normalize weights to sum to 1
        weights = np.clip(x, 0, 1)
        weights = weights / (np.sum(weights) + 1e-8)
        
        # Expected return
        expected_return = weights @ self.returns
        
        # Risk (variance)
        risk = weights @ self.cov @ weights
        
        # Objective: maximize return - risk_aversion * risk
        # Since we're minimizing, we negate this
        value = -expected_return + self.risk_aversion * risk
        
        # Add penalty for weights not summing to 1
        sum_constraint = (np.sum(weights) - 1.0)**2
        value += 100.0 * sum_constraint
        
        return self.add_noise(value)