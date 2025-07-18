"""
Core optimizer module for M.I.A.-simbolic.

This module implements the main MIAOptimizer class, which coordinates
the multi-agent optimization process.
"""

import time
import logging
import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass

from ..config import Config, load_config
from .agents import SymbolicGenerator, Orchestrator, ValidationAgent
from .auto_tuner import BayesianAutoTuner

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result of an optimization process.
    
    Attributes:
        x: Optimal point found
        fun: Function value at the optimal point
        nit: Number of iterations
        nfev: Number of function evaluations
        time: Time taken for optimization
        converged: Whether the optimization converged
        success: Whether the optimization was successful
        message: Message describing the result
        gradient_norm: Norm of the gradient at the optimal point
        efficiency_score: Efficiency score of the optimization
    """
    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    time: float
    converged: bool
    success: bool
    message: str
    gradient_norm: Optional[float] = None
    efficiency_score: Optional[float] = None
    
    def __str__(self) -> str:
        """String representation of the optimization result."""
        return (
            f"OptimizationResult:\n"
            f"  Success: {self.success}\n"
            f"  Converged: {self.converged}\n"
            f"  Function value: {self.fun:.6e}\n"
            f"  Iterations: {self.nit}\n"
            f"  Function evaluations: {self.nfev}\n"
            f"  Time: {self.time:.6f} seconds\n"
            f"  Message: {self.message}\n"
            f"  Gradient norm: {self.gradient_norm if self.gradient_norm is not None else 'N/A'}\n"
            f"  Efficiency score: {self.efficiency_score if self.efficiency_score is not None else 'N/A'}"
        )


class MIAOptimizer:
    """Multi-Agent Threshold Optimizer.
    
    This is the main optimizer class that coordinates the multi-agent
    optimization process, including the symbolic generator, orchestrator,
    and validation agents.
    
    Attributes:
        config: Configuration object
        generator: Symbolic generator agent
        orchestrator: Orchestrator agent
        validator: Validation agent
        auto_tuner: Bayesian auto-tuner
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 convergence_tolerance: Optional[float] = None,
                 max_iterations: Optional[int] = None,
                 auto_tune: Optional[bool] = None,
                 monitor: Optional[Any] = None,
                 **kwargs):
        """Initialize the MIAOptimizer.
        
        Args:
            config_path: Path to configuration file
            convergence_tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            auto_tune: Whether to use Bayesian auto-tuning
            monitor: Monitor object for tracking optimization progress
            **kwargs: Additional configuration parameters
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Override configuration with provided parameters
        if convergence_tolerance is not None:
            self.config.convergence_tolerance = convergence_tolerance
        if max_iterations is not None:
            self.config.max_iterations = max_iterations
        if auto_tune is not None:
            self.config.auto_tune = auto_tune
        
        # Update config with any additional parameters
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Initialize agents
        self.generator = SymbolicGenerator(self.config)
        self.orchestrator = Orchestrator(self.config)
        self.validator = ValidationAgent(self.config)
        
        # Initialize auto-tuner if enabled
        self.auto_tuner = BayesianAutoTuner(self.config) if self.config.auto_tune else None
        
        # Set up monitoring
        self.monitor = monitor
        
        logger.info(f"Initialized MIAOptimizer with convergence_tolerance={self.config.convergence_tolerance}, "
                   f"max_iterations={self.config.max_iterations}, auto_tune={self.config.auto_tune}")
    
    def optimize(self, 
                objective_function: Callable[[np.ndarray], float], 
                initial_point: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None,
                constraints: Optional[List[Dict[str, Any]]] = None) -> OptimizationResult:
        """Optimize the objective function.
        
        Args:
            objective_function: Function to optimize
            initial_point: Starting point for optimization
            bounds: Bounds for optimization variables
            constraints: Constraints for optimization
            
        Returns:
            OptimizationResult object with optimization results
        """
        start_time = time.time()
        
        # Initialize optimization state
        x = np.array(initial_point, dtype=float)
        n_dim = len(x)
        
        # Set up bounds if provided
        if bounds is None:
            bounds = [(-np.inf, np.inf)] * n_dim
        
        # Auto-tune hyperparameters if enabled
        if self.auto_tuner is not None:
            logger.info("Auto-tuning hyperparameters...")
            self.auto_tuner.tune(objective_function, x, bounds)
            # Update agent parameters with tuned values
            self.generator.update_params(self.auto_tuner.get_params())
            self.orchestrator.update_params(self.auto_tuner.get_params())
            self.validator.update_params(self.auto_tuner.get_params())
        
        # Initialize optimization variables
        iteration = 0
        n_func_evals = 1
        f_val = objective_function(x)
        best_x = x.copy()
        best_f_val = f_val
        converged = False
        
        # Initialize gradient information
        gradient = self.generator.estimate_gradient(objective_function, x)
        n_func_evals += 2 * n_dim  # Central difference requires 2n evaluations
        
        # Set up early stopping
        patience_counter = 0
        
        # Start optimization loop
        while iteration < self.config.max_iterations:
            # Log progress
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: f(x) = {f_val:.6e}, ||g|| = {np.linalg.norm(gradient):.6e}")
            
            # Update monitor if provided
            if self.monitor is not None:
                self.monitor.update(iteration, x, f_val, gradient)
            
            # Generate step using symbolic generator
            step = self.generator.generate_step(objective_function, x, gradient)
            
            # Coordinate step using orchestrator
            step = self.orchestrator.coordinate_step(step, x, gradient, bounds)
            
            # Take step
            new_x = x + step
            
            # Clip to bounds
            for i in range(n_dim):
                new_x[i] = max(bounds[i][0], min(bounds[i][1], new_x[i]))
            
            # Evaluate new point
            new_f_val = objective_function(new_x)
            n_func_evals += 1
            
            # Update best point if improved
            if new_f_val < best_f_val:
                best_x = new_x.copy()
                best_f_val = new_f_val
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Update current point
            x = new_x
            f_val = new_f_val
            
            # Update gradient
            gradient = self.generator.estimate_gradient(objective_function, x)
            n_func_evals += 2 * n_dim
            
            # Check convergence using validation agent
            converged = self.validator.check_convergence(
                x, f_val, gradient, iteration, patience_counter
            )
            
            if converged:
                logger.info(f"Converged after {iteration+1} iterations")
                break
            
            # Check early stopping
            if self.config.early_stopping and patience_counter >= self.config.patience:
                logger.info(f"Early stopping after {iteration+1} iterations (no improvement for {patience_counter} iterations)")
                break
            
            iteration += 1
        
        # Compute final metrics
        end_time = time.time()
        time_taken = end_time - start_time
        gradient_norm = np.linalg.norm(gradient)
        
        # Compute efficiency score
        efficiency_score = self._compute_efficiency_score(
            best_f_val, iteration, n_func_evals, time_taken, gradient_norm
        )
        
        # Create result object
        result = OptimizationResult(
            x=best_x,
            fun=best_f_val,
            nit=iteration + 1,
            nfev=n_func_evals,
            time=time_taken,
            converged=converged,
            success=converged or iteration == self.config.max_iterations - 1,
            message="Optimization converged successfully" if converged else 
                    "Maximum iterations reached without convergence",
            gradient_norm=gradient_norm,
            efficiency_score=efficiency_score
        )
        
        logger.info(f"Optimization completed: {result.message}")
        
        return result
    
    def _compute_efficiency_score(self, 
                                 f_val: float, 
                                 iterations: int, 
                                 func_evals: int, 
                                 time_taken: float,
                                 gradient_norm: float) -> float:
        """Compute efficiency score for the optimization.
        
        The efficiency score is a weighted combination of:
        - Function value (lower is better)
        - Number of iterations (lower is better)
        - Number of function evaluations (lower is better)
        - Time taken (lower is better)
        - Gradient norm (lower is better)
        
        Returns:
            Efficiency score between 0 and 1 (higher is better)
        """
        # Normalize metrics to [0, 1] range (inverted so lower values are better)
        norm_f_val = 1.0 / (1.0 + abs(f_val))
        norm_iterations = 1.0 / (1.0 + iterations / self.config.max_iterations)
        norm_func_evals = 1.0 / (1.0 + func_evals / (self.config.max_iterations * 2 * len(self.generator.last_x)))
        norm_time = 1.0 / (1.0 + time_taken)
        norm_gradient = 1.0 / (1.0 + gradient_norm)
        
        # Weighted combination
        weights = {
            'f_val': 0.4,
            'iterations': 0.2,
            'func_evals': 0.2,
            'time': 0.1,
            'gradient': 0.1
        }
        
        score = (
            weights['f_val'] * norm_f_val +
            weights['iterations'] * norm_iterations +
            weights['func_evals'] * norm_func_evals +
            weights['time'] * norm_time +
            weights['gradient'] * norm_gradient
        )
        
        return score