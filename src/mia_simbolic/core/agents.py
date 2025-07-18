"""
Agents module for M.I.A.-simbolic.

This module implements the three main agents in the M.I.A.-simbolic system:
1. SymbolicGenerator: Generates optimization steps using symbolic reasoning
2. Orchestrator: Coordinates the optimization process
3. ValidationAgent: Validates convergence and stability
"""

import logging
import numpy as np
from typing import Callable, Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod

from ..config import Config

logger = logging.getLogger(__name__)

class Agent(ABC):
    """Abstract base class for optimization agents.
    
    Attributes:
        config: Configuration object
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the agent.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def update_params(self, params: Dict[str, Any]) -> None:
        """Update agent parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        pass


class SymbolicGenerator(Agent):
    """Symbolic Generator Agent.
    
    This agent is responsible for generating optimization steps using
    symbolic reasoning and gradient information.
    
    Attributes:
        config: Configuration object
        learning_rate: Learning rate for step generation
        momentum: Momentum coefficient
        last_step: Last step taken
        last_x: Last point evaluated
        last_gradient: Last gradient evaluated
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the symbolic generator.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.learning_rate = self.config.generator_learning_rate
        self.momentum = 0.9
        self.last_step = None
        self.last_x = None
        self.last_gradient = None
        
        logger.debug(f"Initialized SymbolicGenerator with learning_rate={self.learning_rate}, momentum={self.momentum}")
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """Update agent parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        if 'momentum' in params:
            self.momentum = params['momentum']
        
        logger.debug(f"Updated SymbolicGenerator parameters: learning_rate={self.learning_rate}, momentum={self.momentum}")
    
    def estimate_gradient(self, 
                         objective_function: Callable[[np.ndarray], float], 
                         x: np.ndarray, 
                         eps: float = 1e-6) -> np.ndarray:
        """Estimate the gradient of the objective function.
        
        Uses central difference approximation for gradient computation.
        
        Args:
            objective_function: Function to optimize
            x: Point at which to evaluate the gradient
            eps: Step size for finite difference approximation
            
        Returns:
            Gradient vector
        """
        n = len(x)
        grad = np.zeros(n)
        
        f_x = objective_function(x)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            f_plus = objective_function(x_plus)
            f_minus = objective_function(x_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        self.last_x = x.copy()
        self.last_gradient = grad.copy()
        
        return grad
    
    def generate_step(self, 
                     objective_function: Callable[[np.ndarray], float], 
                     x: np.ndarray, 
                     gradient: np.ndarray) -> np.ndarray:
        """Generate an optimization step.
        
        Uses gradient information and momentum to generate a step.
        
        Args:
            objective_function: Function to optimize
            x: Current point
            gradient: Gradient at the current point
            
        Returns:
            Step vector
        """
        # Basic gradient descent with momentum
        step = -self.learning_rate * gradient
        
        # Apply momentum if we have a previous step
        if self.last_step is not None:
            step = step + self.momentum * self.last_step
        
        self.last_step = step.copy()
        self.last_x = x.copy()
        self.last_gradient = gradient.copy()
        
        return step
    
    def symbolic_step(self, 
                     objective_function: Callable[[np.ndarray], float], 
                     x: np.ndarray, 
                     gradient: np.ndarray) -> np.ndarray:
        """Generate a step using symbolic reasoning.
        
        This is a more advanced step generation method that uses
        symbolic reasoning to generate steps that are more likely
        to lead to convergence.
        
        Args:
            objective_function: Function to optimize
            x: Current point
            gradient: Gradient at the current point
            
        Returns:
            Step vector
        """
        # Compute Hessian approximation using BFGS update
        n = len(x)
        if self.last_x is None or self.last_gradient is None:
            # Initialize with identity matrix
            H = np.eye(n)
        else:
            # BFGS update
            s = x - self.last_x
            y = gradient - self.last_gradient
            
            # Ensure s and y are not too small
            if np.linalg.norm(s) < 1e-10 or np.linalg.norm(y) < 1e-10:
                H = np.eye(n)
            else:
                rho = 1.0 / (y @ s)
                
                # Initialize H if needed
                if not hasattr(self, 'H'):
                    self.H = np.eye(n)
                
                H = self.H
                
                # BFGS update formula
                H = (np.eye(n) - rho * np.outer(s, y)) @ H @ (np.eye(n) - rho * np.outer(y, s)) + rho * np.outer(s, s)
                
                # Ensure H is positive definite
                eigvals = np.linalg.eigvals(H)
                if np.any(eigvals <= 0):
                    logger.warning("Hessian approximation is not positive definite, resetting")
                    H = np.eye(n)
        
        # Store H for next iteration
        self.H = H
        
        # Compute step using Newton's method
        step = -np.linalg.solve(H, gradient)
        
        # Apply line search
        alpha = self._line_search(objective_function, x, step, gradient)
        
        # Scale step by line search result
        step = alpha * step
        
        self.last_step = step.copy()
        self.last_x = x.copy()
        self.last_gradient = gradient.copy()
        
        return step
    
    def _line_search(self, 
                    objective_function: Callable[[np.ndarray], float], 
                    x: np.ndarray, 
                    step: np.ndarray, 
                    gradient: np.ndarray) -> float:
        """Perform line search to find optimal step size.
        
        Uses backtracking line search with Armijo condition.
        
        Args:
            objective_function: Function to optimize
            x: Current point
            step: Proposed step
            gradient: Gradient at the current point
            
        Returns:
            Step size
        """
        # Armijo condition parameters
        alpha = 1.0
        c = 0.5
        rho = 0.5
        
        f_x = objective_function(x)
        
        # Compute directional derivative
        directional_derivative = gradient @ step
        
        # Ensure step is a descent direction
        if directional_derivative > 0:
            logger.warning("Step is not a descent direction, reversing")
            step = -step
            directional_derivative = -directional_derivative
        
        # Backtracking line search
        while objective_function(x + alpha * step) > f_x + c * alpha * directional_derivative:
            alpha *= rho
            
            # Prevent too small steps
            if alpha < 1e-10:
                logger.warning("Line search failed, using minimum step size")
                alpha = 1e-10
                break
        
        return alpha


class Orchestrator(Agent):
    """Orchestrator Agent.
    
    This agent is responsible for coordinating the optimization process,
    including step clipping, learning rate adjustment, and multi-objective
    balancing.
    
    Attributes:
        config: Configuration object
        update_frequency: Frequency of orchestrator updates
        step_history: History of steps taken
        gradient_history: History of gradients
        function_value_history: History of function values
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the orchestrator.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.update_frequency = self.config.orchestrator_update_frequency
        self.step_history = []
        self.gradient_history = []
        self.function_value_history = []
        
        logger.debug(f"Initialized Orchestrator with update_frequency={self.update_frequency}")
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """Update agent parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        if 'update_frequency' in params:
            self.update_frequency = params['update_frequency']
        
        logger.debug(f"Updated Orchestrator parameters: update_frequency={self.update_frequency}")
    
    def coordinate_step(self, 
                       step: np.ndarray, 
                       x: np.ndarray, 
                       gradient: np.ndarray, 
                       bounds: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """Coordinate the optimization step.
        
        Args:
            step: Proposed step
            x: Current point
            gradient: Gradient at the current point
            bounds: Bounds for optimization variables
            
        Returns:
            Coordinated step
        """
        # Store history
        self.step_history.append(step.copy())
        self.gradient_history.append(gradient.copy())
        
        # Clip step if too large
        step_norm = np.linalg.norm(step)
        max_step_norm = 1.0
        
        if step_norm > max_step_norm:
            logger.debug(f"Clipping step: {step_norm:.6e} -> {max_step_norm:.6e}")
            step = step * (max_step_norm / step_norm)
        
        # Ensure step respects bounds
        if bounds is not None:
            for i in range(len(x)):
                if x[i] + step[i] < bounds[i][0]:
                    step[i] = bounds[i][0] - x[i]
                elif x[i] + step[i] > bounds[i][1]:
                    step[i] = bounds[i][1] - x[i]
        
        # Adjust step based on history (every update_frequency iterations)
        if len(self.step_history) >= self.update_frequency:
            # Compute average step and gradient over history
            avg_step = np.mean(self.step_history[-self.update_frequency:], axis=0)
            avg_gradient = np.mean(self.gradient_history[-self.update_frequency:], axis=0)
            
            # Check if we're oscillating
            if np.linalg.norm(avg_step) < 0.1 * np.linalg.norm(step) and np.linalg.norm(avg_gradient) > 0.5 * np.linalg.norm(gradient):
                logger.debug("Detected oscillation, adjusting step")
                
                # Dampen oscillation by averaging with previous steps
                step = 0.5 * step + 0.5 * avg_step
        
        return step
    
    def update_weights(self, 
                      function_value: float, 
                      iteration: int) -> Tuple[float, float, float]:
        """Update weights for multi-objective optimization.
        
        Args:
            function_value: Current function value
            iteration: Current iteration
            
        Returns:
            Updated weights (alpha, beta, gamma)
        """
        # Store function value
        self.function_value_history.append(function_value)
        
        # Only update weights periodically
        if iteration % self.update_frequency != 0 or iteration == 0:
            return self.config.alpha, self.config.beta, self.config.gamma
        
        # Check if we're making progress
        if len(self.function_value_history) >= 2 * self.update_frequency:
            recent_values = self.function_value_history[-self.update_frequency:]
            previous_values = self.function_value_history[-2*self.update_frequency:-self.update_frequency]
            
            recent_avg = np.mean(recent_values)
            previous_avg = np.mean(previous_values)
            
            # If we're not making progress, adjust weights
            if recent_avg > 0.9 * previous_avg:
                logger.debug("Not making sufficient progress, adjusting weights")
                
                # Increase weight on validation loss
                alpha = min(0.8, self.config.alpha * 1.1)
                beta = max(0.1, self.config.beta * 0.9)
                gamma = max(0.1, self.config.gamma * 0.9)
                
                # Normalize weights
                total = alpha + beta + gamma
                alpha /= total
                beta /= total
                gamma /= total
                
                logger.debug(f"Updated weights: alpha={alpha}, beta={beta}, gamma={gamma}")
                
                return alpha, beta, gamma
        
        # Default: return original weights
        return self.config.alpha, self.config.beta, self.config.gamma


class ValidationAgent(Agent):
    """Validation Agent.
    
    This agent is responsible for checking convergence and monitoring
    stability during the optimization process.
    
    Attributes:
        config: Configuration object
        validation_threshold: Threshold for convergence validation
        gradient_history: History of gradient norms
        step_history: History of step norms
        function_value_history: History of function values
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the validation agent.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.validation_threshold = self.config.validation_threshold
        self.gradient_history = []
        self.step_history = []
        self.function_value_history = []
        
        logger.debug(f"Initialized ValidationAgent with validation_threshold={self.validation_threshold}")
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """Update agent parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        if 'validation_threshold' in params:
            self.validation_threshold = params['validation_threshold']
        
        logger.debug(f"Updated ValidationAgent parameters: validation_threshold={self.validation_threshold}")
    
    def check_convergence(self, 
                         x: np.ndarray, 
                         function_value: float, 
                         gradient: np.ndarray, 
                         iteration: int, 
                         patience_counter: int) -> bool:
        """Check if the optimization has converged.
        
        Args:
            x: Current point
            function_value: Current function value
            gradient: Gradient at the current point
            iteration: Current iteration
            patience_counter: Number of iterations with no improvement
            
        Returns:
            True if converged, False otherwise
        """
        # Store history
        gradient_norm = np.linalg.norm(gradient)
        self.gradient_history.append(gradient_norm)
        self.function_value_history.append(function_value)
        
        # Check gradient norm convergence
        if gradient_norm < self.config.convergence_tolerance:
            logger.info(f"Converged: gradient norm {gradient_norm:.6e} < tolerance {self.config.convergence_tolerance:.6e}")
            return True
        
        # Check function value convergence
        if len(self.function_value_history) > 1:
            prev_value = self.function_value_history[-2]
            rel_change = abs(function_value - prev_value) / (abs(prev_value) + 1e-10)
            
            if rel_change < self.config.convergence_tolerance:
                logger.info(f"Converged: relative change in function value {rel_change:.6e} < tolerance {self.config.convergence_tolerance:.6e}")
                return True
        
        # Check patience-based convergence
        if patience_counter >= self.config.patience:
            logger.info(f"Converged: no improvement for {patience_counter} iterations")
            return True
        
        # Not converged
        return False
    
    def check_stability(self, 
                       x: np.ndarray, 
                       function_value: float, 
                       gradient: np.ndarray, 
                       iteration: int) -> bool:
        """Check if the optimization is stable.
        
        Args:
            x: Current point
            function_value: Current function value
            gradient: Gradient at the current point
            iteration: Current iteration
            
        Returns:
            True if stable, False otherwise
        """
        # Check for NaN or Inf
        if np.isnan(function_value) or np.isinf(function_value) or np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            logger.warning("Instability detected: NaN or Inf values")
            return False
        
        # Check for extreme function values
        if abs(function_value) > 1e10:
            logger.warning(f"Instability detected: extreme function value {function_value:.6e}")
            return False
        
        # Check for extreme gradient values
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 1e10:
            logger.warning(f"Instability detected: extreme gradient norm {gradient_norm:.6e}")
            return False
        
        # Check for oscillation
        if len(self.gradient_history) > 10:
            recent_gradients = self.gradient_history[-10:]
            if np.std(recent_gradients) > 10 * np.mean(recent_gradients):
                logger.warning("Instability detected: gradient oscillation")
                return False
        
        # Stable
        return True