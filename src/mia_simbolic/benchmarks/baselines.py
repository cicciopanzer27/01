"""
Baseline optimizers for M.I.A.-simbolic.

This module implements baseline optimizers for comparison with the
M.I.A.-simbolic optimizer, including standard optimization algorithms
like Adam, SGD, L-BFGS, RMSprop, and Adagrad.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass

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


class BaselineOptimizer:
    """Base class for baseline optimizers.
    
    Attributes:
        name: Name of the optimizer
        max_iterations: Maximum number of iterations
        convergence_tolerance: Convergence tolerance
    """
    
    def __init__(self, 
                name: str, 
                max_iterations: int = 1000, 
                convergence_tolerance: float = 1e-6):
        """Initialize the baseline optimizer.
        
        Args:
            name: Name of the optimizer
            max_iterations: Maximum number of iterations
            convergence_tolerance: Convergence tolerance
        """
        self.name = name
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        
        logger.debug(f"Initialized {self.name} optimizer with max_iterations={max_iterations}, "
                    f"convergence_tolerance={convergence_tolerance}")
    
    def optimize(self, 
                objective_function: Callable[[np.ndarray], float], 
                initial_point: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizationResult:
        """Optimize the objective function.
        
        Args:
            objective_function: Function to optimize
            initial_point: Starting point for optimization
            bounds: Bounds for optimization variables
            
        Returns:
            OptimizationResult object with optimization results
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _estimate_gradient(self, 
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
        
        return grad


class AdamOptimizer(BaselineOptimizer):
    """Adam optimizer.
    
    Attributes:
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, 
                lr: float = 0.001, 
                beta1: float = 0.9, 
                beta2: float = 0.999, 
                epsilon: float = 1e-8, 
                max_iterations: int = 1000, 
                convergence_tolerance: float = 1e-6):
        """Initialize the Adam optimizer.
        
        Args:
            lr: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            max_iterations: Maximum number of iterations
            convergence_tolerance: Convergence tolerance
        """
        super().__init__("Adam", max_iterations, convergence_tolerance)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        logger.debug(f"Initialized Adam optimizer with lr={lr}, beta1={beta1}, beta2={beta2}")
    
    def optimize(self, 
                objective_function: Callable[[np.ndarray], float], 
                initial_point: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizationResult:
        """Optimize the objective function using Adam.
        
        Args:
            objective_function: Function to optimize
            initial_point: Starting point for optimization
            bounds: Bounds for optimization variables
            
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
        
        # Initialize Adam parameters
        m = np.zeros_like(x)  # First moment
        v = np.zeros_like(x)  # Second moment
        
        # Initialize optimization variables
        iteration = 0
        n_func_evals = 1
        f_val = objective_function(x)
        best_x = x.copy()
        best_f_val = f_val
        converged = False
        
        # Initialize gradient information
        gradient = self._estimate_gradient(objective_function, x)
        n_func_evals += 2 * n_dim  # Central difference requires 2n evaluations
        
        # Start optimization loop
        while iteration < self.max_iterations:
            # Update moments
            m = self.beta1 * m + (1 - self.beta1) * gradient
            v = self.beta2 * v + (1 - self.beta2) * gradient**2
            
            # Bias correction
            m_hat = m / (1 - self.beta1**(iteration + 1))
            v_hat = v / (1 - self.beta2**(iteration + 1))
            
            # Compute step
            step = -self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
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
            
            # Update current point
            x = new_x
            f_val = new_f_val
            
            # Update gradient
            gradient = self._estimate_gradient(objective_function, x)
            n_func_evals += 2 * n_dim
            
            # Check convergence
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm < self.convergence_tolerance:
                converged = True
                break
            
            iteration += 1
        
        # Compute final metrics
        end_time = time.time()
        time_taken = end_time - start_time
        gradient_norm = np.linalg.norm(gradient)
        
        # Create result object
        result = OptimizationResult(
            x=best_x,
            fun=best_f_val,
            nit=iteration + 1,
            nfev=n_func_evals,
            time=time_taken,
            converged=converged,
            success=converged or iteration == self.max_iterations - 1,
            message="Optimization converged successfully" if converged else 
                    "Maximum iterations reached without convergence",
            gradient_norm=gradient_norm
        )
        
        return result


class SGDOptimizer(BaselineOptimizer):
    """Stochastic Gradient Descent optimizer.
    
    Attributes:
        lr: Learning rate
        momentum: Momentum coefficient
    """
    
    def __init__(self, 
                lr: float = 0.01, 
                momentum: float = 0.0, 
                max_iterations: int = 1000, 
                convergence_tolerance: float = 1e-6):
        """Initialize the SGD optimizer.
        
        Args:
            lr: Learning rate
            momentum: Momentum coefficient
            max_iterations: Maximum number of iterations
            convergence_tolerance: Convergence tolerance
        """
        super().__init__("SGD", max_iterations, convergence_tolerance)
        self.lr = lr
        self.momentum = momentum
        
        logger.debug(f"Initialized SGD optimizer with lr={lr}, momentum={momentum}")
    
    def optimize(self, 
                objective_function: Callable[[np.ndarray], float], 
                initial_point: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizationResult:
        """Optimize the objective function using SGD.
        
        Args:
            objective_function: Function to optimize
            initial_point: Starting point for optimization
            bounds: Bounds for optimization variables
            
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
        
        # Initialize momentum
        velocity = np.zeros_like(x)
        
        # Initialize optimization variables
        iteration = 0
        n_func_evals = 1
        f_val = objective_function(x)
        best_x = x.copy()
        best_f_val = f_val
        converged = False
        
        # Initialize gradient information
        gradient = self._estimate_gradient(objective_function, x)
        n_func_evals += 2 * n_dim  # Central difference requires 2n evaluations
        
        # Start optimization loop
        while iteration < self.max_iterations:
            # Update velocity
            velocity = self.momentum * velocity - self.lr * gradient
            
            # Take step
            new_x = x + velocity
            
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
            
            # Update current point
            x = new_x
            f_val = new_f_val
            
            # Update gradient
            gradient = self._estimate_gradient(objective_function, x)
            n_func_evals += 2 * n_dim
            
            # Check convergence
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm < self.convergence_tolerance:
                converged = True
                break
            
            iteration += 1
        
        # Compute final metrics
        end_time = time.time()
        time_taken = end_time - start_time
        gradient_norm = np.linalg.norm(gradient)
        
        # Create result object
        result = OptimizationResult(
            x=best_x,
            fun=best_f_val,
            nit=iteration + 1,
            nfev=n_func_evals,
            time=time_taken,
            converged=converged,
            success=converged or iteration == self.max_iterations - 1,
            message="Optimization converged successfully" if converged else 
                    "Maximum iterations reached without convergence",
            gradient_norm=gradient_norm
        )
        
        return result


class LBFGSOptimizer(BaselineOptimizer):
    """L-BFGS optimizer.
    
    Attributes:
        m: Number of corrections to approximate the inverse Hessian matrix
        max_line_search: Maximum number of line search iterations
        c1: Sufficient decrease constant for Armijo condition
        c2: Curvature condition constant
    """
    
    def __init__(self, 
                m: int = 10, 
                max_line_search: int = 20, 
                c1: float = 1e-4, 
                c2: float = 0.9, 
                max_iterations: int = 1000, 
                convergence_tolerance: float = 1e-6):
        """Initialize the L-BFGS optimizer.
        
        Args:
            m: Number of corrections to approximate the inverse Hessian matrix
            max_line_search: Maximum number of line search iterations
            c1: Sufficient decrease constant for Armijo condition
            c2: Curvature condition constant
            max_iterations: Maximum number of iterations
            convergence_tolerance: Convergence tolerance
        """
        super().__init__("L-BFGS", max_iterations, convergence_tolerance)
        self.m = m
        self.max_line_search = max_line_search
        self.c1 = c1
        self.c2 = c2
        
        logger.debug(f"Initialized L-BFGS optimizer with m={m}")
    
    def optimize(self, 
                objective_function: Callable[[np.ndarray], float], 
                initial_point: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizationResult:
        """Optimize the objective function using L-BFGS.
        
        Args:
            objective_function: Function to optimize
            initial_point: Starting point for optimization
            bounds: Bounds for optimization variables
            
        Returns:
            OptimizationResult object with optimization results
        """
        # Try to use scipy's L-BFGS-B implementation if available
        try:
            from scipy.optimize import minimize
            
            start_time = time.time()
            
            # Set up bounds for scipy
            scipy_bounds = None
            if bounds is not None:
                scipy_bounds = bounds
            
            # Run optimization
            result = minimize(
                objective_function,
                initial_point,
                method='L-BFGS-B',
                bounds=scipy_bounds,
                options={
                    'maxiter': self.max_iterations,
                    'gtol': self.convergence_tolerance,
                    'maxls': self.max_line_search
                }
            )
            
            end_time = time.time()
            time_taken = end_time - start_time
            
            # Convert scipy result to our format
            gradient = self._estimate_gradient(objective_function, result.x)
            gradient_norm = np.linalg.norm(gradient)
            
            return OptimizationResult(
                x=result.x,
                fun=result.fun,
                nit=result.nit,
                nfev=result.nfev,
                time=time_taken,
                converged=result.success,
                success=result.success,
                message=result.message,
                gradient_norm=gradient_norm
            )
        
        except ImportError:
            logger.warning("scipy not found, using custom L-BFGS implementation")
            
            # Custom implementation (simplified)
            start_time = time.time()
            
            # Initialize optimization state
            x = np.array(initial_point, dtype=float)
            n_dim = len(x)
            
            # Set up bounds if provided
            if bounds is None:
                bounds = [(-np.inf, np.inf)] * n_dim
            
            # Initialize L-BFGS parameters
            s_list = []  # List of s vectors
            y_list = []  # List of y vectors
            rho_list = []  # List of rho values
            
            # Initialize optimization variables
            iteration = 0
            n_func_evals = 1
            f_val = objective_function(x)
            best_x = x.copy()
            best_f_val = f_val
            converged = False
            
            # Initialize gradient information
            gradient = self._estimate_gradient(objective_function, x)
            n_func_evals += 2 * n_dim  # Central difference requires 2n evaluations
            
            # Start optimization loop
            while iteration < self.max_iterations:
                # Compute search direction using L-BFGS two-loop recursion
                q = gradient.copy()
                alpha_list = []
                
                # First loop
                for i in range(len(s_list) - 1, -1, -1):
                    alpha = rho_list[i] * np.dot(s_list[i], q)
                    alpha_list.append(alpha)
                    q = q - alpha * y_list[i]
                
                # Initialize Hessian approximation
                if len(s_list) > 0:
                    s = s_list[-1]
                    y = y_list[-1]
                    gamma = np.dot(s, y) / np.dot(y, y)
                    H0 = gamma * np.eye(n_dim)
                else:
                    H0 = np.eye(n_dim)
                
                # Apply Hessian approximation
                r = H0 @ q
                
                # Second loop
                for i in range(len(s_list)):
                    beta = rho_list[i] * np.dot(y_list[i], r)
                    r = r + s_list[i] * (alpha_list[len(s_list) - 1 - i] - beta)
                
                # Search direction
                direction = -r
                
                # Line search
                alpha = 1.0
                new_x = x + alpha * direction
                
                # Clip to bounds
                for i in range(n_dim):
                    new_x[i] = max(bounds[i][0], min(bounds[i][1], new_x[i]))
                
                # Evaluate new point
                new_f_val = objective_function(new_x)
                n_func_evals += 1
                
                # Simple backtracking line search
                line_search_iter = 0
                while new_f_val > f_val + self.c1 * alpha * np.dot(gradient, direction) and line_search_iter < self.max_line_search:
                    alpha *= 0.5
                    new_x = x + alpha * direction
                    
                    # Clip to bounds
                    for i in range(n_dim):
                        new_x[i] = max(bounds[i][0], min(bounds[i][1], new_x[i]))
                    
                    new_f_val = objective_function(new_x)
                    n_func_evals += 1
                    line_search_iter += 1
                
                # Update best point if improved
                if new_f_val < best_f_val:
                    best_x = new_x.copy()
                    best_f_val = new_f_val
                
                # Compute new gradient
                new_gradient = self._estimate_gradient(objective_function, new_x)
                n_func_evals += 2 * n_dim
                
                # Update L-BFGS memory
                s = new_x - x
                y = new_gradient - gradient
                
                # Skip update if s or y is too small
                if np.linalg.norm(s) > 1e-8 and np.linalg.norm(y) > 1e-8:
                    rho = 1.0 / np.dot(y, s)
                    
                    # Store vectors
                    s_list.append(s)
                    y_list.append(y)
                    rho_list.append(rho)
                    
                    # Limit memory
                    if len(s_list) > self.m:
                        s_list.pop(0)
                        y_list.pop(0)
                        rho_list.pop(0)
                
                # Update current point
                x = new_x
                f_val = new_f_val
                gradient = new_gradient
                
                # Check convergence
                gradient_norm = np.linalg.norm(gradient)
                if gradient_norm < self.convergence_tolerance:
                    converged = True
                    break
                
                iteration += 1
            
            # Compute final metrics
            end_time = time.time()
            time_taken = end_time - start_time
            gradient_norm = np.linalg.norm(gradient)
            
            # Create result object
            result = OptimizationResult(
                x=best_x,
                fun=best_f_val,
                nit=iteration + 1,
                nfev=n_func_evals,
                time=time_taken,
                converged=converged,
                success=converged or iteration == self.max_iterations - 1,
                message="Optimization converged successfully" if converged else 
                        "Maximum iterations reached without convergence",
                gradient_norm=gradient_norm
            )
            
            return result


class RMSpropOptimizer(BaselineOptimizer):
    """RMSprop optimizer.
    
    Attributes:
        lr: Learning rate
        decay: Decay rate for moving average
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, 
                lr: float = 0.01, 
                decay: float = 0.9, 
                epsilon: float = 1e-8, 
                max_iterations: int = 1000, 
                convergence_tolerance: float = 1e-6):
        """Initialize the RMSprop optimizer.
        
        Args:
            lr: Learning rate
            decay: Decay rate for moving average
            epsilon: Small constant for numerical stability
            max_iterations: Maximum number of iterations
            convergence_tolerance: Convergence tolerance
        """
        super().__init__("RMSprop", max_iterations, convergence_tolerance)
        self.lr = lr
        self.decay = decay
        self.epsilon = epsilon
        
        logger.debug(f"Initialized RMSprop optimizer with lr={lr}, decay={decay}")
    
    def optimize(self, 
                objective_function: Callable[[np.ndarray], float], 
                initial_point: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizationResult:
        """Optimize the objective function using RMSprop.
        
        Args:
            objective_function: Function to optimize
            initial_point: Starting point for optimization
            bounds: Bounds for optimization variables
            
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
        
        # Initialize RMSprop parameters
        v = np.zeros_like(x)  # Moving average of squared gradients
        
        # Initialize optimization variables
        iteration = 0
        n_func_evals = 1
        f_val = objective_function(x)
        best_x = x.copy()
        best_f_val = f_val
        converged = False
        
        # Initialize gradient information
        gradient = self._estimate_gradient(objective_function, x)
        n_func_evals += 2 * n_dim  # Central difference requires 2n evaluations
        
        # Start optimization loop
        while iteration < self.max_iterations:
            # Update moving average
            v = self.decay * v + (1 - self.decay) * gradient**2
            
            # Compute step
            step = -self.lr * gradient / (np.sqrt(v) + self.epsilon)
            
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
            
            # Update current point
            x = new_x
            f_val = new_f_val
            
            # Update gradient
            gradient = self._estimate_gradient(objective_function, x)
            n_func_evals += 2 * n_dim
            
            # Check convergence
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm < self.convergence_tolerance:
                converged = True
                break
            
            iteration += 1
        
        # Compute final metrics
        end_time = time.time()
        time_taken = end_time - start_time
        gradient_norm = np.linalg.norm(gradient)
        
        # Create result object
        result = OptimizationResult(
            x=best_x,
            fun=best_f_val,
            nit=iteration + 1,
            nfev=n_func_evals,
            time=time_taken,
            converged=converged,
            success=converged or iteration == self.max_iterations - 1,
            message="Optimization converged successfully" if converged else 
                    "Maximum iterations reached without convergence",
            gradient_norm=gradient_norm
        )
        
        return result


class AdagradOptimizer(BaselineOptimizer):
    """Adagrad optimizer.
    
    Attributes:
        lr: Learning rate
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, 
                lr: float = 0.01, 
                epsilon: float = 1e-8, 
                max_iterations: int = 1000, 
                convergence_tolerance: float = 1e-6):
        """Initialize the Adagrad optimizer.
        
        Args:
            lr: Learning rate
            epsilon: Small constant for numerical stability
            max_iterations: Maximum number of iterations
            convergence_tolerance: Convergence tolerance
        """
        super().__init__("Adagrad", max_iterations, convergence_tolerance)
        self.lr = lr
        self.epsilon = epsilon
        
        logger.debug(f"Initialized Adagrad optimizer with lr={lr}")
    
    def optimize(self, 
                objective_function: Callable[[np.ndarray], float], 
                initial_point: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizationResult:
        """Optimize the objective function using Adagrad.
        
        Args:
            objective_function: Function to optimize
            initial_point: Starting point for optimization
            bounds: Bounds for optimization variables
            
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
        
        # Initialize Adagrad parameters
        G = np.zeros_like(x)  # Sum of squared gradients
        
        # Initialize optimization variables
        iteration = 0
        n_func_evals = 1
        f_val = objective_function(x)
        best_x = x.copy()
        best_f_val = f_val
        converged = False
        
        # Initialize gradient information
        gradient = self._estimate_gradient(objective_function, x)
        n_func_evals += 2 * n_dim  # Central difference requires 2n evaluations
        
        # Start optimization loop
        while iteration < self.max_iterations:
            # Update sum of squared gradients
            G += gradient**2
            
            # Compute step
            step = -self.lr * gradient / (np.sqrt(G) + self.epsilon)
            
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
            
            # Update current point
            x = new_x
            f_val = new_f_val
            
            # Update gradient
            gradient = self._estimate_gradient(objective_function, x)
            n_func_evals += 2 * n_dim
            
            # Check convergence
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm < self.convergence_tolerance:
                converged = True
                break
            
            iteration += 1
        
        # Compute final metrics
        end_time = time.time()
        time_taken = end_time - start_time
        gradient_norm = np.linalg.norm(gradient)
        
        # Create result object
        result = OptimizationResult(
            x=best_x,
            fun=best_f_val,
            nit=iteration + 1,
            nfev=n_func_evals,
            time=time_taken,
            converged=converged,
            success=converged or iteration == self.max_iterations - 1,
            message="Optimization converged successfully" if converged else 
                    "Maximum iterations reached without convergence",
            gradient_norm=gradient_norm
        )
        
        return result