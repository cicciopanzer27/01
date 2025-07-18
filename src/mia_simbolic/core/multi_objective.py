"""
Multi-objective problem module for M.I.A.-simbolic.

This module implements the MultiObjectiveProblem class, which defines
the structure for multi-objective optimization problems with validation loss,
computational cost, and regularization components.
"""

import time
import logging
import numpy as np
from typing import Callable, Optional, Dict, Any, List, Tuple
from abc import ABC, abstractmethod

from ..config import Config

logger = logging.getLogger(__name__)

class MultiObjectiveProblem(ABC):
    """Abstract base class for multi-objective optimization problems.
    
    This class defines the interface for multi-objective optimization problems
    with validation loss, computational cost, and regularization components.
    
    Attributes:
        config: Configuration object
        alpha: Weight for validation loss component
        beta: Weight for computational cost component
        gamma: Weight for regularization component
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the multi-objective problem.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.alpha = self.config.alpha
        self.beta = self.config.beta
        self.gamma = self.config.gamma
        
        logger.debug(f"Initialized MultiObjectiveProblem with weights: "
                    f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")
    
    @abstractmethod
    def validation_loss(self, x: np.ndarray) -> float:
        """Compute the validation loss component.
        
        Args:
            x: Point at which to evaluate the validation loss
            
        Returns:
            Validation loss value
        """
        pass
    
    @abstractmethod
    def computational_cost(self, x: np.ndarray) -> float:
        """Compute the computational cost component.
        
        Args:
            x: Point at which to evaluate the computational cost
            
        Returns:
            Computational cost value
        """
        pass
    
    @abstractmethod
    def regularization(self, x: np.ndarray) -> float:
        """Compute the regularization component.
        
        Args:
            x: Point at which to evaluate the regularization
            
        Returns:
            Regularization value
        """
        pass
    
    def objective(self, x: np.ndarray) -> float:
        """Compute the combined objective function.
        
        The objective function is a weighted combination of validation loss,
        computational cost, and regularization components.
        
        Args:
            x: Point at which to evaluate the objective
            
        Returns:
            Combined objective value
        """
        start_time = time.time()
        
        # Compute individual components
        val_loss = self.validation_loss(x)
        comp_cost = self.computational_cost(x)
        reg = self.regularization(x)
        
        # Compute weighted combination
        obj_val = self.alpha * val_loss + self.beta * comp_cost + self.gamma * reg
        
        end_time = time.time()
        logger.debug(f"Objective evaluation took {end_time - start_time:.6f} seconds")
        logger.debug(f"Components: val_loss={val_loss:.6e}, comp_cost={comp_cost:.6e}, reg={reg:.6e}")
        logger.debug(f"Combined objective: {obj_val:.6e}")
        
        return obj_val
    
    def gradient(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute the gradient of the objective function.
        
        Uses central difference approximation for gradient computation.
        
        Args:
            x: Point at which to evaluate the gradient
            eps: Step size for finite difference approximation
            
        Returns:
            Gradient vector
        """
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
    
    def update_weights(self, alpha: Optional[float] = None, 
                      beta: Optional[float] = None, 
                      gamma: Optional[float] = None) -> None:
        """Update the weights for the objective components.
        
        Args:
            alpha: New weight for validation loss component
            beta: New weight for computational cost component
            gamma: New weight for regularization component
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
            
        # Normalize weights to sum to 1
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total
        
        logger.info(f"Updated weights: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")


class QuadraticProblem(MultiObjectiveProblem):
    """Quadratic test problem for multi-objective optimization.
    
    This is a simple quadratic problem for testing the multi-objective
    optimization framework.
    
    Attributes:
        dimension: Dimension of the problem
        A: Matrix for quadratic term in validation loss
        b: Vector for linear term in validation loss
        C: Matrix for quadratic term in computational cost
        d: Vector for linear term in computational cost
    """
    
    def __init__(self, dimension: int = 10, config: Optional[Config] = None):
        """Initialize the quadratic problem.
        
        Args:
            dimension: Dimension of the problem
            config: Configuration object
        """
        super().__init__(config)
        self.dimension = dimension
        
        # Generate random problem data
        np.random.seed(42)  # For reproducibility
        self.A = np.random.randn(dimension, dimension)
        self.A = self.A.T @ self.A  # Make positive definite
        self.b = np.random.randn(dimension)
        
        self.C = np.random.randn(dimension, dimension)
        self.C = self.C.T @ self.C  # Make positive definite
        self.d = np.random.randn(dimension)
        
        logger.debug(f"Initialized QuadraticProblem with dimension {dimension}")
    
    def validation_loss(self, x: np.ndarray) -> float:
        """Compute the validation loss component.
        
        Quadratic function: 0.5 * x^T A x + b^T x
        
        Args:
            x: Point at which to evaluate the validation loss
            
        Returns:
            Validation loss value
        """
        return 0.5 * x @ self.A @ x + self.b @ x
    
    def computational_cost(self, x: np.ndarray) -> float:
        """Compute the computational cost component.
        
        Quadratic function: 0.5 * x^T C x + d^T x
        
        Args:
            x: Point at which to evaluate the computational cost
            
        Returns:
            Computational cost value
        """
        return 0.5 * x @ self.C @ x + self.d @ x
    
    def regularization(self, x: np.ndarray) -> float:
        """Compute the regularization component.
        
        L2 regularization: 0.5 * ||x||^2
        
        Args:
            x: Point at which to evaluate the regularization
            
        Returns:
            Regularization value
        """
        return 0.5 * np.sum(x**2)


class NeuralNetworkProblem(MultiObjectiveProblem):
    """Neural network problem for multi-objective optimization.
    
    This class represents a neural network training problem with
    validation loss, computational cost, and regularization components.
    
    Attributes:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use for computation
    """
    
    def __init__(self, model: Any, train_loader: Any, val_loader: Any, 
                criterion: Any, device: str = 'cpu', config: Optional[Config] = None):
        """Initialize the neural network problem.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            device: Device to use for computation
            config: Configuration object
        """
        super().__init__(config)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Get initial parameters
        self.param_shapes = [p.shape for p in self.model.parameters()]
        self.param_sizes = [p.numel() for p in self.model.parameters()]
        self.total_params = sum(self.param_sizes)
        
        logger.info(f"Initialized NeuralNetworkProblem with {self.total_params} parameters")
    
    def _vector_to_parameters(self, x: np.ndarray) -> None:
        """Convert a vector to model parameters.
        
        Args:
            x: Vector of parameters
        """
        import torch
        
        offset = 0
        for i, param in enumerate(self.model.parameters()):
            size = self.param_sizes[i]
            param.data = torch.tensor(
                x[offset:offset+size].reshape(self.param_shapes[i]),
                dtype=param.dtype,
                device=self.device
            )
            offset += size
    
    def _parameters_to_vector(self) -> np.ndarray:
        """Convert model parameters to a vector.
        
        Returns:
            Vector of parameters
        """
        import torch
        
        params = []
        for param in self.model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        
        return np.concatenate(params)
    
    def validation_loss(self, x: np.ndarray) -> float:
        """Compute the validation loss component.
        
        Args:
            x: Vector of model parameters
            
        Returns:
            Validation loss value
        """
        import torch
        
        # Set model parameters
        self._vector_to_parameters(x)
        
        # Evaluate on validation set
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)
    
    def computational_cost(self, x: np.ndarray) -> float:
        """Compute the computational cost component.
        
        Args:
            x: Vector of model parameters
            
        Returns:
            Computational cost value
        """
        import torch
        
        # Set model parameters
        self._vector_to_parameters(x)
        
        # Measure inference time
        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            for inputs, _ in self.val_loader:
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
        end_time = time.time()
        
        # Compute average inference time per batch
        inference_time = (end_time - start_time) / len(self.val_loader)
        
        # Compute FLOPs (approximate)
        flops = self._estimate_flops()
        
        # Normalize and combine
        norm_time = inference_time / 0.1  # Normalize to ~0.1s baseline
        norm_flops = flops / 1e9  # Normalize to ~1 GFLOP baseline
        
        return 0.5 * norm_time + 0.5 * norm_flops
    
    def regularization(self, x: np.ndarray) -> float:
        """Compute the regularization component.
        
        Args:
            x: Vector of model parameters
            
        Returns:
            Regularization value
        """
        # L2 regularization
        l2_reg = np.sum(x**2)
        
        # Sparsity regularization (L1)
        l1_reg = np.sum(np.abs(x))
        
        return 0.5 * l2_reg + 0.5 * l1_reg
    
    def _estimate_flops(self) -> float:
        """Estimate the number of FLOPs for the model.
        
        Returns:
            Estimated number of FLOPs
        """
        # This is a very rough estimate
        # For a more accurate estimate, use a dedicated profiler
        
        total_flops = 0
        
        # Estimate FLOPs for each layer
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    # Linear layer: 2 * in_features * out_features
                    total_flops += 2 * module.in_features * module.out_features
                elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                    # Conv layer: 2 * kernel_size^2 * in_channels * out_channels * output_size
                    if hasattr(module, 'kernel_size'):
                        kernel_size = module.kernel_size
                        if isinstance(kernel_size, tuple):
                            kernel_size = kernel_size[0] * kernel_size[1]
                        
                        # Rough estimate of output size
                        output_size = 32 * 32  # Assume 32x32 feature maps
                        
                        total_flops += 2 * kernel_size * module.in_channels * module.out_channels * output_size
        
        return total_flops