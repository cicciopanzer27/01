"""
PyTorch integration module for M.I.A.-simbolic.

This module implements the MIAPyTorchOptimizer class, which integrates
the M.I.A.-simbolic optimizer with PyTorch.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Iterator

import numpy as np

from ..config import Config
from ..core.optimizer import MIAOptimizer

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.optim import Optimizer
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not found, PyTorch integration will not be available")
    TORCH_AVAILABLE = False
    # Create dummy Optimizer class for type hints
    class Optimizer:
        pass


if TORCH_AVAILABLE:
    class MIAPyTorchOptimizer(Optimizer):
        """PyTorch optimizer using M.I.A.-simbolic.
        
        This optimizer integrates the M.I.A.-simbolic optimizer with PyTorch,
        allowing it to be used as a drop-in replacement for standard PyTorch
        optimizers.
        
        Attributes:
            params: Model parameters
            lr: Learning rate
            auto_tune: Whether to use Bayesian auto-tuning
            optimizer: M.I.A.-simbolic optimizer
            state: Optimizer state
            param_groups: Parameter groups
        """
        
        def __init__(self, 
                    params: Iterator[torch.Tensor], 
                    lr: float = 0.01, 
                    auto_tune: bool = True, 
                    **kwargs):
            """Initialize the PyTorch optimizer.
            
            Args:
                params: Model parameters
                lr: Learning rate
                auto_tune: Whether to use Bayesian auto-tuning
                **kwargs: Additional parameters for the optimizer
            """
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not found, PyTorch integration is not available")
            
            defaults = dict(lr=lr, auto_tune=auto_tune, **kwargs)
            super(MIAPyTorchOptimizer, self).__init__(params, defaults)
            
            # Create M.I.A.-simbolic optimizer
            config = Config()
            config.generator_learning_rate = lr
            config.auto_tune = auto_tune
            
            # Override config with additional parameters
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            self.optimizer = MIAOptimizer(config=config)
            
            # Initialize state
            self.state['step'] = 0
            self.state['loss'] = None
            self.state['params_vec'] = None
            self.state['grad_vec'] = None
            
            logger.debug(f"Initialized MIAPyTorchOptimizer with lr={lr}, auto_tune={auto_tune}")
        
        def _get_flat_params(self) -> torch.Tensor:
            """Get flattened parameters.
            
            Returns:
                Flattened parameters tensor
            """
            params = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        params.append(p.data.view(-1))
            
            return torch.cat(params)
        
        def _get_flat_grad(self) -> torch.Tensor:
            """Get flattened gradient.
            
            Returns:
                Flattened gradient tensor
            """
            grads = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        grads.append(p.grad.data.view(-1))
            
            return torch.cat(grads)
        
        def _set_flat_params(self, flat_params: torch.Tensor) -> None:
            """Set flattened parameters.
            
            Args:
                flat_params: Flattened parameters tensor
            """
            offset = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        numel = p.numel()
                        p.data.copy_(flat_params[offset:offset + numel].view_as(p.data))
                        offset += numel
        
        def step(self, closure=None) -> Optional[float]:
            """Perform a single optimization step.
            
            Args:
                closure: Closure that reevaluates the model and returns the loss
                
            Returns:
                Loss value
            """
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not found, PyTorch integration is not available")
            
            loss = None
            if closure is not None:
                loss = closure()
            
            # Get current parameters and gradients
            params_vec = self._get_flat_params()
            grad_vec = self._get_flat_grad()
            
            # Convert to numpy
            params_np = params_vec.cpu().numpy()
            grad_np = grad_vec.cpu().numpy()
            
            # Store in state
            self.state['params_vec'] = params_np
            self.state['grad_vec'] = grad_np
            
            # Define objective function for the optimizer
            def objective_function(x: np.ndarray) -> float:
                # Convert to torch tensor
                x_torch = torch.from_numpy(x).to(params_vec.device).type(params_vec.dtype)
                
                # Set parameters
                self._set_flat_params(x_torch)
                
                # Evaluate loss
                if closure is not None:
                    loss_val = closure()
                    return loss_val.item()
                else:
                    return 0.0
            
            # Perform optimization step
            if self.state['step'] == 0:
                # First step: initialize optimizer
                logger.debug("Initializing M.I.A.-simbolic optimizer")
                
                # Run optimization with limited iterations
                result = self.optimizer.optimize(
                    objective_function=objective_function,
                    initial_point=params_np,
                    bounds=None,
                )
                
                # Update parameters
                params_np = result.x
                
                # Convert back to torch tensor
                params_torch = torch.from_numpy(params_np).to(params_vec.device).type(params_vec.dtype)
                
                # Set parameters
                self._set_flat_params(params_torch)
            else:
                # Regular step: use gradient information
                step_np = -self.optimizer.generator.learning_rate * grad_np
                
                # Apply step
                params_np = params_np + step_np
                
                # Convert back to torch tensor
                params_torch = torch.from_numpy(params_np).to(params_vec.device).type(params_vec.dtype)
                
                # Set parameters
                self._set_flat_params(params_torch)
            
            # Increment step counter
            self.state['step'] += 1
            
            return loss
else:
    # Dummy implementation for when PyTorch is not available
    class MIAPyTorchOptimizer:
        """Dummy PyTorch optimizer for when PyTorch is not available."""
        
        def __init__(self, *args, **kwargs):
            """Initialize the dummy optimizer."""
            raise ImportError("PyTorch not found, PyTorch integration is not available")